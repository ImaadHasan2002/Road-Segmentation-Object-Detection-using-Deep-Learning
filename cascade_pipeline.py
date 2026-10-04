"""
Cascade perception pipeline: drivable-road segmentation -> YOLO object detection.

Stage 1  FCN-ResNet101 predicts a binary drivable-road mask for the frame.
Stage 2  The road mask defines a region of interest (ROI). YOLO (any Ultralytics
         detection or instance-segmentation model, e.g. yolov8n.pt / yolov8n-seg.pt)
         runs only on that ROI, which removes irrelevant scenery (sky, buildings)
         and gives small, distant objects more pixels at the detector input.
Stage 3  Every detection is fused with the road mask: the ground-contact region
         of the object (bottom strip of its box, or of its instance mask when a
         -seg model is used) is intersected with the road to decide whether the
         object is on the drivable surface.

Usage:
    python cascade_pipeline.py --input testimg.jpg --seg-weights model.pth
    python cascade_pipeline.py --input DSC_0006.mp4 --seg-weights model.pth \
        --yolo-weights yolov8n-seg.pt --show
    python cascade_pipeline.py --input 0 --seg-weights model.pth   # webcam
"""
import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# normalisation statistics used when training the segmentation model
SEG_MEAN = np.array([0.45734706, 0.43338275, 0.40058118], dtype=np.float32)
SEG_STD = np.array([0.23965294, 0.23532275, 0.2398498], dtype=np.float32)
ROAD_CLASS_INDEX = 0  # `road` is label 0, `background` is label 1 (see utils/helpers.py)

ROAD_COLOR = (74, 163, 17)       # BGR version of the Road colour in utils/helpers.py
ON_ROAD_COLOR = (0, 0, 255)      # red: object on the drivable surface
OFF_ROAD_COLOR = (180, 180, 180)  # grey: object beside the road
ROI_COLOR = (255, 200, 0)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


@dataclass
class CascadeConfig:
    seg_weights: str
    yolo_weights: str = 'yolov8n.pt'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seg_input_size: Optional[int] = None  # shorter side for segmentation; None = native
    yolo_imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    classes: Optional[List[int]] = None   # restrict YOLO to these class ids
    use_roi: bool = True                  # gate YOLO with the road ROI
    roi_padding: float = 0.15             # ROI growth as a fraction of frame size
    roi_min_road_fraction: float = 0.02   # below this, fall back to the full frame
    contact_fraction: float = 0.2         # bottom part of a box treated as ground contact
    on_road_threshold: float = 0.3        # min road overlap of the contact region
    keep_off_road: bool = True            # keep detections that are not on the road
    overlay_alpha: float = 0.4


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 in full-frame pixels
    confidence: float
    class_id: int
    class_name: str
    road_overlap: float = 0.0
    on_road: bool = False
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # full-frame bool mask


@dataclass
class CascadeResult:
    road_mask: np.ndarray                 # HxW bool
    roi: Tuple[int, int, int, int]        # x1, y1, x2, y2 the detector saw
    detections: List[Detection]
    timings: dict


def build_segmentation_model(num_classes=2):
    """FCN-ResNet101 with the same heads as `model.py`, without downloading weights."""
    seg_model = models.segmentation.fcn_resnet101(
        weights=None, weights_backbone=None, aux_loss=True)
    seg_model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    return seg_model


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no `weights_only`
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint


class RoadSegmenter:
    """Stage 1: predicts the drivable-road mask."""

    def __init__(self, weights, device, input_size=None):
        self.device = device
        self.input_size = input_size
        self.model = build_segmentation_model()
        state_dict = load_checkpoint(weights, device)
        # `strict=False` lets checkpoints saved without the auxiliary head load too
        missing, _ = self.model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith('aux_classifier')]
        if missing:
            raise RuntimeError(f'Segmentation checkpoint is missing weights: {missing[:5]}')
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.input_size:
            scale = self.input_size / min(height, width)
            image = cv2.resize(image, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_LINEAR)
        image = (image.astype(np.float32) / 255.0 - SEG_MEAN) / SEG_STD
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)['out']
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode='bilinear',
                                   align_corners=False)
        labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        return labels == ROAD_CLASS_INDEX


class ObjectDetector:
    """Stage 2: thin wrapper around an Ultralytics YOLO model."""

    def __init__(self, weights, device, imgsz=640, conf=0.25, iou=0.45, classes=None):
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError('The cascade pipeline needs `ultralytics`: '
                              'pip install ultralytics') from error
        self.model = YOLO(weights)
        self.names = self.model.names
        self.kwargs = dict(device=device, imgsz=imgsz, conf=conf, iou=iou,
                           classes=classes, verbose=False)

    def __call__(self, image_bgr, offset=(0, 0), frame_shape=None):
        """Detect objects in `image_bgr` and map them back to full-frame coordinates."""
        result = self.model.predict(image_bgr, **self.kwargs)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        off_x, off_y = offset
        crop_h, crop_w = image_bgr.shape[:2]
        frame_h, frame_w = frame_shape if frame_shape else (crop_h, crop_w)
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        # instance-mask polygons (only for *-seg models), already in crop pixel coordinates
        polygons = result.masks.xy if result.masks is not None else None

        detections = []
        for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
            x1, y1, x2, y2 = box
            full_box = (int(x1 + off_x), int(y1 + off_y), int(x2 + off_x), int(y2 + off_y))
            full_mask = None
            if polygons is not None and len(polygons[i]):
                full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                points = np.round(polygons[i] + (off_x, off_y)).astype(np.int32)
                cv2.fillPoly(full_mask, [points], 1)
                full_mask = full_mask.astype(bool)
            detections.append(Detection(
                box=full_box, confidence=float(conf), class_id=int(class_id),
                class_name=str(self.names[int(class_id)]), mask=full_mask))
        return detections


def road_roi(road_mask, padding, min_fraction):
    """
    Bounding box of the road, grown by `padding`. The top is grown further than the
    sides because vehicles and pedestrians stand *above* the road pixels they touch.
    Returns the full frame when too little road is visible to trust the mask.
    """
    height, width = road_mask.shape
    full = (0, 0, width, height)
    if road_mask.mean() < min_fraction:
        return full
    ys, xs = np.nonzero(road_mask)
    pad_x, pad_y = int(padding * width), int(padding * height)
    x1 = max(0, xs.min() - pad_x)
    x2 = min(width, xs.max() + 1 + pad_x)
    y1 = max(0, ys.min() - 2 * pad_y)
    y2 = min(height, ys.max() + 1 + pad_y)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return full
    return int(x1), int(y1), int(x2), int(y2)


def road_overlap(detection, road_mask, contact_fraction):
    """Fraction of the object's ground-contact region that lies on the road."""
    height, width = road_mask.shape
    x1, y1, x2, y2 = detection.box
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    contact_top = int(y2 - max(1, (y2 - y1) * contact_fraction))
    contact_road = road_mask[contact_top:y2, x1:x2]
    if detection.mask is not None:
        contact_obj = detection.mask[contact_top:y2, x1:x2]
        if contact_obj.any():
            return float((contact_road & contact_obj).sum() / contact_obj.sum())
    return float(contact_road.mean())


class CascadePipeline:
    """Runs segmentation -> ROI-gated detection -> road-aware fusion on BGR frames."""

    def __init__(self, cfg: CascadeConfig):
        self.cfg = cfg
        self.segmenter = RoadSegmenter(cfg.seg_weights, cfg.device, cfg.seg_input_size)
        self.detector = ObjectDetector(cfg.yolo_weights, cfg.device, cfg.yolo_imgsz,
                                       cfg.conf, cfg.iou, cfg.classes)

    def __call__(self, frame_bgr) -> CascadeResult:
        cfg = self.cfg
        timings = {}
        start = time.perf_counter()
        road_mask = self.segmenter(frame_bgr)
        timings['segmentation'] = time.perf_counter() - start

        height, width = road_mask.shape
        if cfg.use_roi:
            roi = road_roi(road_mask, cfg.roi_padding, cfg.roi_min_road_fraction)
        else:
            roi = (0, 0, width, height)
        x1, y1, x2, y2 = roi

        start = time.perf_counter()
        detections = self.detector(frame_bgr[y1:y2, x1:x2], offset=(x1, y1),
                                   frame_shape=(height, width))
        timings['detection'] = time.perf_counter() - start

        start = time.perf_counter()
        for det in detections:
            det.road_overlap = road_overlap(det, road_mask, cfg.contact_fraction)
            det.on_road = det.road_overlap >= cfg.on_road_threshold
        if not cfg.keep_off_road:
            detections = [det for det in detections if det.on_road]
        timings['fusion'] = time.perf_counter() - start
        return CascadeResult(road_mask, roi, detections, timings)


def draw_result(frame_bgr, result: CascadeResult, alpha=0.4, draw_roi=True):
    canvas = frame_bgr.copy()
    color_layer = canvas.copy()
    color_layer[result.road_mask] = ROAD_COLOR
    for det in result.detections:
        if det.mask is not None:
            color_layer[det.mask] = ON_ROAD_COLOR if det.on_road else OFF_ROAD_COLOR
    canvas = cv2.addWeighted(color_layer, alpha, canvas, 1 - alpha, 0)

    if draw_roi and result.roi != (0, 0, canvas.shape[1], canvas.shape[0]):
        x1, y1, x2, y2 = result.roi
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), ROI_COLOR, 1, cv2.LINE_AA)

    for det in result.detections:
        color = ON_ROAD_COLOR if det.on_road else OFF_ROAD_COLOR
        x1, y1, x2, y2 = det.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f'{det.class_name} {det.confidence:.2f}'
        label += ' | on road' if det.on_road else ''
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y1, text_h + baseline)
        cv2.rectangle(canvas, (x1, text_y - text_h - baseline), (x1 + text_w, text_y),
                      color, -1)
        cv2.putText(canvas, label, (x1, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    on_road = sum(det.on_road for det in result.detections)
    total_ms = 1000 * sum(result.timings.values())
    summary = (f'road {100 * result.road_mask.mean():.1f}% | objects {len(result.detections)} '
               f'(on road {on_road}) | {total_ms:.0f} ms')
    cv2.putText(canvas, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def run_on_image(pipeline, path, output_dir, show, alpha):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    result = pipeline(frame)
    canvas = draw_result(frame, result, alpha)
    save_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(path))[0]}_cascade.jpg')
    cv2.imwrite(save_path, canvas)
    for det in result.detections:
        print(f'{det.class_name:>12s}  conf={det.confidence:.2f}  '
              f'road_overlap={det.road_overlap:.2f}  on_road={det.on_road}  box={det.box}')
    print(f'Saved {save_path}')
    if show:
        cv2.imshow('Cascade', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(pipeline, source, output_dir, show, alpha):
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise FileNotFoundError(f'Could not open video source: {source}')
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    name = 'webcam' if source.isdigit() else os.path.splitext(os.path.basename(source))[0]
    save_path = os.path.join(output_dir, f'{name}_cascade.mp4')
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count, total_time = 0, 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            result = pipeline(frame)
            total_time += time.perf_counter() - start
            frame_count += 1
            canvas = draw_result(frame, result, alpha)
            writer.write(canvas)
            if show:
                cv2.imshow('Cascade', canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    if frame_count:
        print(f'Processed {frame_count} frames, average {frame_count / total_time:.2f} FPS')
    print(f'Saved {save_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Road segmentation -> YOLO cascade pipeline')
    parser.add_argument('-i', '--input', required=True,
                        help='image, video, or webcam index (e.g. 0)')
    parser.add_argument('-m', '--seg-weights', required=True,
                        help='segmentation checkpoint saved by train.py (model.pth)')
    parser.add_argument('-y', '--yolo-weights', default='yolov8n.pt',
                        help='Ultralytics weights; use a *-seg.pt model for mask-level fusion')
    parser.add_argument('-o', '--output-dir', default='outputs')
    parser.add_argument('--device', default=None, help='cuda, cuda:0, or cpu')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='YOLO NMS IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO input size')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='only detect these YOLO class ids (e.g. 0 2 3 5 7)')
    parser.add_argument('--seg-size', type=int, default=None,
                        help='resize the shorter side to this before segmentation')
    parser.add_argument('--no-roi', action='store_true',
                        help='run YOLO on the full frame instead of the road ROI')
    parser.add_argument('--roi-padding', type=float, default=0.15)
    parser.add_argument('--on-road-threshold', type=float, default=0.3)
    parser.add_argument('--on-road-only', action='store_true',
                        help='discard detections that are not on the drivable road')
    parser.add_argument('--alpha', type=float, default=0.4, help='overlay transparency')
    parser.add_argument('--show', action='store_true', help='display results in a window')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CascadeConfig(
        seg_weights=args.seg_weights,
        yolo_weights=args.yolo_weights,
        seg_input_size=args.seg_size,
        yolo_imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        use_roi=not args.no_roi,
        roi_padding=args.roi_padding,
        on_road_threshold=args.on_road_threshold,
        keep_off_road=not args.on_road_only,
        overlay_alpha=args.alpha,
    )
    if args.device:
        cfg.device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = CascadePipeline(cfg)
    if args.input.lower().endswith(IMAGE_EXTENSIONS):
        run_on_image(pipeline, args.input, args.output_dir, args.show, cfg.overlay_alpha)
    else:
        run_on_video(pipeline, args.input, args.output_dir, args.show, cfg.overlay_alpha)


if __name__ == '__main__':
    main()
