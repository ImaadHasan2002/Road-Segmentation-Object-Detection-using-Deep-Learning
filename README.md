# Road-Segmentation-Object-Detection-using-Deep-Learning
In this project, a deep learning-based approach is used for lane detection on semi-urban roads. The proposed model consists of two main components: a CNN architecture, ResNet101, for semantic segmentation to accurately detect and classify road features, and YOLOv8 for object detection.
## Abstract
Autonomous vehicles require a precise and accurate perception of their surroundings to
navigate safely. In order for the vehicle to comprehend its position and trajectory on the
road, lane detection is a critical component of this perception. This can be particularly
difficult in semi-urban environments in which there are both residential and commercial
areas and a variety of road types and conditions.
In this project, a deep learning-based approach is used for lane detection on semi-urban
roads. The proposed model consists of two main components: a CNN architecture,
ResNet101, for semantic segmentation to accurately detect and classify road features, and
YOLOv8 for object detection. This approach aims to handle the variability and
complexity of semi-urban road environments, such as road surface textures.
A dataset of semi-urban road images is used for training the model to track lanes, other
vehicles, and obstacles in real time. The ResNet101 model is trained for semantic
segmentation, which helps in understanding the road environment and identifying
different road types and conditions. YOLOv8, on the other hand, is used for object
detection, which helps in localizing and classifying objects in the scene, such as vehicles
and pedestrians.
The two models allow the autonomous vehicle to have a precise and accurate perception
of its surroundings, enabling it to navigate safely in semi-urban environments with
various road types and conditions.
![image](https://github.com/ImaadHasan2002/Road-Segmentation-Object-Detection-using-Deep-Learning/assets/114683650/9a63aae2-c362-48ff-bebb-01d036e5d1e6)
**End to End semantic segmentation based drivable road detection on unstructured roads for self driving cars.**
## Description
A PyTorch implementation of drivable road region detection and steering angle estimation methods proposed for the autonomous vehicle. The dataset has been taken from [CARL-DATASET](https://carl-dataset.github.io/index/)
![147745756-f0d18207-a9f5-4b88-872b-73b80f1d3731](https://github.com/ImaadHasan2002/Road-Segmentation-Object-Detection-using-Deep-Learning/assets/114683650/cb626e48-8150-4f79-a991-d1d617c2597c)
![147745840-9df41a44-f05c-4ce5-be18-3a3c221327e4](https://github.com/ImaadHasan2002/Road-Segmentation-Object-Detection-using-Deep-Learning/assets/114683650/65ff9ab9-d116-4fb4-bdcf-256395fb3d66)
## Prerequistes
1. Windows
2. Anaconda Python(Spyder)
3. PyTorch==2.0.1+cu118
4. Albumentations==1.0.3
5. Tensorboard==1.6.0
6. TensorboardX==2.2
7. Ultralytics
   
## About The Dataset

The CARL-Dataset – a benchmark designed for drivable road region detection on unstructured roads, essential for enabling autonomous driving in the absence of lane markings.

### Key Features:

- **15,000 Fine-Annotated Images**
  A rich collection of meticulously annotated images to enhance the robustness of your drivable road region detection models.

- **Diverse Road Types**
  The dataset encompasses a variety of road types, ensuring your model is well-equipped to handle different driving scenarios.

- **Binary Classes**
  - (i) Road
  - (ii) Background
  Simplifying the task with a binary classification: distinguishing between road and background.

- **COCO Format Annotations**
  Annotations are structured in COCO format, providing a standardized and widely used format for seamless integration into your computer vision projects.

- **Polygonal Annotations**
  Annotations in the form of polygons, allowing for precise delineation of drivable road regions.

### Drivable Road Segmentation

### Object Detection Dataset Integration
We have seamlessly integrated our own dataset, enhancing the diversity and complexity of object detection scenarios.

### PotholeDataset Inclusion
To further enrich your model's understanding, we've incorporated the PotholeDataset, ensuring drivable road region detection system is adept at identifying potential road hazards.

## Getting Started

### Directory Structure:

- **CARL_DATASET:**
  - *test*
  - *test_label*
  - *train*
  - *train_label*
  - *val*
  - *val_label*

- **runs:**
  Segmentation logs of the validation data.

- **train_seg_maps:**
  Save predictions of the trained model on randomly selected images during the validation process.

- **utils:**
  - *helper.py*
  - *metrics.py*

- **config.py:**
  Configuration file containing dataset path, name, and number of classes. Adjust other parameters as needed.

- **dataset.py:**
  Functions related to the dataset.

- **helper.py:**
  Helper functions for the project.

- **train.py:**
  Used to train the model.

- **road_detection_test.py:**
  Test the model on randomly selected images.

- **test_vid.py:**
  Detect driveable road regions in a given video input.

- **model.py:**
  Train the model from scratch.

- **engine.py:**
  Save the model as a *model.pth*.

- **metrics.py:**
  Helper functions to measure performance metrics.
  
- **yolov8.py:**
  implementation for real-time object detection with seamless integration, training support, and customization options.

Feel free to explore, experiment, and contribute to advancing drivable road region detection for autonomous driving! 🚗💨
## How to Run the Python Scripts

### For Training
1. Train the model for the first time on the road detection dataset CARL-DATASET.
2. Before starting training, place the downloaded dataset folders (train, train_labels, test, test_labels, val, and val_labels) in the CARL-Dataset directory.
3. After completion, verify the root paths and other configurations in 'config.py'.
4. Run the following command:
    ```bash
    python train.py --resume-training no
    ```

### For Testing (Image)
- Test the model on an image using the following command:
    ```bash
    python test_road_detection.py --model-path <path to saved checkpoint/weight file> --input <path to image>
    ```
    Example:
    ```bash
    python test_road_detection.py --model-path model.pth --input abc.jpg
    ```

### For Testing (Video)
- Test the model on a video using the following command:
    ```bash
    python test_vid.py --input <path to video> --model-path <path to saved checkpoint/weight file>
    ```
    Example:
    ```bash
    python test_vid.py --input DSC_0006.mp4 --model-path model.pth
    ```

Make sure to adjust paths and filenames according to the specific setup.
