# Off-Road Image Segmentation Project

This repository contains the implementation of a complete semantic segmentation pipeline designed to classify off-road terrain features. The project focuses on identifying drivable paths, vegetation types, sky, and obstacles in complex outdoor environments to support autonomous ground vehicle navigation.

---

## 🎯 Project Overview

In off-road autonomous driving, path detection and obstacle avoidance are major challenges due to unstructured terrain, varying illumination, and the absence of clear lane markings. This project addresses these challenges by developing, training, and validating a deep learning semantic segmentation model in PyTorch. 

The pipeline includes:
* **Dataset Cleaning**: Identification and tracking of mislabeled or noisy samples.
* **Data Augmentation**: Advanced transformations via Albumentations to improve model generalization.
* **Model Selection**: A DeepLabV3 architecture with a MobileNetV3-Large backbone, customized for multi-class terrain segmentation.
* **Training Strategy**: 5-Fold Cross-Validation using a hybrid loss function (Dice + Focal Loss) to address severe class imbalance.
* **Monitoring & Evaluation**: Metric tracking via TensorBoard and testing on unseen samples using Mean Intersection over Union (mIoU).

---

## 🛠️ Technology Stack & Libraries

* **Deep Learning Framework**: [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/stable/index.html)
* **Image Augmentations**: [Albumentations](https://albumentations.ai/)
* **Visualization & Logging**: [TensorBoard](https://www.tensorflow.org/tensorboard)
* **Data Handling & Math**: NumPy, Pandas, PIL (Pillow), OpenCV
* **Development Environment**: Jupyter Notebooks / Google Colab (with T4 GPU acceleration)

---

## 📁 Repository Structure

```text
.
├── resources/
│   ├── train/                             # Folder containing training dataset samples
│   └── test/                              # Folder containing test dataset samples (evaluated via testing notebook)
│
├── runs/                                  # TensorBoard logs and metric graphs generated per training fold
│
├── Project_Work_Training.ipynb            # Jupyter Notebook containing dataset splitting, augmentation, and 5-Fold CV training
│
├── Project_Work_Testing.ipynb             # Jupyter Notebook containing testing loop, loading routine, inference, and metric scoring
│
├── Incorrect_Sample_List.csv              # List of dataset sample directories identified as mislabeled or noisy
│
├── best_segmentation_model.pth            # The saved PyTorch state dictionary representing the best-performing trained model weights
│
├── Project_Documentation.pdf              # Comprehensive academic documentation detailing the project
│
└── Project_Presentation.pptx              # Presentation slides explaining the project results
```

---

## 📊 Dataset & Class Mappings

The dataset consists of rgb images (`rgb.jpg`) paired with ground-truth semantic mask labels (`labels.png`). The target labels map specific RGB color codes to class IDs:

| Class ID | Class Name | Mask RGB Color | Navigation Classification | Description |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Background | `(255, 255, 255)` | Non-drivable / Excluded | Background area / Regions of no interest |
| **1** | Smooth Trail | `(178, 176, 153)` | Drivable (High Quality) | Flat dirt path, gravel roads, or smooth terrain |
| **2** | Traversable Grass | `(128, 255, 0)` | Drivable (Medium Quality)| Flat or low grass lawns suitable for driving |
| **3** | Rough Trail | `(156, 76, 30)` | Drivable (Low Quality) | Rocky, bumpy, or uneven trails |
| **4** | Puddle | `(255, 0, 128)` | Hazard | Muddy or watery pools on the trail |
| **5** | Obstacle | `(255, 0, 0)` | Hazard | Rocks, tree stumps, fences, or hard blockages |
| **6** | Non-Traversable Low Veg. | `(0, 160, 0)` | Obstacle / Blocked | Dense bushes, tall weeds, or thick scrub |
| **7** | High Vegetation | `(40, 80, 0)` | Obstacle / Blocked | Large trees, dense forests, or canopy |
| **8** | Sky | `(1, 88, 255)` | Non-drivable | Sky region |

### 🧼 Data Cleaning
To ensure training quality, visual inspections identified several noisy or mislabeled ground truth masks. These problematic directories are listed in [Incorrect_Sample_List.csv](file:///c:/Users/angel/Desktop/Off_Road_Image_Segmentation_Project_Work/Incorrect_Sample_List.csv) so they can be isolated, reviewed, or filtered out during dataset preprocessing.

---

## 🧠 Model Architecture

We employ **DeepLabV3** with a **MobileNetV3-Large** backbone (`deeplabv3_mobilenet_v3_large`):
* **Feature Extractor**: MobileNetV3-Large is lightweight and computationally efficient, making it suitable for potential real-time deployment on mobile robots or vehicles.
* **Pretrained Weights**: Initialized with `COCO_WITH_VOC_LABELS_V1` weights for transfer learning, accelerating convergence and improving final performance.
* **Custom Head**: The final 1x1 convolution layer in the classifier has been modified to output raw logits for **9 classes** (from the original 21 COCO classes):
  ```python
  self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
  ```

---

## 🚀 Training Strategy

The model is trained via [Project_Work_Training.ipynb](file:///c:/Users/angel/Desktop/Off_Road_Image_Segmentation_Project_Work/Project_Work_Training.ipynb) with the following key configurations:

### 1. Data Augmentation
To counter overfitting and prepare the model for varying environment conditions, the training dataset is augmented using the `albumentations` library. Transformations include:
* **Spatial Transforms**: `RandomResizedCrop`, `HorizontalFlip`, `VerticalFlip`, `Rotate` (up to 10°), `Affine` translations, and `Perspective` changes.
* **Color & Contrast Transforms**: `RandomBrightnessContrast` and `GaussianBlur` (to simulate motion blur or camera focus issues).
* **Normalization**: Inputs are normalized using ImageNet standard values: Mean `(0.485, 0.456, 0.406)` and Std `(0.229, 0.224, 0.225)`.

### 2. Validation & Cross-Validation
* **5-Fold Cross-Validation**: The training set is split into 5 distinct folds to ensure robust performance estimation.
* **Folds Execution**: The model is trained for **20 epochs** per fold.
* **Early Stopping / Model Selection**: Metrics are verified on the validation subset after each epoch. The fold achieving the highest validation mean Intersection over Union (mIoU) is selected, and its weights are saved to [best_segmentation_model.pth](file:///c:/Users/angel/Desktop/Off_Road_Image_Segmentation_Project_Work/best_segmentation_model.pth).

### 3. Optimizer & Hybrid Loss Function
* **Optimizer**: Adam with a learning rate of `1e-4`.
* **Loss Function**: Since certain classes (e.g., puddles, obstacles) are highly underrepresented, a simple Cross-Entropy loss would bias predictions towards background, sky, or trails. We utilize a hybrid loss function:
  $$\text{Loss} = 0.3 \times \text{Focal Loss} + 0.7 \times \text{Dice Loss}$$
  * *Focal Loss* focuses on hard-to-classify samples.
  * *Dice Loss* optimizes region overlap and handles extreme class imbalances.

### 4. Tracking and Logging
All training details (loss, accuracies, and IoUs) are logged in real time using TensorBoard. The logs are saved in the `runs/` directory (e.g., `runs/Fold_1` to `runs/Fold_5`) and can be visualized running:
```bash
tensorboard --logdir=runs
```

---

## 🧪 Evaluation and Testing

Model testing is implemented in [Project_Work_Testing.ipynb](file:///c:/Users/angel/Desktop/Off_Road_Image_Segmentation_Project_Work/Project_Work_Testing.ipynb):
1. **Model Loading**: The model architecture is loaded, and the trained weights from [best_segmentation_model.pth](file:///c:/Users/angel/Desktop/Off_Road_Image_Segmentation_Project_Work/best_segmentation_model.pth) are restored.
2. **Inference Loop**:
   * Test images are resized to $512 \times 512$ pixels and normalized.
   * A forward pass computes logit predictions.
   * Pixel-wise labels are selected via `argmax` over the channel dimension.
   * The predicted mask is resized back to the original test image dimensions using **nearest-neighbor interpolation** to ensure fair evaluation.
3. **Metrics Computed**:
   * **Intersection over Union (IoU)** for each of the 8 foreground classes (excluding background).
   * **Mean IoU (mIoU)** across all evaluation classes to determine the overall test score.
   * Interactive plots showing: Original RGB image $\rightarrow$ Ground-truth mask $\rightarrow$ Predicted mask.
