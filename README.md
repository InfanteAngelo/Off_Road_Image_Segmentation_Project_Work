# Semantic Segmentation Project

This repository showcases my work on training, evaluating, and deploying a semantic segmentation model as part of an academic project.  
It has been organized to highlight the engineering and machine learning skills demonstrated throughout its development.

---

## Project Overview

The objective of this project is to design a full semantic segmentation pipeline, including:

- Dataset preparation and cleaning  
- Cross-validation training  
- Model evaluation on test samples   
- Visualization and tracking of training metrics   

The final deliverable is a PyTorch-based segmentation model achieving solid performance on the dataset provided.

---

## 📁 Repository Structure

```text
.
├── runs/                                 # TensorBoard logs and generated graphs
│
├── resources/
│   ├── train/                             # Training samples
│   └── test/                              # Test samples (currently empty)
│
├── Incorrect_Sample_List.csv              # Samples flagged as misclassified in the dataset
├── best_segmentation_model.pth            # Best-performing trained model checkpoint
├── Project_Documentation.pdf              # Complete documentation
├── Project_Presentation.pptx              # Presentation slides
├── Project_Work_Training.ipynb            # Training notebook (5-fold CV, 20 epochs example)
└── Project_Work_Testing.ipynb             # Testing notebook
