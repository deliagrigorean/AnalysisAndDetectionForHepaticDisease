# Analysis and detection for hepatic disease

 This repository contains a **MATLAB-based image processing system** designed to classify liver ultrasound images into six distinct categories. The project aims to provide a reliable decision-support tool for medical professionals to enhance diagnostic speed and accuracy.

 ---

 ## Project description
 This project develops an advanced medical image processing system dedicated to the analysis of liver ultrasounds, aiming to automatically classify them into six distinct categories. The classification is achieved by combining the anatomical acquisition plane **(sagittal, coronal, or transverse)** with the pathological appearance of the tissue **(benign or malignant)**. Fully implemented in MATLAB, the software serves as a decision-support tool for medical professionals, facilitating a faster and more accurate diagnosis. By enabling the early identification of malignant lesions and the rigorous organization of scanning planes, the system directly contributes to preventing disease progression and ensuring more efficient treatment management, providing a modern technological solution to enhance patient care.

---

## Classification criteria
The system categorizes images based on a 3 X 2 matrix:
- **Anatomical Planes:** Sagittal, Coronal, and Transversal
- **Pathological Status:** Benign and Malignant

| Plane / Status | Benign | Malignant |
| -------------- | ------ | --------- |
| **Sagittal** | Class 1 | Class 2 |
| **Coronal** | Class 3 | Class 4 |
| **Transversal** | Class 5 | Class 6 |

---

## Technical Implementation 
  The system is built upon a custom **Convolutional Neural Network (CNN)** architecture designed specifically for ultrasound feature extraction. The pipeline begins with an automated preprocessing stage where medical images are standardized to a 224 X 224 resolution. The deep learning model consists of three primary convolutional blocks, each utilizing 3 X 3 kernels followed by ReLU activation and Max Pooling layers to capture hierarchical spatial features. To prevent overfitting and ensure robust generalization, the network incorporates a Dropout layer (0.5) before the final classification. The model is trained using the **SGDM (Stochastic Gradient Descent with Momentum)** optimizer and identifies six specific classes: *cancer-coronal*, *cancer-sagittal*, *cancer-transverse*, *normal-coronal*, *normal-sagittal*, and *normal-transverse*.
