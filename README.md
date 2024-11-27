# Quantum Convolutional Neural Network for Multiclass Image Classification

## Overview
This repository contains the implementation of a Quantum-Inspired Convolutional Neural Network (QCNN) aimed at tackling the challenge of multiclass image classification. The model leverages quantum computing concepts such as superposition and entanglement to enhance the feature extraction process in traditional Convolutional Neural Networks (CNNs). The project was conducted as part of a course on Quantum Computing at the National University of Computing & Emerging Sciences (NUCES), Karachi Campus.

#### Project Team:
- **Anas Ghazi** - 21L-5081
- **Arwa Mukhi** - 21k-0500
- **Seher Imtiaz** - 21k-3363

#### Instructor:
- **Ms. Sumaiyyah Zahid**

## Abstract
This project explores the use of Quantum-Inspired Convolutional Neural Networks (QCNN) for the classification of Pokémon images. The goal is to test the efficiency of QCNN in classifying images, especially RGB images, and evaluate its performance in comparison to traditional CNNs. By adapting the architecture in the referenced paper, we aim to improve accuracy and performance when dealing with complex datasets, such as Pokémon images, which heavily rely on color for classification.

## Keywords
- Quantum Computing
- CNN (Convolutional Neural Network)
- Multiclass Image Classification
- Feature Extraction
- Transfer Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [2.1 Datasets](#datasets)
   - [2.2 Model Architecture](#model-architecture)
3. [Findings](#findings)
4. [Proposed Solution and Results](#proposed-solution-and-results)
5. [Conclusion](#conclusion)
6. [References](#references)

## 1. Introduction
Image classification is a prominent task in computer vision, where the goal is to categorize images into predefined classes. Traditional CNNs have achieved great success, but when working with high-dimensional data or smaller datasets, they can face challenges. This project introduces a Quantum-Inspired CNN (QCNN), which aims to address these challenges by incorporating quantum computing principles like superposition and entanglement into the network architecture. Our goal is to evaluate how QCNN performs on RGB images, particularly in classifying Pokémon images, and compare the results with traditional CNNs.

## 2. Methodology
### 2.1 Datasets
- **Pokémon Dataset**: A set of 7,000 labeled Pokémon images, comprising 150 classes of Pokémon. The images are cropped and labeled, with each class containing 25-50 RGB images.
- **Fashion MNIST**: A dataset with 28x28 grayscale images of 70,000 fashion items used to validate the model implementation.

### 2.2 Model Architecture
The model follows a quantum-inspired approach, where a quantum convolutional layer is combined with traditional CNN layers. The architecture involves:
- **Pre-processing**: The Pokémon images were resized to 28x28 and initially converted to grayscale. Later, modifications allowed for RGB images to be processed.
- **Classical CNN Layers**: Initially, the model used only shallow CNN layers (3 layers), but to improve feature extraction, the depth was increased to 4 layers.
- **Quantum Layer**: The QCNN uses quantum gates like RX, RY, and RZ to transform classical features into quantum states, where trainable weights are introduced to enhance learning during the training process.
- **Transfer Learning**: ResNet-50, a pre-trained CNN, was employed for feature extraction, improving the accuracy by capturing detailed spatial features.
- **Training**: The model was trained using 100 epochs with early stopping to prevent overfitting.

## 3. Findings
The primary observation was the low accuracy achieved when applying the QCNN to RGB Pokémon images. The model achieved accuracy between 0% - 2%, and this issue was traced back to several key limitations:
- The conversion of images to grayscale resulted in the loss of crucial color information.
- The shallow convolutional layers were not capable of capturing the intricate features of Pokémon images.
- The use of a small dataset (200 samples for training) resulted in underfitting.

In contrast, the Fashion MNIST dataset achieved better results, consistent with the findings in the original research paper. The results revealed that quantum layers might not yet outperform CNNs for tasks heavily dependent on color and texture.

## 4. Proposed Solution and Results
To improve the model's performance, the following changes were made:
- **Pre-processing**: The images were not converted to grayscale, and a noise adder along with random resizing and cropping were applied to increase variability in the dataset.
- **Quantum Layer Modifications**: The quantum layer was enhanced by introducing more gates and making the layer deeper (3 quantum layers instead of 1).
- **Use of Transfer Learning**: The ResNet-50 model was used to extract features from the images, improving the ability of the QCNN to capture fine-grained textures and spatial features.
- **Training Strategy**: A scheduler was introduced to gradually decrease the learning rate, and early stopping was implemented to prevent overfitting.

After 100 epochs, the accuracy improved significantly, though it still remained lower than expected. With more computational power and extended training (up to 300-500 epochs), better results are expected.

## 5. Conclusion
The use of QCNNs for image classification, especially on RGB datasets, presents both exciting possibilities and challenges. While the model's performance was limited by the need for deeper architectures and better data processing techniques, the inclusion of quantum layers showed potential in improving feature extraction. Future work will focus on optimizing quantum-inspired architectures and exploring the use of actual quantum hardware as it matures.

The project illustrates the potential of hybrid quantum-classical approaches in enhancing machine learning models for complex image classification tasks.

## 6. References
1. Hamza Kamel Ahmed, Baraa Tantawi, and Gehad Ismail Sayed, “Multiclass Image Classification Based on Quantum-Inspired Convolutional Neural Network,” vol. 3392, pp. 177–187, Jan. 2023, DOI: [10.32782/cmis/3392-15](https://doi.org/10.32782/cmis/3392-15).
2. Lance Zhang, “7,000 Labeled Pokémon Dataset,” Kaggle: [Link](https://www.kaggle.com/datasets/lantian773030/pokemonclassification)
3. “Fashion MNIST,” Kaggle: [Link](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
4. Y. Jing et al., “RGB Image Classification with Quantum Convolutional Ansatz,” vol. 21, no. 3, Feb. 2022, DOI: [10.1007/s11128-022-03442-8](https://doi.org/10.1007/s11128-022-03442-8).
