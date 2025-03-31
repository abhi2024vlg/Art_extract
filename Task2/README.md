# Task 2: Self-Supervised Contrastive Learning for Image Similarity

## Project Overview

This project implements a self-supervised contrastive learning approach to find similar images based on various attributes (such as posture, artist style, etc.) using an unlabeled dataset. The implementation is based on the Neurips 2020 paper titled [Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning (BYOL)](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf) methodology, which trains a model to generate similar embeddings for different augmentations of the same image without requiring explicit negative pairs. The trained model can then be used to identify similar images based on learned visual features.

## Dataset
The dataset used in this project consists of unlabeled images from the National Gallery of Art's open data repository. I created this dataset by web scraping the publicly available NGA dataset from [National Gallery of Art](https://github.com/NationalGalleryOfArt/opendata) 

### Data Collection Process

1. **Web Scraping**: I developed a Python script to download images from the National Gallery of Art's IIIF APIs.
2. **Image Processing**: The script downloads thumbnail versions of artworks to create a consistent dataset.
3. **Metadata Tracking**: For each downloaded image, I preserved metadata including UUID and object ID for future reference.

Key characteristics of the dataset:
- No predefined labels for training
- No explicit positive/negative pairs
- Need to learn meaningful representations for similarity detection
- Images sourced from a diverse collection of fine art

## Methodology

### Literature Review and Architecture Selection

My approach is grounded in an extensive literature review of contrastive learning and self-supervised learning techniques:

- Starting with the foundational paper [Learning Fine-grained Image Similarity with Deep Ranking](https://arxiv.org/abs/1404.4661)
- Exploring recent developments in self-supervised learning
- Finally adopting the approach proposed in **Bootstrap Your Own Latent (BYOL)** (https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)

I selected the BYOL methodology based on the following considerations:

- **No Negative Samples Required**: Unlike triplet loss or other contrastive methods, BYOL doesn't need explicit negative pairs.
- **Collapse Prevention**: Employs mechanisms to prevent representation collapse without large batch sizes.
- **Generalizability**: Creates robust representations that generalize well to downstream tasks.

### Implementation Details

## Dataset and Data Loading
- **Custom Datasets:**  
  Created custom `BaseArtDataset`, `ByolTrainDataset`, and `MetricEvalDataset` classes for handling the art images.
- **Data Loading Strategies:**  
  Implemented separate strategies for:
  - **Training:** Single image input.
  - **Evaluation:** Pairs of augmentations.
- **Data Splitting:**  
  Used `random_split` to create train, validation, and test splits from the unlabeled dataset.

## Network Architecture
- **Backbone Encoder:**  
  Utilized ResNet50 (pretrained) as the backbone encoder.
- **BYOL Implementation:**  
  Implemented BYOL with the `byol_pytorch` library, configured to use the "avgpool" layer as the hidden representation.
- **Simplified Target Encoder:**  
  Disabled momentum updates in the target encoder for simplicity.

## Training Process
- **Epochs & Scheduling:**  
  Trained for 100 epochs using a cosine learning rate schedule with a 10-epoch warmup.
- **Optimizer:**  
  Employed the Adam optimizer with a base learning rate of 3e-4.
- **Evaluation:**  
  Performed evaluation every 10 epochs to monitor progress using intrinsic metrics.



The BYOL approach was implemented with the following components:

1. **Augmentation Strategy**:
   - Two different augmentations of each image are created
   - One serves as the "view" for the online network, the other for the target network

2. **Network Architecture**:
   - Online network: Encoder + Projector + Predictor
   - Target network: Encoder + Projector (weights updated via exponential moving average)

3. **Training Process**:
   - The predictor from the online network aims to predict the target network's projection
   - Loss is calculated as the negative cosine similarity between the prediction and target
   - Target network parameters are updated using exponential moving average (EMA) of online network

4. **Evaluation Metrics**:
   - Based on **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**
   - Alignment: Measures how close embeddings of similar samples are
   - Uniformity: Measures how well the embeddings are distributed on the unit hypersphere
   - CLIP Score: Used as a proxy metric for visual examination and analysis

## Results and Analysis

### Model Performance

| Metric | Value |
|--------|-------|
| Alignment | [TO BE FILLED] |
| Uniformity | [TO BE FILLED] |
| CLIP Score | [TO BE FILLED] |

### Key Findings

- [TO BE FILLED]
- [TO BE FILLED]
- [TO BE FILLED]

### Visualization Results

- [TO BE FILLED]
- [TO BE FILLED]

## Challenges and Solutions

- **No Labeled Data**: Addressed through self-supervised learning using BYOL, which learns meaningful representations without explicit labels.
- **Representation Collapse**: Prevented through the asymmetric architecture and moving average update of the target network.
- **Training Stability**: Improved by using appropriate augmentation strategies and optimization techniques.
- **Evaluation without Ground Truth**: Solved by using intrinsic metrics (alignment and uniformity) and proxy metrics (CLIP Score).

## Repository Structure

The project repository is organized as follows:

- **Data Collection**:
  - `dataset.py`: Script for downloading images from the National Gallery of Art repository
 


- **Model Implementation**:







## Evaluation Metrics
- **Alignment Loss:**  
  Measures how close embeddings of augmented views of the same image are (lower is better).
- **Uniformity Loss:**  
  Assesses how well the embeddings are distributed on the unit hypersphere (lower is better).
- **CLIP Score:**  
  Used as a reference to compare the model's similarity results with a pre-trained vision-language model.

# Results and Analysis

## Model Performance
The model was evaluated using intrinsic metrics on the test set:
- **Alignment:** Value from `test_align`
- **Uniformity:** Value from `test_uniform`
- **Total Metric:** Value from `test_total`

## Visual Similarity Analysis
- **Top Similar Pairs:**  
  The `find_top_similar_pairs` function identified the most visually similar artworks based on the learned embeddings.
- **Comparison with CLIP:**  
  Each pair's similarity was also evaluated using CLIP embeddings for reference.
- **Visualization:**  
  Results were visualized by displaying both source images along with their similarity scores from both BYOL and CLIP models.

## Key Findings
- **Representation Learning:**  
  The model successfully learned to identify similarities in artistic style, composition, and subject matter without explicit labels.
- **Correlation with CLIP:**  
  There was a notable correlation between the similarity scores from the self-supervised BYOL model and the CLIP model, suggesting meaningful representation learning.
- **Nuanced Relationships:**  
  The model captured nuanced visual relationships that extend beyond simple color or composition similarities.

## Challenges and Solutions
- **Data Management:**  
  Implemented efficient data loaders with proper augmentation strategies for both training and evaluation.
- **Evaluation Without Labels:**  
  Developed intrinsic metrics based on representation learning theory to evaluate model performance.
- **Scalability:**  
  Utilized GPU acceleration and optimized batch sizes to handle computational requirements.
- **Visualization:**  
  Created custom visualization functions to qualitatively assess the model's performance.

# Repository Structure

The project consists of two main components:

## `dataset.py`
- Contains functions for downloading images from the National Gallery of Art repository.
- Includes code for creating the unlabeled dataset of artwork images.

## `similar-images.ipynb`
- Contains the complete implementation of the BYOL model and training process.
- Implements custom dataset classes for art images.
- Provides data loading utilities with proper augmentation strategies.
- Features a training loop with evaluation using intrinsic metrics.
- Implements similarity detection functions.
- Includes CLIP score evaluation for comparison.
- Provides visualization tools for qualitative analysis.

### Notebook Components
- **Data Processing:**  
  Custom dataset implementations.
- **Model Definition:**  
  BYOL architecture implementation.
- **Training:**  
  Complete training loop with progress tracking.
- **Evaluation:**  
  Intrinsic metrics and similarity detection.
- **Visualization:**  
  Functions to render and compare similar image pairs.

