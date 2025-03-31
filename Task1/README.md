# Task 1: Convolutional-Recurrent Architectures

## Project Overview

This project implements convolutional-recurrent neural network architectures to classify artwork based on three attributes (Artist, Style, and Genre) using the [WikiArt dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). The implementation builds upon state-of-the-art research in recurrent convolutional networks, with several modifications for improving art classification. The trained model is subsequently used for outlier detection to identify images that may have been assigned incorrect labels.

## Dataset

The WikiArt dataset used in this project contains:
- Approximately 13,000 images labeled with artist information (23 classes)
- Approximately 45,000 images labeled with genre information (10 classes)
- Approximately 57,000 images labeled with style information (27 classes)

Due to these imbalances, I implemented the following training strategy:

**Combined Attribute Model**: A single model predicting all three attributes
   - Used the intersection of all labeled data (approximately 16,000 images)
   - Data split: 80% training / 10% validation / 10% test images
   The final dataset contained 23 artist classes, 10 genre classes, and 16 style classes.

Note: Taking the intersection resulted in a loss of 11 style classes from the original dataset.

## Methodology

### Literature Review

My approach is grounded in an extensive literature review of recurrent convolutional architectures:

1. Starting with the foundational paper [Recurrent Convolutional Neural Network for Object Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_004.pdf)
2. Expanding to more recent developments in the field
3. Ultimately adapting the architecture proposed in [Recurrence along Depth: Deep Convolutional Neural Networks with Recurrent Layer Aggregation](https://proceedings.neurips.cc/paper_files/paper/2021/file/582967e09f1b30ca2539968da0a174fa-Paper.pdf)

### Architecture Selection

After examining multiple recurrent-convolutional architectures proposed in the literature, I selected the resnet_rla_lstm hybrid model based on the following considerations:

1. **LSTM vs. GRU/RNN**: LSTMs offer superior capability for modeling long-range dependencies in visual features, essential for capturing artistic styles.
2. **Computational Feasibility**: While more complex than GRU or RNN variants, this dataset's LSTM model remains computationally tractable.
3. **Feature Propagation**: The LSTM's memory cell provides better propagation of critical visual features through the network depth.
4. **Gradient Flow**: The architecture helps mitigate vanishing/exploding gradient problems common in deep networks.

### Attention Mechanisms

I experimented with several attention mechanisms to enhance model performance:

1. **[Squeeze and Excitation (SE)](https://arxiv.org/abs/1709.01507)**: Channel attention mechanism that adaptively recalibrates channel-wise feature responses (available in paper codebase).
2. **[Efficient Channel Attention (ECA)](https://arxiv.org/abs/1910.03151)**: Lightweight alternative to SE that uses 1D convolutions for local cross-channel interaction (available in paper codebase).
3. **[CBAM (Convolutional Block Attention Module)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)**: Combined spatial and channel attention (integrated by me into the existing code).

Interestingly, the full CBAM implementation (including spatial attention) degraded performance.

**Key Observation**: Different channel attention mechanisms consistently improved results across all models, with SE attention yielding the best performance.

### Data Preprocessing and Augmentation

The original paper proposing resnet_rla_lstm had somewhat outdated training methodology and data augmentation techniques, which led to suboptimal performance in my initial experiments due to model overfitting. Replacing these with current data augmentation techniques, optimizers, and learning schedules led to an improvement of approximately 7% in artist classification tasks by enhancing the model's generalizability.

## Implementation Details

The accompanying code notebook contains more detailed information regarding the observations and results described below. I have documented every challenge faced throughout the project, along with appropriate solutions and explanations.

## Results and Analysis

### Model Performance

| Model Type | Artist Accuracy | Genre Accuracy | Style Accuracy | Average |
|------------|----------------|---------------|----------------|---------|
| ResNet50-RLA  | 79.96% | 70.12% | 77.04% | 75.71% |
| ResNet50-RLA + SE | 81.21% | 72.34% | 79.34% | 77.63% |
| ResNet50-RLA + ECA | 79.26% | 70.21% | 77.75% | 75.74% |
| ResNet50-RLA + CBAM | 76.77% | 68.88% | 75.44% | 73.70% |

### Key Findings

1. The recurrent component (LSTM) consistently performed well across all classification tasks and model variants.
2. Channel attention mechanisms provided further improvements, with SE attention being the most effective.
3. Spatial attention from CBAM unexpectedly reduced performance, suggesting that channel relationships are more important than spatial relationships for art classification.

## Challenges and Solutions

1. **Dataset Imbalance**: Addressed through careful dataset manipulation using NumPy and pandas.
2. **Model Complexity vs. Performance**: Found optimal balance with the ResNet50-LSTM + SE architecture.
3. **Training Stability**: Improved through learning rate scheduling and enhanced optimizer selection.
4. **Overfitting**: Mitigated with enhanced data augmentation, regularization techniques, and early weight saving.

## Outlier Detection

This project focuses on detecting outliers in image data using deep learning techniques and dimensionality reduction methods.

I selected the best classifier model (ResNet50-RLA + SE) for outlier detection. For extracting image embeddings, I removed the final fully connected layer in the model architecture, allowing me to obtain high-dimensional feature representations of the images. This way,I generated embeddings for all images in the test dataset using this modified model.

### Confusion Matrix Visualization

I created confusion matrices using the extracted embeddings to visualize the model's performance and identify potential misclassifications or outliers, yielding the following key conclusions:

#### Artist Classification

Several artists show strong diagonal values (>0.8) in the confusion matrix, indicated by high classification accuracy for artists like Gustave_Dore (1.0), Claude_Monet (0.84), and Pierre_Auguste_Renoir (0.82).

Some artists exhibit lower diagonal values, resulting in confusion patterns with others and suggesting stylistic similarities or overlapping characteristics.

#### Genre Classification

Strong performance for illustration (0.73), landscape (0.87), portrait (0.75), and sketch_and_study (0.78) categories suggests distinctive features for these genres.

Genre_painting shows significant confusion with abstract painting (1.0), indicating strong visual similarities between these categories. Religious_painting shows the most distributed confusion across multiple categories, suggesting it may contain elements common to several genres.

#### Style Classification

Impressionism (0.92) and Northern Renaissance (0.80) demonstrate strong classification performance, indicating highly distinctive features for these styles.

Analytical Cubism and Synthetic Cubism exhibit significant confusion, often misclassified as Cubism, indicating a close relationship between these styles.

### T-SNE Plot Visualization

I created t-SNE plots for visualizing clusters and outliers, which use dimensionality reduction to reduce the embedding dimensions to 2D. From these plots, I observed that most outliers are found on the boundaries of clusters. Key observations include:

The K-means clustering (k=23) shows well-defined artist clusters with distinct boundaries, suggesting the model effectively captures artist-specific visual characteristics. The outlier visualization reveals that misclassified artworks (marked with a red X) tend to appear at cluster boundaries or in regions where multiple artist styles overlap.

The genre K-means clustering (k=10) reveals more distinct and separated clusters than artist clustering, suggesting genre characteristics may be more visually consistent than artist-specific traits. Genre outliers appear more frequently at intersection points between clusters, highlighting works that contain elements of multiple genres.

K-means clustering (k=16) style shows moderately distinct clusters with some overlap, reflecting the natural progression and influence between artistic styles. Style outliers are more evenly distributed across the plot than artist and genre outliers, suggesting style misclassifications occur across a broader range of visual characteristics.

## Image Visualization

Finally, I visualized the top three outliers according to the model's predictions. These are images that the model identifies as potentially having all three attributes (Artist, Style, and Genre) mislabeled.

## Repository Structure

The project repository is organized as follows:

- **Class Mapping Files**:
  - `artist_class.txt`, `genre_class.txt`, `style_class.txt`: These files map class labels to numerical values for model training and evaluation.

- **Dataset Preparation**:
  - `dataset.ipynb`: This notebook contains code for creating the merged dataset (`merged_wikiart_dataset.csv`). The process involves taking the intersection of the three original WikiArt CSV files and remapping class labels for consistent processing.

- **Model Development**:
  - `model-testing.ipynb`: This notebook was used for testing and debugging different model variants using dummy data before full-scale training.

- **Core Implementation**:
  - `classification.ipynb`: The primary notebook containing code for image attribute classification, including model training, validation, and testing.

- **Analysis and Evaluation**:
  - `outlier-detection.ipynb`: This notebook contains code that utilizes the trained model from `classification.ipynb` to perform outlier detection on the test dataset.
