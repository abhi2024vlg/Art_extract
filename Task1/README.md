# Task 1: Convolutional-Recurrent Architectures

## Project Overview

This project implements convolutional-recurrent neural network architectures to classify artwork based on three attributes (Artist, Style, and Genre) using the [WikiArt dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). The implementation builds upon state-of-the-art research in recurrent convolutional networks, with several modifications for improving art classification. The trained model is used for further outlier detection for finding images which might have been assigned wrong labels.

## Dataset

The WikiArt dataset used in this project contains:
- ~13,000 images labeled with artist information ( 23 classes) 
- ~45,000 images labeled with genre information ( 10 classes)
- ~57,000 images labeled with style information (27 classes)

Due to these imbalances, I implemented the following training strategy:

**Combined Attribute Model**: Single model predicting all three attributes
   - Used intersection of all labeled data (~16,000 images)
   - 80% training / 10% validation / 10% test images ( Contained 23 artist classes, 10 genre classes but only 16 style classes)

Note: Taking intersection resulted in a loss of 11 style classes. 
## Methodology

### Literature Review

My approach is grounded in an extensive literature review of recurrent convolutional architectures:

1. Starting with the foundational paper [Recurrent Convolutional Neural Network for Object Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_004.pdf)
2. Expanding to more recent developments in the field
3. Ultimately adapting the architecture proposed in [Recurrence along Depth: Deep Convolutional Neural Networks with Recurrent Layer Aggregation](https://proceedings.neurips.cc/paper_files/paper/2021/file/582967e09f1b30ca2539968da0a174fa-Paper.pdf)

### Architecture Selection

After going through multiple recurrent-convolutional architectures proposed in the paper, I selected the resnet_rla_lstm hybrid model based on the following considerations:

1. **LSTM vs. GRU/RNN**: LSTMs offer the superior capability for modeling long-range dependencies in visual features, which is essential for capturing artistic styles
2. **Computational Feasibility**: While more complex than GRU or RNN variants, the LSTM model remains computationally tractable for this dataset
3. **Feature Propagation**: The LSTM's memory cell provides better propagation of critical visual features through the network depth
4. **Gradient Flow**: The architecture helps mitigate vanishing/exploding gradient problems common in deep networks

### Attention Mechanisms

I experimented with several attention mechanisms to enhance model performance:

1. **[Squeeze and Excitation (SE)](https://arxiv.org/abs/1709.01507)**: Channel attention mechanism that adaptively recalibrates channel-wise feature responses. (Available in paper codebase)
2. **[Efficient Channel Attention (ECA)](https://arxiv.org/abs/1910.03151)**: Lightweight alternative to SE that uses 1D convolutions for local cross-channel interaction (Available in paper codebase)
3. **[CBAM (Convolutional Block Attention Module)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)**: Combined spatial and channel attention (Integrated by me in their code)
   - Interestingly, the full CBAM implementation (including spatial attention) degraded the performance

#### Observation:-> Different Channel attention mechanisms proposed across all three mechanisms consistently improved results across all models


### Data Preprocessing and Augmentation

The original paper proposing resnet_rla_lstm had a somewhat outdated training methodology and data augmentation technique, which I used to achieve sub-optimal performance because the model overfitted the training dataset in my initial experiments. Replacing it with current data augmentation, optimizer, and learning schedule led to an improvement of around 7% in artist classification tasks by improving its generalizability.

## Implementation Details

More details regarding the above observation and the results below are available in the code notebook. I have discussed every problem I faced throughout the project, along with proper documentation.

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
2. Channel attention mechanisms provided further gains, with SE attention being most effective.
3. Spatial attention from CBAM unexpectedly reduced performance, suggesting that channel relationships are more important than spatial relationships for art classification.

## Challenges and Solutions

1. **Dataset Imbalance**: Addressed through careful datasset manipulation using numpy panadas.
2. **Model Complexity vs. Performance**: Found optimal balance with ResNet50-LSTM + SE architecture.
3. **Training Stability**: Improved through learning rate scheduling and improved optimizer.
4. **Overfitting**: Mitigated with enhanced data augmentation, regularization techniques and early weight saving.

## Outlier detection

This project focuses on detecting outliers in image data using deep learning techniques and dimensionality reduction methods.

We choosed the best classifier model (ResNet50-RLA + SE) for outlier detection. 

For extracting image embeddings, I removed the final fully connected layer in model architecture. This allowed us to obtain high-dimensional feature representations of the images. 

Using this modified model, we generated embeddings for all images in our test dataset.

### Confusion Matrix Visualization

We created a confusion matrix using the extracted embeddings to visualize the model's performance and identify potential misclassifications or outliers.
