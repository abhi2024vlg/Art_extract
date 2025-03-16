# Task 1: Convolutional-Recurrent Architectures

## Project Overview

This project implements convolutional-recurrent neural network architectures to classify artwork based on three attributes (Artist, Style, and Genre) using the [WikiArt dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). The implementation builds upon state-of-the-art research in recurrent convolutional networks, with several modifications for improving art classification.

## Dataset

The WikiArt dataset used in this project contains:
- ~13,000 images labeled with artist information ( 23 classes) 
- ~45,000 images labeled with genre information ( 10 classes)
- ~57,000 images labeled with style information (27 classes)

Due to these imbalances, I implemented the following training strategies:

1. **Individual Attribute Models**: Separate models for Artist, Genre, and Style
   - Artist: 11,000 training / 1,000 validation / 1,000 test images
   - Genre & Style: Subset of 32,000 images (30,000 training / 1,000 validation / 1,000 test)

2. **Combined Attribute Model**: Single model predicting all three attributes
   - Used intersection of all labeled data (~11,000 images)
   - 10,000 training / 500 validation / 500 test images ( Contained 23 artist classes, 10 genre classes but only 16 style classes)

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
   - Interestingly, the full CBAM implementation (including spatial attention) sometimes degraded performance

#### Observation:-> Different Channel attention mechanisms proposed across all three mechanisms consistently improved results across all models


### Data Preprocessing and Augmentation

The original paper proposing resnet_rla_lstm had a somewhat outdated training methodology and data augmentation technique, which I used to achieve sub-optimal performance because the model overfitted the training dataset in my initial experiments. Replacing it with current data augmentation, optimizer, and learning schedule led to an improvement of around 7% in artist classification tasks by improving its generalizability.

## Implementation Details

More details regarding the above observation and the results below are available in the code notebook. I have discussed everything I have faced throughout the project, along with proper documentation.


## Results and Analysis

### Individual Attribute Models

| Model Type | Artist Accuracy | Style Accuracy | Genre Accuracy |
|------------|----------------|---------------|----------------|
| ResNet50 (Baseline) | 76.2% | 58.4% | 67.8% |
| ResNet50 + SE | 77.9% | 60.1% | 69.2% |
| ResNet50 + ECA | 78.1% | 60.0% | 69.4% |
| ResNet50-LSTM | 79.4% | 62.5% | 70.8% |
| ResNet50-LSTM + SE | 81.2% | 64.3% | 72.1% |
| ResNet50-LSTM + CBAM | 80.5% | 63.8% | 71.5% |

### Combined Model Performance

| Model Type | Artist Accuracy | Style Accuracy | Genre Accuracy | Average |
|------------|----------------|---------------|----------------|---------|
| ResNet50 (Baseline) | 74.8% | 56.2% | 65.4% | 65.5% |
| ResNet50-LSTM + SE | 78.9% | 61.7% | 69.8% | 70.1% |

### Key Findings

1. The recurrent component (LSTM) consistently improved performance across all classification tasks
2. Channel attention mechanisms provided further gains, with SE attention being most effective
3. The combined model showed slight performance degradation compared to individual models, likely due to the complexity of jointly modeling multiple attributes
4. Spatial attention from CBAM unexpectedly reduced performance in some cases, suggesting that channel relationships are more important than spatial relationships for art classification
5. Data augmentation improvements provided a significant boost (2-3% accuracy) compared to the techniques described in the original papers

## Challenges and Solutions

1. **Dataset Imbalance**: Addressed through careful subset selection and stratified sampling
2. **Model Complexity vs. Performance**: Found optimal balance with ResNet50-LSTM + SE architecture
3. **Training Stability**: Improved through learning rate scheduling and gradient clipping
4. **Overfitting**: Mitigated with enhanced data augmentation and regularization techniques

## Future Work

1. Implement outlier detection to identify misattributed artworks
2. Explore multi-task learning approaches to better leverage relationships between artist, style, and genre
3. Investigate transfer learning from natural image domains to artistic domains


## Repository Structure

