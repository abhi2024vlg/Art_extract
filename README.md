# ArtExtract GSOC 2025 - Evaluation Tasks

## Personal Introduction
Hello! I'm Abhinav Kumar, a 2nd-year Undergraduate student at IIT Roorkee, India. I have a strong interest in computer vision and deep learning. I have accepted papers in Computer Vision at conferences like ICLR 2025 and AAAI 2024. I'm excited to present my solutions for the ArtExtract Project evaluation tasks for Google Summer of Code 2025 under the HumanAI Umbrella Organization.

## Project Overview
This repository contains my pytorch implementations for the two evaluation tasks:

1. **Task 1: Convolutional-Recurrent Architectures** - A deep learning model based on the NeurIPS 2021 paper titled [Recurrence along Depth: Deep Convolutional Neural
Networks with Recurrent Layer Aggregation](https://proceedings.neurips.cc/paper_files/paper/2021/file/582967e09f1b30ca2539968da0a174fa-Paper.pdf) for classifying artistic attributes (Style, Artist, Genre) using the ArtGAN dataset and outlier detection.

2. **Task 2: Similarity Analysis** - A resnet-50 model trained as per Neurips 2020 paper titled [Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf) for discovering similarities between paintings using the National Gallery of Art open dataset.

## Repository Structure
- `Task1/`: Contains all code, notebooks, and results for Task 1
- `Task2/`: Contains all code, notebooks, and results for Task 2


### Dataset Preparation
- **Task 1**: Instructions for obtaining and preparing the ArtGAN dataset can be found in `Task1/README.md`
- **Task 2**: Instructions for obtaining and preparing the National Gallery of Art dataset can be found in `Task2/README.md`

## Task Summaries

### Task 1: Convolutional-Recurrent Architectures
I implemented a hybrid CNN-LSTM architecture to classify artistic attributes. The model achieves 83.05 accuracy on artist classification, 74.86 percent on genre classification and 80.86 accuracy on style classification.
<img width="848" alt="Screenshot 2025-03-31 at 12 45 53 PM" src="https://github.com/user-attachments/assets/739ae7b1-2f8f-4941-a138-8a713106d7a7" />

### Outlier Detection
I used the trained classifier model from previous task and used its final embedding for clustering and outlier detection. I visualised the SNE plot and confusion matrix for all three attributes.
<img width="680" alt="Screenshot 2025-03-31 at 1 10 46 PM" src="https://github.com/user-attachments/assets/f80b0b7d-f23a-43ea-9900-68b7eb71f945" />

For detailed methodology and results, see [Task1/README.md](Task1/README.md).

### Task 2: Similarity Analysis
For similarity detection, I implemented a Siamese network proposed in the paper [Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf). The model successfully identifies similar artistic elements across different paintings with an average of 0.79114 Clip score on topmost similar pairs.

<img width="894" alt="Screenshot 2025-04-01 at 12 22 42 AM" src="https://github.com/user-attachments/assets/e00cb2d6-3e98-4cdf-a3a4-3f6cc8e6abf1" />


For detailed methodology and results, see [Task2/README.md](Task2/README.md).

## Key Results & Contributions
- Developed an effective approach for multi-label art classification
- Identified and analyzed outliers in artistic style attribution
- Demonstrated effective transfer learning from natural images to artistic domains for similar images using pre-trained resnet50 model.



Feel free to reach out with any questions about my implementation or approach!
