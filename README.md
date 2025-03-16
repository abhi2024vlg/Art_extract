# ArtExtract GSOC 2025 - Evaluation Tasks

## Personal Introduction
Hello! I'm Abhinav Kumar, a 2nd-year Undergraduate student at IIT Roorkee, India. I have a strong interest in computer vision and deep learning. I have accepted papers in Computer Vision at conferences like ICLR 2025 and AAAI 2024. I'm excited to present my solutions for the ArtExtract Project evaluation tasks for Google Summer of Code 2025 under the HumanAI Umbrella Organization.

## Project Overview
This repository contains my pytorch implementations for the two evaluation tasks:

1. **Task 1: Convolutional-Recurrent Architectures** - A deep learning model based on the NeurIPS 2021 paper titled [Recurrence along Depth: Deep Convolutional Neural
Networks with Recurrent Layer Aggregation](https://proceedings.neurips.cc/paper_files/paper/2021/file/582967e09f1b30ca2539968da0a174fa-Paper.pdf) for classifying artistic attributes (Style, Artist, Genre) using the ArtGAN dataset.

2. **Task 2: Similarity Analysis** - A resnet-50 model trained as per Neurips 2020 paper titled [Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf) for discovering similarities between paintings using the National Gallery of Art open dataset.

## Repository Structure
- `Task1-ConvRecurrent/`: Contains all code, notebooks, and results for Task 1
- `Task2-Similarity/`: Contains all code, notebooks, and results for Task 2

### Prerequisites
- Python 3.8 or higher


### Environment and Installation Setup
```bash
# Clone this repository
git clone https://github.com/[your-username]/ArtExtract-GSOC2025.git
cd ArtExtract-GSOC2025

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation
- **Task 1**: Instructions for obtaining and preparing the ArtGAN dataset can be found in `Task1-ConvRecurrent/README.md`
- **Task 2**: Instructions for obtaining and preparing the National Gallery of Art dataset can be found in `Task2-Similarity/README.md`

## Task Summaries

### Task 1: Convolutional-Recurrent Architectures
I implemented a hybrid CNN-LSTM architecture to classify artistic attributes. The model achieves [brief mention of performance metrics] accuracy on style classification and [brief mention of performance metrics] accuracy on artist attribution.

[Optional: Add a sample image or result visualization]

For detailed methodology and results, see [Task1-ConvRecurrent/README.md](Task1-ConvRecurrent/README.md).

### Task 2: Similarity Analysis
For similarity detection, I implemented a Siamese network with a custom triplet loss function. The model successfully identifies similar artistic elements across different paintings with a [brief mention of performance metric] retrieval accuracy.

[Optional: Add a sample image or result visualization]

For detailed methodology and results, see [Task2-Similarity/README.md](Task2-Similarity/README.md).

## Key Results & Contributions
- Developed an effective approach for multi-label art classification
- Identified and analyzed outliers in artistic style attribution
- Created a novel similarity metric for comparing compositional elements
- Demonstrated effective transfer learning from natural images to artistic domains



Feel free to reach out with any questions about my implementation or approach!
