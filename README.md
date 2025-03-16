# ArtExtract GSOC 2025 - Evaluation Tasks

## Personal Introduction
Hello! I'm Abhinav Kumar, a 2nd year Undergraduate student studying at IIT Roorkee, India. I have a strong interest in computer vision, deep learning. I have accepted papers at conferences like ICLR and AAAI in the domain of Computer Vision and I'm excited to present my solutions for the ArtExtract Projects evaluation tasks for Google Summer of Code 2025 under the HumanAI Umbrella Organization.

## Project Overview
This repository contains my implementations for the two evaluation tasks:

1. **Task 1: Convolutional-Recurrent Architectures** - A deep learning model for classifying artistic attributes (Style, Artist, Genre) using the ArtGAN dataset.

2. **Task 2: Similarity Analysis** - A model for discovering similarities between paintings using the National Gallery of Art open dataset.

## Technologies Used
- **Programming Language**: Python 3.8+
- **Deep Learning Frameworks**: PyTorch 2.0, 
- **Computer Vision Libraries**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebooks, Google Colab

## Repository Structure
- `Task1-ConvRecurrent/`: Contains all code, notebooks, and results for Task 1
- `Task2-Similarity/`: Contains all code, notebooks, and results for Task 2
- `CV/`: Contains my curriculum vitae

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Environment Setup
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
