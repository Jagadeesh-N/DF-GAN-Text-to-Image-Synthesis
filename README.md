# DF-GAN Text-to-Image Synthesis

## Project Overview

This project focuses on generating high-quality images from text descriptions using Deep Fusion Generative Adversarial Networks (DF-GAN). The model synthesizes realistic and text-image semantically consistent images through an innovative approach, improving upon existing methods by simplifying the architecture and enhancing text-image fusion.

## Introduction

The DF-GAN model aims to generate realistic images from textual descriptions. Unlike traditional models that use a stacked architecture, DF-GAN employs a novel one-stage backbone, deep text-image fusion block, and matching-aware zero-centered gradient penalty to achieve high-quality image synthesis.

## Dataset

The project uses two challenging datasets:
- **CUB Bird Dataset**: Contains 11,788 images of 200 bird species, with 10 textual descriptions per image. The dataset is split into 8,855 training images and 2,933 test images.
- **COCO Dataset**: Contains 80,000 training images and 40,000 test images with more complex scenes.

## Model Architecture

### Generator
- Inputs: Sentence vector from a pre-trained text encoder and a noise vector from a Gaussian distribution.
- Process: The noise vector is reshaped and processed through multiple UPBlocks, each consisting of upsample layers, residual blocks, and DFBlocks, to generate images.

### Discriminator
- Structure: Consists of DownBlocks and convolution layers to convert images into feature maps and evaluate their visual realism and semantic consistency.
- Function: Distinguishes between real and synthetic images and promotes the generator to produce high-quality outputs.

### Deep Text-Image Fusion Block (DFBlock)
- Enhances the fusion of text and visual features during image generation.
- Utilizes affine transformations, ReLU activations, and convolution layers for effective text-image integration.

### Matching-Aware Zero-Centered Gradient Penalty (MA-GP)
- Ensures the generator produces realistic and text-consistent images without extra networks.

## Implementation

### Training Details
- **Optimizer**: Adam with \(\beta_1 = 0.0\) and \(\beta_2 = 0.9\).
- **Learning Rates**: 0.0001 for the generator and 0.0004 for the discriminator.
- **Epochs**: 600 for CUB birds dataset and 120 for COCO dataset.
- **Pre-trained Encoder**: Parameters fixed during training.

### Evaluation Metrics
- **Inception Score (IS)**: Measures the quality and classifiability of generated images.
- **Frechet Inception Distance (FID)**: Measures the distance between synthetic and real data distributions.

## Results

The DF-GAN model outperforms state-of-the-art text-to-image synthesis models, achieving better performance in terms of IS and FID scores on the CUB and COCO datasets.

## Usage

### Clone the Repository
```bash
git clone https://github.com/yourusername/df-gan-text-to-image.git
cd df-gan-text-to-image
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Training Script
```bash
python train.py --dataset CUB --epochs 600 --batch_size 64
```

### Generate Images
```bash
python generate.py --input "A yellow bird with black wings and a short beak."
```

### Acknowledgments
Special thanks to our project guide, Dr. K. V. Ramana, and the Department of Computer Science and Engineering at the University College of Engineering Kakinada for their support
