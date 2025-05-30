# How Synthetic Images Boost Real Classifiers

This project explores how synthetic images generated by various generative models can improve the performance of image classifiers, especially when training data is limited or imbalanced. Using the FashionMNIST dataset, the study compares models like VAE, DVAE, VQ-VAE, GANs, and diffusion models across three data regimes: balanced, semi-balanced, and highly imbalanced. By augmenting real data with model-generated samples, the team measures the resulting accuracy of a CNN classifier and analyzes how different models affect learning in low-data scenarios. The findings suggest that denoising models like DVAE and stable architectures like VQ-VAE can outperform more complex methods like GANs or diffusion models, especially when data is scarce or noisy.

## Project Overview

This project evaluates and compares several deep generative models:

- Variational Autoencoder (VAE)
- Vector Quantized VAE (VQ-VAE) 
- Denoising VAE (DVAE) with multiple noise types:
  - Gaussian noise
  - Salt & pepper noise
  - Rotation
  - Brightness
  - Contrast
  - Blur
- Generative Adversarial Networks (GANs)
- BigGAN
- Diffusion Models
- CLIP-guided Diffusion

The models are evaluated on their ability to:
1. Generate high-quality synthetic Fashion-MNIST images
2. Improve downstream classification performance
3. Handle imbalanced datasets through data augmentation

## Key Features

- Multiple dataset configurations:
  - Balanced (500 samples per class)
  - Semi-imbalanced (varying ratios)
  - Highly imbalanced (long-tailed distribution)

- The study evaluates models across multiple dataset sizes to understand scaling behavior:

  - 1,000 samples (100 per class in balanced case)
  - 2,000 samples (200 per class in balanced case)
  - 3,000 samples (300 per class in balanced case)
  - 4,000 samples (400 per class in balanced case)
  - 5,000 samples (500 per class in balanced case)
    

- Comprehensive evaluation metrics:
  - CNN classification accuracy
  - CART decision tree performance
  - Per-class precision and recall
  - Visual quality assessment

- Advanced architectures:
  - Self-attention mechanisms
  - Residual connections
  - Spectral normalization
  - Conditional generation

## Project Structure

Detailed breakdown of project components:

### Core Components

#### Engine/
- `engine.py`: Core training and evaluation logic, including noise addition and performance metrics
- `train.py`: Implementation of training loops for VAE, GAN, VQ-VAE, and other models

#### Models/
- `VAE_CNN.py`: Implementations of:
  - Variational Autoencoder (VAE)
  - Vector Quantized VAE (VQ-VAE)
  - CNN Classifier

- `GANS.py`: GAN architectures including:
  - Conditional GAN
  - BigGAN

#### Main/
- `main.py`: Primary entry point for VAE training and evaluation
- `CNN_main.py`: Dedicated CNN classifier training pipeline
- `VQ_VAE_main.py`: VQ-VAE specific training and evaluation

### Data and Analysis

#### Data_Prep/
- `create_data.py`: Dataset creation utilities:
  - Balanced dataset generation
  - Semi-imbalanced dataset creation
  - Highly imbalanced dataset generation
  - Train/test splitting functionality

#### Inference/
- `inference_plot.py`: Visualization tools for:
  - Model reconstructions
  - Performance metrics
  - Comparative analysis plots

### Advanced Models and Analysis

#### Latent_Space_Analysis/
- `clip-guided.py`: CLIP-guided diffusion model implementation
- `ddpm_fashionmnist.py`: Diffusion model for Fashion-MNIST
- `combining_cnn_cart.py`: Analysis combining CNN and CART classifier results
- Notebooks:
  - `CNN_Embeddings.ipynb`: Analysis of CNN feature embeddings
  - `DVAE Test.ipynb`: DVAE experimentation and testing
  - `VAE_Replication.ipynb`: VAE implementation and validation
  - `Diffusion and Clip.ipynb`: CLIP-guided generation experiments

### Comparison and Evaluation
- `CNN_VAE_generation_comparison.py`: Comprehensive comparison framework:
  - Model training and evaluation
  - Synthetic data generation
  - Performance metrics calculation
  - Visualization generation
  - Cross-model comparison

## Installation

```bash
git clone https://github.com/yourusername/VAE_VS_OTHERS.git
cd VAE_VS_OTHERS
pip install -r requirements.txt
```

## Usage

Train and evaluate models:
```bash
# Train VAE on balanced dataset
python main/main.py --balanced

# Train DVAE with gaussian noise
python main/main.py --add_noise --noise_type gaussian

# Train VQ-VAE
python main/VQ_VAE_main.py --balanced
```

Compare model performance and generate synthetic data:
```bash
# Run comprehensive comparison across all models
python CNN_VAE_generation_comparison.py --dataset_type balanced --dataset_size 1000

# Available options:
# --dataset_type: balanced, semi, high
# --dataset_size: 1000, 2000, 3000, 4000, 5000
```

> **Note**: The comparison script involves training multiple deep learning models (including computationally intensive diffusion models and CLIP-guided diffusion) and is computationally intensive. Running on a GPU is strongly recommended.

The comparison script:
- Trains each model on the specified dataset configuration:
  - VAE and its variants (DVAE, VQ-VAE)
  - GANs (Conditional GAN, BigGAN)
  - Diffusion models
  - CLIP-guided diffusion
- Generates synthetic samples from each model
- Evaluates CNN classifier performance on:
  - Original + synthetic data
  - Various augmentation strategies
- Produces comparison plots and metrics in the Results/ directory:
  - Accuracy vs dataset size plots
  - Per-class performance metrics
  - Sample reconstructions

## Results

The results show that synthetic data significantly boosts classification accuracy, particularly in low-data and imbalanced scenarios. Among the generative models, DVAE and VQ-VAE consistently produced the most beneficial augmentations, leading to higher classifier performance compared to using only real data. While GANs and diffusion models showed potential, their results were less stable and more sensitive to training conditions. In the highly imbalanced regime, DVAE-augmented data improved accuracy by up to 10%, highlighting its effectiveness in addressing class imbalance. Overall, simpler and more robust models like DVAE and VQ-VAE proved to be the most practical for data augmentation in constrained settings.

## Authors

Azfal Peermohammed, Prateek Gautam, Aziz Malouche

