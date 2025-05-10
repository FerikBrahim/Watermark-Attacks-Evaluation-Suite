# Watermark-Attacks-Evaluation-Suite

## Description

This repository provides a structured environment for testing and evaluating the robustness of biometric watermarking algorithms against a range of signal-processing and geometric attacks. It accompanies the research article:

**"Healthcare-Oriented Biometric Watermarking Framework Using FlexenTech and Logistic Maps"**

This suite plays a critical role in the experimental validation of our watermarking framework by enabling reproducible benchmarking of its resilience and imperceptibility across real-world degradation scenarios.

## Objective

In sensitive domains like healthcare, biometric watermarking must preserve fidelity while resisting various image manipulations. This repository was developed to:

- **Challenge** the watermarking system with controlled, real-world attacks.
- **Quantify** the degradation using image similarity and watermark integrity metrics.
- **Benchmark** comparative results across images, embedding techniques, and encryption schemes.

## Attack Simulation Modules

The suite includes several classes of attacks applied to watermarked medical and natural images:

- **Noise-Based**
  - Additive White Gaussian Noise
  - Salt & Pepper Noise
- **Compression**
  - JPEG Compression (variable quality factors)
- **Filtering**
  - Median Filtering
  - Gaussian Blur
- **Geometric**
  - Cropping
  - Rotation
  - Rescaling (Downsample/Upsample)
- **Photometric**
  - Histogram Equalization
  - Brightness Scaling

Each module provides configurable parameters to adjust intensity levels and distortion severity.

## Evaluation Metrics

The following metrics are computed to evaluate watermark imperceptibility and robustness:

| Metric | Description |
|--------|-------------|
| **PSNR** | Measures image quality degradation due to attack |
| **SSIM** | Evaluates perceptual similarity between original and attacked images |
| **BER**  | Indicates bit-wise discrepancy in extracted watermark |
| **NC**   | Assesses correlation between original and retrieved watermark |



