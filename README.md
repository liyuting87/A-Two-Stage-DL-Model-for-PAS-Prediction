# A Two-Stage Deep Learning Model with Segmentation-Guided Top-K Slice Selection for Patient-Level PAS Prediction on MRI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the official implementation of the paper:

> **"A Two‑Stage Deep Learning Model with Segmentation‑Guided Top‑K Slice Selection for Patient‑Level PAS Prediction on MRI"**  
> *Authors: [Your Names]*  
> *Conference/Journal: [Where it is published]*

## Overview

We propose a **two‑stage deep learning pipeline** for automatic prediction of **Placenta Accreta Spectrum (PAS)** from MRI slices at the **patient‑level**.  
The method consists of:

1. **Segmentation Stage** – A U‑Net generates per‑slice probability maps of placental tissue.
2. **Slice Selection Stage** – A **confidence‑guided contiguous window** selects the most informative K slices (default K=2) based on segmentation confidence.
3. **Classification Stage** – A ResNet‑18 classifier predicts PAS probability for each selected ROI, then aggregates slice‑level predictions (default `max` aggregation) to produce a patient‑level diagnosis.

Compared to naive top‑K or random selection, our segmentation‑guided selection improves both accuracy and interpretability.

![Pipeline](docs/pipeline.png) <!-- Add a figure of your pipeline -->

## Key Features

- **End‑to‑end inference** from raw MRI slices to patient‑level PAS/non‑PAS prediction.
- **Segmentation‑guided ROI cropping** (largest connected component + margin) ensures consistent input for the classifier.
- **Contiguous window selection** respects anatomical continuity.
- **Fixed hyperparameters** (K=2, margin=10, threshold=0.6, aggregation=`max`) – ready for reproduction.
- **Outputs** include slice‑level probabilities, patient‑level predictions, and ROI images for manual inspection.

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- MONAI
- Other packages: `numpy`, `pandas`, `pillow`, `scipy`, `tqdm`, `scikit‑learn`, `matplotlib`

Create a conda environment and install dependencies:
```bash
conda create -n pas python=3.8
conda activate pas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # adjust CUDA version
pip install monai numpy pandas pillow scipy tqdm scikit-learn matplotlib
