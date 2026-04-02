# Training Guide

Instructions for fine-tuning bh-sentinel-ml models on your organization's clinical notes.

## Overview

The default transformer model uses zero-shot classification (no training data needed). For higher accuracy on your organization's clinical documentation, bh-sentinel supports fine-tuning on de-identified clinical notes.

## Public Datasets

See the [README](../README.md#public-datasets-for-pre-training) for a list of publicly available datasets suitable for pre-training.

## Training Pipeline

The `training/` directory contains scripts for the full pipeline:

1. `prepare_data.py` -- Dataset preparation and formatting
2. `train.py` -- Fine-tuning pipeline
3. `evaluate.py` -- Model evaluation harness
4. `export.py` -- Export to ONNX with INT8 quantization

<!-- TODO: Expand with step-by-step instructions, hyperparameter guidance, evaluation metrics, and de-identification requirements. -->
