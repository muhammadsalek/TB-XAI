# TB-XAI

Explainable Tuberculosis Detection and Lesion Localization from Chest X-rays.

## Overview

TB-XAI combines deep-learning classification, model calibration, explainable AI, and object detection for tuberculosis chest X-ray analysis.

## Pipeline

Chest X-ray  
→ ConvNeXtTiny classification  
→ Temperature scaling  
→ Grad-CAM / Grad-CAM++  
→ YOLO26n lesion localization  
→ Localization comparison  
→ Failure and subgroup analysis

## Models

- ConvNeXtTiny for TB classification
- YOLO26n for lesion localization
- Grad-CAM and Grad-CAM++ for explainability

## Main Results

### Classification

- Test AUROC: 0.5333
- Test AUPRC: 0.5847
- Sensitivity: 0.5660
- Specificity: 0.4340
- Temperature: 0.9915
- Decision threshold: 0.4959

### YOLO26n Localization

- Test mAP@50: 0.7098
- Test mAP@50–95: 0.3079
- Precision: 0.7357
- Recall: 0.6503
- Selected confidence threshold: 0.40

### Localization Comparison

- YOLO26 mean IoU: 0.5386
- Grad-CAM mean IoU: 0.0436
- Grad-CAM++ mean IoU: 0.0370

## Repository Structure

```text
TB-XAI/
├── figures/
├── math/
├── models/
├── notebooks/
├── results/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Model Weights

The YOLO26n detector is stored in the `models` directory.

The ConvNeXtTiny classifier checkpoint is provided through the GitHub Release `v1.0-models`.

## Disclaimer

This project is intended for research and educational purposes only.

It is not intended for clinical diagnosis or medical decision-making.
