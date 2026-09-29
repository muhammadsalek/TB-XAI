<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=TB-XAI&fontSize=56&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Explainable%20Tuberculosis%20Lesion%20Analysis%20from%20Chest%20X-rays&descAlignY=58&descSize=17" width="100%"/>

<img src="https://avatars.githubusercontent.com/u/180872571?v=4" width="120" height="120" style="border-radius:50%;border:3px solid #4F9DFF;"/>

### Md Salek Miah
**Statistician · Epidemiologist · ML Researcher · GBD Collaborator, IHME**

<a href="https://tb-xai-salek.streamlit.app/"><img src="https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=0d1117"/></a>
<a href="https://github.com/muhammadsalek"><img src="https://img.shields.io/badge/GitHub-muhammadsalek-181717?style=for-the-badge&logo=github&logoColor=white&labelColor=0d1117"/></a>
<a href="https://orcid.org/0009-0005-5973-461X"><img src="https://img.shields.io/badge/ORCID-0009--0005--5973--461X-A6CE39?style=for-the-badge&logo=orcid&logoColor=white&labelColor=0d1117"/></a>
<a href="https://www.youtube.com/@SalekResearch"><img src="https://img.shields.io/badge/YouTube-Salek%20Data%20Lab-FF0000?style=for-the-badge&logo=youtube&logoColor=white&labelColor=0d1117"/></a>

<a href="https://tb-xai-salek.streamlit.app/">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=3000&pause=1000&color=4F9DFF&center=true&vCenter=true&width=640&lines=ConvNeXtTiny+%E2%86%92+Temperature+Scaling+%E2%86%92+Grad-CAM%2B%2B;YOLO26n+Lesion+Localization+vs.+Saliency+Maps;Failure+%C2%B7+Subgroup+%C2%B7+Calibration+Analysis;Chest+X-ray+%E2%86%92+Explainable+Output" alt="Typing SVG" />
</a>

</div>

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge&labelColor=0d1117)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-orange?style=for-the-badge&labelColor=0d1117)]()
[![Release](https://img.shields.io/badge/Release-v1.0--models-4F9DFF?style=for-the-badge&labelColor=0d1117)](https://github.com/muhammadsalek/TB-XAI/releases)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch/TF](https://img.shields.io/badge/Deep_Learning-ConvNeXtTiny-EE4C2C?style=flat-square&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO26n-Detection-00FFFF?style=flat-square&logoColor=black)
![Grad-CAM](https://img.shields.io/badge/XAI-Grad--CAM%20%2F%20Grad--CAM++-8b5cf6?style=flat-square&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

> **Research status.** TB-XAI is a research and educational prototype. It is **not** a diagnostic tool. See [Interpreting the Results](#interpreting-the-results) before drawing conclusions from any figure below.

---

## Overview

TB-XAI is an end-to-end chest X-ray analysis pipeline that combines deep-learning classification, probability calibration, explainable AI, and object detection. It classifies single vs. multiple annotated TB lesions, explains predictions with Grad-CAM and Grad-CAM++, localizes lesions with YOLO26n, and quantitatively compares saliency-based explanations against detector-based localization.

**Pipeline:**

```
        Chest X-ray
            │
            ▼
   ConvNeXtTiny Classification        single vs. multiple annotated TB lesions
            │
            ▼
     Temperature Scaling              probability calibration
            │
            ▼
   Grad-CAM / Grad-CAM++              saliency-based explanation
            │
            ▼
   YOLO26n Lesion Localization        bounding-box detection
            │
            ▼
   Localization Comparison            IoU: YOLO26 vs. Grad-CAM vs. Grad-CAM++
            │
            ▼
   Failure & Subgroup Analysis        error cases · calibration · subgroups
            │
            ▼
   Streamlit Web Application          upload · classify · explain · localize
```

---

## Models

| Model | Role |
|:--|:--|
| **ConvNeXtTiny** | Classification of single vs. multiple annotated TB lesions |
| **Temperature scaling** | Post-hoc calibration for probability reliability |
| **Grad-CAM / Grad-CAM++** | Visual explanations of classifier decisions |
| **YOLO26n** | TB lesion localization (bounding boxes) |

---

## Main Results

### Classification (ConvNeXtTiny)

| Metric | Test Value |
|:--|:--:|
| AUROC | 0.5333 |
| AUPRC | 0.5847 |
| Sensitivity | 0.5660 |
| Specificity | 0.4340 |
| Temperature | 0.9915 |
| Decision threshold | 0.4959 |

### Localization (YOLO26n)

| Metric | Test Value |
|:--|:--:|
| mAP@50 | 0.7098 |
| mAP@50–95 | 0.3079 |
| Precision | 0.7357 |
| Recall | 0.6503 |
| Selected confidence threshold | 0.40 |

### Localization Comparison

| Method | Mean IoU |
|:--|:--:|
| **YOLO26** | **0.5386** |
| Grad-CAM | 0.0436 |
| Grad-CAM++ | 0.0370 |

### Interpreting the Results

- **Localization is the strong component.** YOLO26n reaches mAP@50 of 0.71, and its mean IoU with annotated lesions is over 12× higher than either Grad-CAM variant.
- **Classification is near chance.** An AUROC of 0.53 means the ConvNeXtTiny classifier separates single-lesion from multiple-lesion cases only marginally better than random. The temperature (0.99) shows the model was already close to calibrated, but a calibrated near-chance classifier is not a reliable one.
- **Saliency maps are not localizers.** Grad-CAM and Grad-CAM++ highlight regions that influence the classifier, which does not match annotated lesion boxes (IoU below 0.05). This supports using a dedicated detector when spatial accuracy matters.

---

## Analyses Included

| Analysis | Description |
|:--|:--|
| **Calibration** | Temperature scaling and reliability assessment |
| **Explainability** | Grad-CAM and Grad-CAM++ heatmaps |
| **Localization** | YOLO26n detection with IoU comparison against saliency maps |
| **Failure analysis** | Inspection of misclassified and missed cases |
| **Subgroup analysis** | Performance across data subgroups |
| **Mathematical interpretation** | Gradients, PCA/SVD, and backpropagation (`math/`) |
| **Dataset splits** | Split manifests and figures for reproducibility (`results/`, `figures/`) |

---

## Repository Structure

```
TB-XAI/
│
├── .devcontainer/          ← Dev Container configuration
├── figures/                ← Result and dataset-split figures
├── math/                   ← Advanced mathematical analysis figures
├── models/                 ← YOLO26n trained detector
├── notebooks/              ← End-to-end TB-XAI notebook
├── results/                ← Metrics and dataset split manifests
├── streamlit_app.py        ← Interactive web application
├── packages.txt            ← System-level dependencies
├── requirements.txt        ← Python dependencies
└── README.md
```

---

## Model Weights

| Model | Location |
|:--|:--|
| YOLO26n detector | [`models/`](models/) directory |
| ConvNeXtTiny classifier | GitHub Release [`v1.0-models`](https://github.com/muhammadsalek/TB-XAI/releases) |

---

## Quick Start

```bash
git clone https://github.com/muhammadsalek/TB-XAI.git
cd TB-XAI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Download the ConvNeXtTiny checkpoint from the `v1.0-models` release and place it where `streamlit_app.py` expects it. To reproduce the full analysis, open the notebook in `notebooks/`.

**Live demo:** [tb-xai-salek.streamlit.app](https://tb-xai-salek.streamlit.app/)

---

## Key Highlights

| Feature | Details |
|:--|:--|
| **Task** | Lesion-level TB chest X-ray analysis |
| **Classifier** | ConvNeXtTiny (single vs. multiple annotated lesions) |
| **Calibration** | Temperature scaling |
| **Explainability** | Grad-CAM · Grad-CAM++ |
| **Detector** | YOLO26n |
| **Evaluation** | AUROC/AUPRC · mAP · IoU comparison · failure and subgroup analysis |
| **Deployment** | Streamlit web app |

---

## Citation

```bibtex
@software{miah_tb_xai_2026,
  author = {Miah, Md Salek},
  title  = {TB-XAI: Explainable Tuberculosis Lesion Analysis from Chest X-rays},
  year   = {2026},
  url    = {https://github.com/muhammadsalek/TB-XAI},
  note   = {Research prototype}
}
```

---

## Disclaimer

This project is intended for research and educational purposes only.

It is not intended for clinical diagnosis or medical decision-making.

---

## Author

<table>
<tr>
<td width="110" align="center">
<img src="https://avatars.githubusercontent.com/u/180872571?v=4" width="90" style="border-radius:50%;"/>
</td>
<td>

**Md Salek Miah**
Department of Statistics, Shahjalal University of Science and Technology (SUST), Sylhet-3114, Bangladesh
Biostatistics, Epidemiology, and Public Health Research Group
📧 [saleksta@gmail.com](mailto:saleksta@gmail.com)

[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--5973--461X-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0005-5973-461X)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Md_Salek_Miah-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-salek-miah-b34309329/)

</td>
</tr>
</table>

---

## License

MIT License. Copyright (c) 2026 Md Salek Miah.

<div align="center">

*⭐ Star this repo if it helped your research!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%"/>

</div>
