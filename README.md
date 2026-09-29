# AcousticBiomarker-GH

**Cough-Acoustic Screening for Respiratory Conditions: On-Device Inference with TensorFlow Lite.**

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://acoustic-biomarker-gh-salek05.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow_Lite-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-orange?style=flat-square)]()

> **Research status.** This is a research prototype and interface demonstrator, not a validated diagnostic device. Model inference on uploaded audio is real. The Advanced Analytics and Explainability tabs currently show illustrative placeholder statistics. See [Validation Status](#validation-status) before citing any performance figure.

## Overview

AcousticBiomarker-GH combines audio signal processing, a quantized deep-learning classifier, rule-based clinical triage, and structured reporting for cough-based respiratory screening.

A 3-second cough recording (`.wav`) is converted to a log-mel spectrogram and passed through a TFLite MobileNetV2 model that outputs probabilities for three classes: Healthy, Symptomatic, and COVID-19. A triage layer maps these probabilities to a clinical action, and the app exports the result as PDF, JSON, CSV, or a compact telemetry packet for low-bandwidth settings.

## Pipeline

Cough recording (.wav)
→ Resample to 16 kHz, pad/truncate to 3.0 s
→ Peak normalization
→ 128-band log-mel spectrogram
→ 128 × 94 × 3 input tensor
→ Quantized MobileNetV2 (TFLite, INT8)
→ Softmax: Healthy / Symptomatic / COVID-19
→ Rule-based triage
→ Clinical report and export

## Models

- **MobileNetV2** (quantized, ~2.3M parameters) for 3-class cough classification
- **TensorFlow Lite 2.15** interpreter for on-device inference
- Stated training corpora: COUGHVID and Virufy (per in-app system panel; see [Validation Status](#validation-status))

## Signal Processing

| Stage | Parameter |
|---|---|
| Sample rate | 16,000 Hz |
| Clip duration | 3.0 s (48,000 samples) |
| Normalization | Peak amplitude |
| Mel bands | 128 |
| FFT size | 2,048 |
| Hop length | 512 |
| Frequency range | 0-8,000 Hz |
| Model input | 128 × 94 × 3 (log-mel replicated across 3 channels) |

## Main Results

### Live Components

| Component | Status |
|---|---|
| Audio preprocessing (`librosa`) | Live, deterministic |
| TFLite MobileNetV2 inference | Live forward pass |
| Triage thresholds | Live, rule-based |
| Inference latency | ~10.6 ms per clip (in-app) |
| Telemetry packet | 19 bytes, base64-encoded |
| Exports | JSON, CSV, TXT, PDF with SHA-256 hash |

### Clinical Triage Logic

| Condition | Level | Action Code | Suggested Action |
|---|---|:---:|---|
| P(COVID) ≥ 0.70 | Critical | 1 | Immediate clinical evaluation, isolation, urgent care escalation |
| P(COVID) ≥ 0.35 or P(Symptomatic) > 0.50 | Moderate | 2 | Telemedicine consult, PCR testing within 24 h, self-isolation |
| Otherwise | Stable | 3 | Routine monitoring, standard precautions |

These thresholds are fixed constants, not calibrated cut-points. They are an interface convention to be tuned with decision-curve analysis on real data.

## Validation Status

| Component | Status |
|---|:---:|
| AUC-ROC / Sensitivity / Specificity / MCC table | Placeholder (fixed demonstration values) |
| ROC and calibration curves | Simulated for interface design |
| SHAP-style feature importance | Randomly generated each run |
| Confusion matrix | Static placeholder |
| Training/evaluation split, class balance | Not yet documented in this repository |

The pipeline and interface are real and reproducible. The reported clinical performance metrics are placeholders describing the target reporting format, not results from an external validation study.

## Dashboard Tabs

| Tab | Contents |
|---|---|
| Spectrogram | Log-mel spectrogram and raw waveform |
| Advanced Analytics | Metrics table, triage distribution, ROC and calibration curves, Kappa / F1 / MCC (placeholders) |
| Explainability | Feature-importance chart and confusion matrix (placeholders) |
| Telemetry | 19-byte packet (`struct.pack('>IBBfffB', ...)`): device ID, age, gender, class probabilities, action code |
| Export | JSON, CSV, and text report with SHA-256 integrity hash |
| PDF Report | Multi-section clinical PDF via `reportlab` |

## Repository Structure

```
acoustic-biomarker-gh/
├── app.py                              # Streamlit application (v2.1)
├── acoustic_biomarker_quantized.tflite # Quantized MobileNetV2 model
├── requirements.txt
├── runtime.txt
└── README.md
```

Key functions in `app.py`: `load_model()`, `preprocess_audio()`, `get_triage_level()`, `get_recommendation()`, `generate_telemetry()`, `generate_pdf_report()`.

## Installation

```bash
git clone https://github.com/muhammadsalek/acoustic-biomarker-gh.git
cd acoustic-biomarker-gh
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**requirements.txt**

```txt
streamlit>=1.30.0
numpy>=1.23.5
librosa>=0.10.1
tensorflow>=2.15.0
matplotlib>=3.7.0
seaborn>=0.13.0
scikit-learn>=1.2.2
pandas>=2.0.0
plotly>=5.18.0
scipy>=1.11.0
reportlab>=4.0.0
```

**runtime.txt**

```txt
python-3.10
```

## Privacy

- Patient data is held in `st.session_state` (in memory) for the browser session only.
- Audio is not sent to any external API; inference runs locally through the bundled TFLite interpreter.
- Refreshing the page resets the session, so export anything needed beforehand.

## Roadmap

- [x] End-to-end audio to spectrogram to TFLite inference
- [x] Rule-based triage layer
- [x] JSON / CSV / TXT / PDF export with SHA-256 stamp
- [x] Low-bandwidth binary telemetry
- [ ] Replace placeholder analytics with metrics from a documented held-out split
- [ ] Replace simulated ROC/calibration curves with curves fit to real predictions and labels
- [ ] Replace random SHAP panel with real SHAP or Integrated Gradients attribution
- [ ] Publish a model and data card (splits, class balance, deduplication across COUGHVID/Virufy, failure modes)
- [ ] External prospective validation on a geographically distinct cohort
- [ ] Bengali / English UI

## Citation

```bibtex
@software{miah_acoustic_biomarker_gh,
  author = {Miah, Md Salek},
  title  = {AcousticBiomarker-GH: A TensorFlow Lite Cough-Acoustic Screening Interface},
  year   = {2026},
  url    = {https://github.com/muhammadsalek/acoustic-biomarker-gh},
  note   = {Research prototype; see Validation Status for verified vs. illustrative components}
}
```

## Disclaimer

This project is intended for research and educational purposes only.

It is not intended for clinical diagnosis or medical decision-making. All clinical decisions should be validated by healthcare professionals.

## Contact

Md Salek Miah · [saleksta@gmail.com](mailto:saleksta@gmail.com) · [GitHub](https://github.com/muhammadsalek) · [ORCID](https://orcid.org/0009-0005-5973-461X) · [LinkedIn](https://www.linkedin.com/in/md-salek-miah-b34309329/)

## License

Released under the MIT License.
