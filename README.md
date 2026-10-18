# RetinaAI — Explainable AI for Diabetic Retinopathy Screening

[![CI Pipeline](https://github.com/navjot-singh-2005/RetinaAI-Explainable-DR-Screening/actions/workflows/ci.yml/badge.svg)](https://github.com/navjot-singh-2005/RetinaAI-Explainable-DR-Screening/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RetinaAI** is a clinical-grade, safety-critical computer vision platform designed for automated Diabetic Retinopathy (DR) grading and triage from fundus photography. Powered by an EfficientNetV2-S backbone, the system integrates **Grad-CAM++ visual explainability**, **Monte Carlo Dropout epistemic uncertainty**, **energy-based out-of-distribution (OOD) detection**, **geometric fundus domain validation**, and a **calibrated clinical decision engine**.

---

## 🌟 Key Capabilities

- **5-Grade DR Classification**: ICDR grading (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).
- **Geometric Domain & Quality Gate**: Rejects non-fundus images (documents, outdoor scenes) via circular FOV contour analysis and dark border metrics; screens for severe blur or lighting defects.
- **Explainable AI (XAI)**: Grad-CAM++ lesion heatmaps overlaid on retinal structures, verified via spatial entropy and deletion/insertion curves.
- **Epistemic Uncertainty Estimation**: Fast head-cached Monte Carlo Dropout (15 stochastic forward passes) yielding entropy and predictive standard deviation.
- **Energy-Based OOD Scoring**: Free-energy metric detects corruptions, non-retinal imagery, and artifact-heavy inputs.
- **Clinical Decision Engine (Policy v2.1)**: Operating threshold calibrated for **$\ge 90\%$ referable DR sensitivity** with automated triage (`ROUTINE SCREENING`, `REFERRAL RECOMMENDED`, `URGENT REFERRAL`, `UNGRADABLE`).
- **Smartphone Multi-Frame Fusion**: Quality-weighted aggregation over video or burst frames to eliminate false severity spikes.
- **Deployment Ready**: Dual-interface support with an interactive **Streamlit Dashboard**, a **FastAPI REST API**, **Docker** containers (CPU & GPU), and **ONNX Runtime** export.

---

## 🏗️ System Pipeline Architecture

```
                                  [ Input Image ]
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │  Geometric & Quality  │ ──► [REJECT: Invalid / Blurry]
                             │       Domain Gate     │
                             └───────────────────────┘
                                         │ Pass
                                         ▼
                             ┌───────────────────────┐
                             │ Preprocessing Engine  │ (Circle Crop, Ben Graham CLAHE,
                             │   (512×512 Standard)  │  ImageNet Normalization)
                             └───────────────────────┘
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │   EfficientNetV2-S    │
                             │ Backbone Feature Extr │
                             └───────────────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 │                       │                       │
                 ▼                       ▼                       ▼
      ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
      │  Calibrated Head    │ │  Fast MC Dropout    │ │  Grad-CAM++ Engine  │
      │   Classification    │ │  Uncertainty Engine │ │  Visual Heatmaps    │
      └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         │
                                         ▼
                             ┌───────────────────────┐
                             │  Clinical Decision    │
                             │  Engine (Policy v2.1) │
                             └───────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
       [ Interactive Streamlit App ]                 [ FastAPI Microservice ]
```

---

## 📂 Repository Structure

```
.
├── .github/workflows/        # Automated GitHub Actions CI pipeline
├── .streamlit/               # Streamlit theme and UI configurations
├── api/                      # Production FastAPI service (endpoints, schemas)
├── data/                     # Screening history DB & dataset split definitions
├── docker/                   # Dockerfiles (Dockerfile.cpu & Dockerfile.gpu)
├── docs/                     # Architectural documentation, Model Card, and results
│   ├── results/              # Calibration, error analysis, and benchmark JSONs
│   ├── CHANGELOG_UPGRADE.md  # Detailed phase changelogs
│   └── MODEL_CARD.md         # Comprehensive clinical model card
├── experiments/              # Inference audit scripts and verification runs
├── models/                   # Model weight checkpoints and deployment binaries
├── scripts/                  # Evaluation, calibration, export, and robustness scripts
├── src/                      # Core RetinaAI Python package
│   ├── config.py             # System hyperparameters, paths, and clinical policies
│   ├── dataset.py            # PyTorch datasets & Albumentations augmentations
│   ├── evaluate.py           # Evaluation metrics (QWK, sensitivity, specificity)
│   ├── explain.py            # Grad-CAM++ implementation & spatial entropy
│   ├── history.py            # SQLite screening session logger & privacy mode
│   ├── model.py              # EfficientNetV2-S model definition & weight loading
│   ├── multiframe.py         # Multi-frame burst/video fusion algorithms
│   ├── predict.py            # Inference orchestrator & clinical decision rules
│   ├── preprocessing.py      # Retinal circle crop & Ben Graham CLAHE
│   ├── provenance.py         # Reproducibility tracking & SHA-256 weight locks
│   ├── quality.py            # Geometric FOV validation & blur detection
│   ├── uncertainty.py        # Fast MC Dropout & Free-Energy OOD estimation
│   └── utils.py              # Logging, visualization, and image converters
├── tests/                    # 10-phase automated Pytest integration & unit test suite
├── app.py                    # Streamlit web application dashboard
├── requirements.txt          # Python runtime dependencies
└── requirements.lock         # Exact pinned dependency lockfile
```

---

## 🚀 Quick Start

### 1. Local Setup

```powershell
# Clone the repository
git clone https://github.com/navjot-singh-2005/RetinaAI-Explainable-DR-Screening.git
cd RetinaAI-Explainable-DR-Screening

# Create & activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Linux/macOS: source venv/bin/activate

# Install dependencies (CPU or CUDA)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or /cpu
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the clinical screening interface.

### 3. Launch the REST API Service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

### 4. Run via Docker

```bash
# Build & run CPU container
docker build -f docker/Dockerfile.cpu -t retinaai:latest .
docker run -p 8501:8501 retinaai:latest
```

---

## 🧪 Running the Test Suite

RetinaAI includes a rigorous 10-phase unit and property-based test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/test_phase3.py -v   # Geometric gating & domain rejection
pytest tests/test_phase7.py -v   # Clinical invariant testing (Hypothesis)
```

---

## 📊 Clinical Performance & Invariant Summary

| Metric | Target | Verified Performance |
| :--- | :--- | :--- |
| **Referable DR Sensitivity** | $\ge 90.0\%$ | **$93.18\%$** ($95\%$ CI: $[89.8\%, 96.6\%]$) |
| **Referable DR Specificity** | $\ge 70.0\%$ | **$73.86\%$** |
| **Expected Calibration Error (ECE)** | $< 5.0\%$ | **$2.68\%$** (Post Temperature Scaling) |
| **Quadratic Weighted Kappa (QWK)** | $\ge 0.80$ | **$0.865$** |
| **Non-Fundus False Accept Rate (FAR)** | $0.0\%$ | **$0.0\%$** (Rigid Geometric FOV Rejection) |
| **Inference Latency** | $< 100\text{ ms}$ | **$28\text{ ms}$** (GPU) / **$65\text{ ms}$** (CPU) |

---

## 🛡️ Disclaimer

*RetinaAI is an investigational software platform developed for research and screening triage assistance. It is intended to assist medical professionals, not replace formal clinical examinations by licensed ophthalmologists.*

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
