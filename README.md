# RetinaAI — Explainable AI for Diabetic Retinopathy Screening

🚀 **[TRY LIVE DEMO — Open RetinaAI](https://retinaai-explainable-dr-screening-website-fsrv3isas7vp3mbqhwtv.streamlit.app/)**

---

## About the Project

RetinaAI is an AI-assisted diabetic retinopathy (DR) screening platform that uses deep learning to classify retinal fundus images into five severity grades. The project integrates Explainable AI (XAI), uncertainty estimation, image quality assessment, and out-of-distribution (OOD) detection to make predictions more interpretable and support reliability-aware screening.

The platform uses **EfficientNetV2-S** for image classification and provides an interactive dashboard built with Streamlit.

## Key Features

- **Five-Class Classification:** Classifies diabetic retinopathy into five severity grades.
- **EfficientNetV2-S:** Uses a deep learning model fine-tuned for retinal image classification.
- **Grad-CAM++ Explainability:** Generates heatmaps highlighting image regions contributing to predictions.
- **MC Dropout:** Estimates model uncertainty through multiple stochastic predictions.
- **OOD Detection:** Uses energy-based scoring to help identify potentially unfamiliar inputs.
- **Image Quality Assessment:** Evaluates image quality before classification.
- **Smartphone Multi-Frame Analysis:** Supports extracting and evaluating multiple frames from smartphone video, where implemented.
- **Interactive Dashboard:** Displays predictions, confidence scores, and visual explanations through Streamlit.

## DR Severity Classification

| Grade | Description |
|---|---|
| Grade 0 | No Diabetic Retinopathy |
| Grade 1 | Mild Diabetic Retinopathy |
| Grade 2 | Moderate Diabetic Retinopathy |
| Grade 3 | Severe Diabetic Retinopathy |
| Grade 4 | Proliferative Diabetic Retinopathy (PDR) |

## System Workflow

1. **Image Input:** Upload a retinal fundus image or provide supported smartphone video input.
2. **Quality Assessment:** Check whether the image is suitable for analysis.
3. **Preprocessing:** Resize and prepare the image for the model.
4. **Classification:** EfficientNetV2-S predicts one of the five DR severity grades.
5. **Explainability:** Grad-CAM++ generates a heatmap of regions contributing to the prediction.
6. **Reliability Assessment:** MC Dropout estimates uncertainty, while energy-based OOD scoring evaluates unfamiliar inputs.
7. **Result Display:** The Streamlit dashboard presents the prediction and supporting visual information.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Model Architecture:** EfficientNetV2-S
- **Image Processing:** OpenCV, Pillow
- **Explainable AI:** Grad-CAM++
- **Uncertainty Estimation:** MC Dropout
- **User Interface:** Streamlit
- **Additional Libraries:** NumPy, scikit-learn, timm
- **Deployment:** Streamlit Community Cloud

## Dataset

The model training experiment used a combined retinal fundus dataset containing images from APTOS, DDR, IDRiD, EyePACS, and Messidor.

**Dataset source:** [Fundus APTOS + DDR + IDRiD + EyePACS + Messidor on Kaggle](https://www.kaggle.com/datasets/sehastrajits/fundus-aptosddridirdeyepacsmessidor)

The dataset was organized into training, validation, and testing splits. Refer to the original dataset source for licensing and usage conditions.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/navjot-singh-2005/RetinaAI-Explainable-DR-Screening.git
cd RetinaAI-Explainable-DR-Screening
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Model Checkpoint

Ensure that the trained model checkpoint is available at the location expected by the application. If the model is hosted externally, configure the required download settings and secrets.

Do not commit private access tokens or credentials to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

## Explainable AI and Reliability

- **Grad-CAM++:** Visualizes image regions contributing to model predictions.
- **MC Dropout:** Estimates predictive uncertainty through multiple stochastic predictions.
- **Energy-Based OOD Detection:** Helps assess whether an input may differ from the expected data distribution.
- **Image Quality Assessment:** Helps identify images that may be unsuitable for analysis.

These components support interpretation and reliability assessment but do not guarantee correct predictions or reliable rejection of every unfamiliar input.

## Innovation

The main contribution of RetinaAI is the integration of multiple AI-assisted screening components into a unified application:

- Five-class diabetic retinopathy grading.
- Visual explanations using Grad-CAM++.
- Predictive uncertainty estimation using MC Dropout.
- Energy-based out-of-distribution scoring.
- Fundus image quality assessment.
- An interactive screening dashboard.

The innovation lies in this integrated workflow rather than claiming that each individual technique is a newly invented algorithm.

## Future Improvements

- Improve detection of non-fundus images before classification.
- Evaluate performance on independent datasets and different camera sources.
- Improve classification of visually similar severity grades.
- Enhance smartphone image capture and multi-frame aggregation.
- Improve uncertainty calibration and OOD detection.
- Conduct external evaluation and clinical validation with qualified professionals.

## Limitations

- The model may misclassify images, particularly visually similar severity grades.
- Confidence scores and uncertainty estimates do not guarantee correctness.
- OOD detection may fail to reject some unfamiliar or non-fundus images.
- Grad-CAM++ visualizations do not independently confirm lesions.
- Performance may vary across datasets, cameras, image quality, and patient populations.
- Clinical utility and generalization require independent evaluation and validation.

## Research and References

This project builds on the EfficientNetV2 architecture and research into automated diabetic retinopathy screening, fundus image quality assessment, and explainable AI.

Please consult the original research publications and dataset sources for methodological details and attribution.

## Acknowledgements

Thanks to the researchers and organizations who developed EfficientNetV2 and made retinal fundus datasets publicly available for research.

## Disclaimer

**For research and educational purposes only.**

RetinaAI is an experimental AI-assisted screening prototype and is not a clinically validated diagnostic system. Its predictions may be incorrect. It does not replace professional ophthalmic examination, medical advice, or diagnosis. All screening results should be reviewed by qualified healthcare professionals.
