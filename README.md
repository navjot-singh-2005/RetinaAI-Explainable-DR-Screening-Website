# RetinaAI — Explainable AI for Diabetic Retinopathy Screening

## About the Project

RetinaAI is an AI-assisted diabetic retinopathy (DR) screening platform that uses deep learning to classify retinal fundus images into five severity grades. The project integrates Explainable AI (XAI), uncertainty estimation, image quality assessment, and out-of-distribution (OOD) detection to make predictions more interpretable and support reliability-aware screening.

The platform uses **EfficientNetV2-S** for image classification and provides an interactive dashboard built with Streamlit.

## Key Features

- **Five-Class Classification:** Identifies five diabetic retinopathy severity grades.
- **EfficientNetV2-S:** Uses a deep learning model fine-tuned for retinal image classification.
- **Grad-CAM++ Explainability:** Generates heatmaps highlighting image regions contributing to predictions.
- **MC Dropout:** Estimates model uncertainty through multiple stochastic predictions.
- **OOD Detection:** Uses energy-based scoring to help identify potentially unfamiliar inputs.
- **Image Quality Assessment:** Evaluates image quality before classification.
- **Smartphone Multi-Frame Analysis:** Supports extracting and evaluating multiple frames from smartphone video.
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
- **Image Processing:** OpenCV
- **Explainable AI:** Grad-CAM++
- **Uncertainty Estimation:** MC Dropout
- **User Interface:** Streamlit
- **Additional Libraries:** NumPy, scikit-learn, timm

## Dataset

The model training experiment used a combined retinal fundus dataset containing images from APTOS, DDR, IDRiD, EyePACS, and Messidor.

Dataset source: [Fundus APTOS + DDR + IDRiD + EyePACS + Messidor](https://www.kaggle.com/datasets/sehastrajits/fundus-aptosddridirdeyepacsmessidor)

The dataset was organized into training, validation, and testing splits. Refer to the original dataset source for licensing and usage conditions.

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd RetinaAI-Explainable-DR-Screening
```

Replace `YOUR_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Install Dependencies

It is recommended to create a Python virtual environment before installing dependencies.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

Ensure the trained model checkpoint is available at the location expected by the application.

## Explainable AI and Reliability

RetinaAI combines classification with supporting reliability features:

- **Grad-CAM++** helps visualize image regions contributing to the model's prediction.
- **MC Dropout** provides an estimate of predictive uncertainty.
- **Energy-Based OOD Detection** helps assess whether an input may differ from the expected data distribution.
- **Image Quality Assessment** helps identify images that may be unsuitable for analysis.

These components can help users interpret model behavior, but they do not guarantee correct predictions or reliable rejection of every non-fundus image.

## Future Improvements

- Improve detection of non-fundus images before classification.
- Evaluate performance on independent datasets and different camera sources.
- Improve classification of visually similar severity grades.
- Enhance smartphone image capture and multi-frame aggregation.
- Improve uncertainty calibration and OOD detection.
- Conduct external evaluation and clinical validation with qualified professionals.

## Limitations

RetinaAI is an experimental AI-assisted screening platform. Its predictions may be incorrect, and its reliability features have limitations. Grad-CAM++ visualizations do not independently confirm lesions, while uncertainty and OOD scores are not guarantees of correctness.

## Disclaimer

**For research and educational purposes only.** RetinaAI is not a clinically validated diagnostic system and does not replace professional ophthalmic examination, medical advice, or diagnosis. Screening results should be reviewed by qualified healthcare professionals.

## Acknowledgements

This project builds on the EfficientNetV2 architecture and publicly available retinal fundus datasets. Please consult the original research publications and dataset sources for further details.

