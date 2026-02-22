# Pneumonia Detection & Analysis Pipeline – PneumoniaMNIST  
**AlfaisalX Postdoctoral Technical Challenge Submission**  
**AI Medical Imaging, Visual Language Models, and Semantic Retrieval**

**Author:** FARMAN  
**GitHub:** https://github.com/Farman335/pneumonia-mnist-pipeline  
**Date:** February 21, 2026

This repository contains a complete end-to-end pipeline for pneumonia detection and analysis using the **PneumoniaMNIST** dataset (MedMNIST v2). It covers all three tasks of the challenge:

- **Task 1** — Classification with Vision Transformer (ViT-B/16)  
- **Task 2** — Medical Report Generation with MedGemma-4b-it  
- **Task 3** — Content-Based Image Retrieval (CBIR) with FAISS

## Key Results Summary

| Task | Technology                       | Primary Metric(s)                          | Highlight |
|------|----------------------------------|--------------------------------------------|-----------|
| 1    | ViT-B/16 (ImageNet pre-trained) | Accuracy: 91.51%<br>Recall: 97.98%<br>AUC: 0.9818 | Outstanding sensitivity – very few missed cases |
| 2    | MedGemma-4b-it                   | Qualitative: structured radiology reports  | Explains misclassifications & complements Task 1 |
| 3    | ViT-B/16 embeddings + FAISS      | Precision@1: 0.92<br>Precision@5: 0.89     | Strong grouping of similar pathological cases |

## Project Structure (after unzipping uploaded files)

After unzipping the uploaded .zip files, the working directory should look like this:

pneumonia-mnist-pipeline/
├── data/                       # (optional – auto-downloaded anyway)
├── models1/                     # saved weights (from models1.zip)
│   └── shared link of model
├── task1_classification/       # from task1_classification.zip
│   └── task1_classification_report.md
├── task2_report_generation/    # from task2_report_generation.zip
│   └── task2_report_generation.md
├── task3_retrieval/            # from task3_retrieval.zip
│   └── task3_retrieval_system.md
├── reports/                    # Training vs Validation loss, Confusion Matrix, & ROC Curve (from reports.zip)
├── notebooks/                  # all .ipynb files (from notebooks.zip)
│   ├── task_1_classification.ipynb
│   ├── task_2_report_generation.ipynb
│   └── task_3_retrieval_(1).ipynb
├── requirements.txt            # (rename requirements.txt.txt if needed)
└── ReadMe.md

**Note:** The dataset details is provided into **pneumonia-mnist_dataset_info** file by unzipping `data.zip`.

## Prerequisites

- **Recommended:** Google Colab (free T4 GPU)  
- **Alternative:** Local Python 3.9+ with NVIDIA GPU + CUDA

## Environment Setup

1. **Clone the repository**

```bash
git clone https://github.com/Farman335/pneumonia-mnist-pipeline.git
cd pneumonia-mnist-pipeline

2. **Unzip the uploaded files (if you downloaded them separately)
unzip data.zip
unzip models1.zip
unzip notebooks.zip
unzip reports.zip
unzip task1_classification.zip
unzip task2_report_generation.zip
unzip task3_retrieval.zip

3. **Install dependencies**
pip install -r requirements.txt

requirements.txt contents (already uploaded):
torch>=2.0.0
torchvision>=0.15.0
medmnist>=2.2.3
transformers>=4.35.0
huggingface_hub>=0.19.0
faiss-cpu>=1.7.4
matplotlib>=3.7.0
numpy>=1.24.0
scikit-learn>=1.3.0
pillow>=10.0.0
tqdm>=4.66.0

4. **Colab one-liner (run in first cell of any notebook)**
!pip install medmnist torch torchvision transformers huggingface_hub faiss-cpu scikit-learn matplotlib numpy pillow tqdm

5. **Hugging Face Login – required only for Task 2**

Accept gated model: https://huggingface.co/google/medgemma-4b-it
Generate token: https://huggingface.co/settings/tokens 
 
In notebook:
from huggingface_hub import login
login:Name: HF_TOKEN
Value: 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  # ← your token

##Running Each Task
**Task 1 – Classification (ViT-B/16)**
Notebook: notebooks/task_1_classification.ipynb
(or unzipped path: task1_classification/task_1_classification.ipynb)

Open in Colab
Run all cells
Training: ~15–25 min on T4 GPU
Outputs: Test metrics (91.51% acc, 97.98% recall, 0.9818 AUC)
Plots → saved to reports/
Model weights → models/vit_pneumonia_model.pth

Report: task1_classification/task1_classification_report.md

**Task 2 – Medical Report Generation (MedGemma)**
Notebook: notebooks/task_2_report_generation.ipynb
(or unzipped path)

Open in Colab (GPU required)
Authenticate HF (above)
Run all cells
Outputs:
10 sample images + generated reports
Console logs of full text
Report: task2_report_generation/task2_report_generation.md

**Task 3 – Semantic Image Retrieval (ViT + FAISS)**
Notebook: notebooks/task_3_retrieval_(1).ipynb
(or unzipped path)

Open and run all cells
Builds FAISS index (~1–2 min)
Shows retrieval examples

Outputs: Top-k indices + distances
Visual grid (query + top-5 results)

Report: task3_retrieval/task3_retrieval_system.md (P@1 = 0.92, P@5 = 0.89)


