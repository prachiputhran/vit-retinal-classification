# Evaluating Explainable AI Methods in High-Stakes Healthcare ML Systems

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Research-orange)](Research_Paper.pdf)

> A quantitative evaluation of SHAP and LIME post-hoc explainability methods across
> Logistic Regression, Random Forest, and Neural Network models on two healthcare datasets.

**Author:** Prachi Puthran · B.Tech AI & DS · RAIT, Navi Mumbai  
**Guided by:** Prof. Sandeep Sangle · Department of CSE, RAIT  

---

## Overview

Black-box ML models increasingly power high-stakes healthcare decisions, yet their
internal reasoning remains opaque. This project provides a rigorous, quantitative
comparison of two widely-used post-hoc XAI tools — **SHAP** and **LIME** — evaluated
across three dimensions: consistency, stability under input perturbation, and rank
correlation with an interpretable ground-truth baseline.

---

## Datasets

| Dataset | Instances | Features | Task |
|---|---|---|---|
| Breast Cancer Wisconsin | 569 | 30 | Binary (Malignant / Benign) |
| Heart Disease UCI | 297 | 13 | Binary (Presence / Absence) |

---

## Key Results

### Model Accuracy

| Dataset | Model | Accuracy | F1 Score |
|---|---|---|---|
| Breast Cancer Wisconsin | Logistic Regression | 98.25% | 0.9825 |
| Breast Cancer Wisconsin | Random Forest | 95.61% | 0.9560 |
| Breast Cancer Wisconsin | Neural Network | 96.49% | 0.9651 |
| Heart Disease UCI | Logistic Regression | 83.33% | 0.8328 |
| Heart Disease UCI | Random Forest | 86.67% | 0.8662 |
| Heart Disease UCI | Neural Network | 81.67% | 0.8168 |

### Stability Under Input Perturbation (↑ higher is better)

| Dataset | Model | SHAP Stability | LIME Stability |
|---|---|---|---|
| Breast Cancer Wisconsin | Logistic Regression | 0.9867 | 0.6876 |
| Breast Cancer Wisconsin | Random Forest | 0.9922 | 0.6937 |
| Breast Cancer Wisconsin | Neural Network | 0.9844 | 0.6793 |
| Heart Disease UCI | Logistic Regression | 0.9467 | 0.7053 |
| Heart Disease UCI | Random Forest | 1.0000 | 0.7065 |
| Heart Disease UCI | Neural Network | 0.9586 | 0.6888 |

**SHAP stability: 0.9467–1.0000 vs LIME stability: 0.6793–0.7065** across all
dataset/model combinations.

---

## Project Structure