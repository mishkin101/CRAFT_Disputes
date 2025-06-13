# CRAFT – Custom KODIS Implementation

This repository contains our tailored implementation of the **Conversational Recurrent Architecture for ForecasTing (CRAFT)** neural model, originally introduced in the EMNLP 2019 paper [Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop](https://arxiv.org/abs/1909.01362) to the dispute resolution domain. We pre-train the CRAFT model architecture with a custom corpus of CaSiNo, Deal no Deal, and KODIS dialogs and finetune on the  **KODIS** dataset to research whether we can learn unsupervised representation of conversational dynamics in negotiation-based dialogues and expoit the structure via supervised learning in fine-tune for predicting for outcomes in Dispute resolution (KODIS).

---

# 1. Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Ray 2.x (Tune & AIR)
- MLflow 2.x
- ConvoKit 3.x
scikit-learn, pandas, matplotlib, NLTK
---

# 2. Fine-tuning on KODIS with Ray Tune & MLflow
Everything is orchestrated in `src/runners/raytune.py`. By default it will:

- Perform k-fold cross-validation on your train split of KODIS.

- Report per-fold batch losses and mean validation metrics each epoch via tune.report.

- Use ASHAScheduler to early-stop underperforming hyperparameter trials.

- Log all parameters, metrics, and model artifacts to MLflow.

---
