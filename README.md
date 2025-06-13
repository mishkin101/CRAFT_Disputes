# CRAFT – Custom KODIS Implementation

This repository contains our tailored implementation of the **Conversational Recurrent Architecture for ForecasTing (CRAFT)** neural model, originally introduced in the EMNLP 2019 paper “Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop.” Here, we show how to plug in your **KODIS** dataset in place of the original demo corpora.

---

## UPDATE April 2024

If you have previously run this code on any other dataset, please pull the latest changes—especially if you worked on **CGA-CMV**. A bug affecting CGA-CMV processing was fixed on April 24, 2024. Your results on CMV may change (for the better!), but **KODIS**–based runs are unaffected by that issue.

---

## Prerequisites

- **Python** ≥ 3.6 (validated up through 3.8)  
- **PyTorch** ≥ 1.5  
- **ConvoKit** ≥ 1.3  
- **NLTK**, **pandas**, **matplotlib**  

We recommend using Anaconda:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install convokit
conda install nltk pandas matplotlib
