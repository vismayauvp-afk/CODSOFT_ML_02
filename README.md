# Credit Card Fraud Detection

## Overview
This project implements a **machine learning model to detect credit card fraud**. Using historical transaction data, the model predicts whether a transaction is **legitimate** or **fraudulent**. The goal is to help banks and financial institutions **protect customers and reduce fraud losses**.

The model uses techniques to handle **imbalanced datasets** and evaluates predictions using metrics like **precision, recall, F1-score, and confusion matrix**.

---

## Features
- Predicts if a transaction is **fraudulent** or **legitimate**.
- Handles **class imbalance** using **SMOTE (Synthetic Minority Oversampling Technique)**.
- Real-time transaction input for **interactive predictions**.
- Produces **confusion matrix** and **classification report** for evaluation.

---

## Requirements
- Python 3.10 or higher
- Libraries:
  ```bash
  pandas
  numpy
  scikit-learn
  imbalanced-learn
