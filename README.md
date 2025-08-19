# Deep Learning Practice – Binary Classification Model

This repository contains a practice deep learning project implemented in PyTorch and Jupyter Notebook.  
The model was trained to classify samples into two categories: **Sus** and **Not Sus**.

---

## 📌 Project Overview
- Built a simple feed-forward neural network using PyTorch.  
- Trained the model with a training set, validated on a held-out validation set.  
- Evaluated performance using accuracy, ROC-AUC, precision, recall, and F1-score.  
- Goal: practice workflow from data preprocessing → model training → evaluation → metrics visualization.  

---

## ⚙️ Technologies Used
- Python 3.9+  
- PyTorch  
- TorchMetrics  
- scikit-learn  
- matplotlib / seaborn  
- Jupyter Notebook (via VS Code extension)

---

## 📊 Model Performance

| Metric       | Score   |
|--------------|---------|
| **Accuracy** | 0.9720  |
| **ROC AUC**  | 0.6682  |
| **Precision**| 0.0109  |
| **Recall**   | 0.9885  |
| **F1 Score** | 0.02167 |


  precision    recall  f1-score   support

     Not Sus       1.00      0.63      0.77    188181
         Sus       0.01      0.99      0.02       786

    accuracy                           0.63    188967
   macro avg       0.51      0.81      0.40    188967
weighted avg       1.00      0.63      0.77    188967

## Evaluation inference:
   The model has great recall, but terrible precision; it rarely misses a true "Sus", but almost all its "Sus" predictions are wrong. 
   This is because of dataset imbalance which was left unaddressed.

## 📂 Repository Structure
