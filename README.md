# 🧠 Digit Classification using Machine Learning (MNIST)

This project demonstrates a **complete end-to-end machine learning workflow** for handwritten digit classification using the **MNIST dataset**.  
Instead of using deep learning, the focus is on **strong ML fundamentals** such as preprocessing, dimensionality reduction, classical algorithms, and proper validation.

---

## 📌 Project Overview

Handwritten digit classification is a **multi-class classification problem** where the task is to identify digits from **0 to 9** based on pixel intensity values.

In this notebook, I:
- Performed data inspection and validation
- Built a baseline model using Logistic Regression
- Improved performance using PCA + SVM
- Applied cross-validation
- Tested the final model on a **completely unseen external dataset**

---

## 📂 Dataset Description

- **Training Dataset:** `mnist_train.csv` link = https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
- **External Test Dataset:** `mnist_test.csv`
- **Target Column:** `label`
- **Features:** 784 pixel values (28×28 image flattened)

Each row represents a grayscale handwritten digit image.

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes:
- `head()` and `tail()` to inspect data
- Shape and dimensionality checks
- Data types and memory usage (`info()`)
- Statistical summary using `describe()`

EDA ensures:
- No missing values
- All features are numeric
- Correct identification of features and target

---

## ⚙️ Data Preparation

### Feature & Target Split
- **X:** Pixel intensity values
- **y:** Digit labels (0–9)

### Train-Test Split
- 80% Training data
- 20% Testing data
- Stratified split to maintain class balance

---

## 🔄 Feature Scaling

Pixel values are standardized using:
- `StandardScaler`
- Implemented with `ColumnTransformer`

Why scaling?
- Improves convergence
- Essential for SVM and distance-based models

---

## 🧪 Baseline Model: Logistic Regression

A Logistic Regression classifier is trained as a baseline:
- `max_iter = 1000`
- Trained on scaled data
- Evaluated using:
  - Confusion Matrix
  - Classification Report

Purpose:
- Establish a simple, interpretable baseline
- Compare with advanced models later

---

## 📉 Dimensionality Reduction using PCA

Since MNIST has **784 features**, PCA is applied to:
- Reduce dimensionality
- Remove noise
- Improve training speed and generalization

### PCA Configuration:
- `n_components = 100`
- Fitted on scaled training data

---

## 🚀 Advanced Model: Support Vector Machine (SVM)

After PCA, an SVM classifier is trained.

### Model Details:
- Kernel: `rbf`
- C: `5`
- Gamma: `scale`

Why SVM?
- Performs well in high-dimensional spaces
- RBF kernel captures non-linear relationships effectively

---

## 📊 Model Evaluation

Evaluation metrics used:
- Accuracy Score
- Precision, Recall, F1-score
- Confusion Matrix

### Cross-Validation:
- `cross_val_score` applied on training data
- Ensures performance stability and reliability

---

## 🌍 External Dataset Validation

To avoid overfitting and data leakage, the trained model is tested on a **completely unseen dataset** (`mnist_test.csv`).

Steps:
1. Load external dataset
2. Apply the same scaler and PCA
3. Predict using trained SVM
4. Evaluate accuracy and confusion matrix

This confirms **real-world generalization**, not just test-set success.

---

## 🧠 Key Learnings

- Classical ML can perform extremely well on image data
- PCA significantly boosts SVM performance
- External validation is critical for trustworthy ML
- High accuracy alone is meaningless without proper validation

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn (optional)

---

## 📁 Project Structure

Digit_Classification_MNIST.ipynb
mnist_train.csv
mnist_test.csv
README.md

---

## 🎯 Final Notes

This project showcases:
- Strong machine learning fundamentals
- Clean and structured ML workflow
- Emphasis on validation and generalization

Ideal for:
- ML interviews
- Foundational ML portfolios
- Demonstrating classical ML strength before deep learning
