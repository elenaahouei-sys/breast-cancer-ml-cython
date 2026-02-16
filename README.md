# breast-cancer-ml-cython
High-performanced breast cancer classification using custom Cython KNN and advanced ML models
Breast Cancer Classification with Custom Cython-Optimized KNN & Advanced ML Models 🧬⚡

🚀 Project Overview

This project demonstrates a high-performance breast cancer classification system that I developed from scratch. It combines both classical and advanced machine learning models and showcases optimization, analysis, and visualization skills.

Key highlights:

⚡ Custom KNN implemented and optimized with Cython: I manually implemented KNN and compiled it with Cython for dramatically faster distance computations compared to pure Python.

📊 Full model comparison: Classical models including Naive Bayes, Decision Tree, Random Forest, SVM, Logistic Regression, and MLP Neural Networks.

🧪 Comprehensive evaluation metrics: Accuracy, Precision, Recall, F1 Score, ROC Curve, and Confusion Matrix.

🔬 MLP experiments: Investigated the effect of training dataset size, number of epochs, and learning rates on MLP performance.

🧩 Data preprocessing & analysis: Standard scaling, PCA visualization, and feature correlation heatmaps.

🗂️ Dataset

I used the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn:

569 samples with 30 numerical features

Binary classification: Malignant (1) vs Benign (0)

Clean dataset with no missing values

from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X, y = bc.data, bc.target

⚡ My Custom Cython KNN
Why Cython?

Standard Python KNN implementations can be slow due to nested loops. I optimized the KNN algorithm using Cython, compiling Python code into C to achieve orders-of-magnitude speedups while preserving accuracy.

Example: Euclidean Distance in Cython
%%cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

ctypedef cnp.float64_t DTYPE_t

def euclidean_dist_cython(cnp.ndarray[DTYPE_t, ndim=2] X_train,
                          cnp.ndarray[DTYPE_t, ndim=1] x_test):
    cdef int n_samples = X_train.shape[0]
    cdef int n_features = X_train.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=1] distances = np.empty(n_samples, dtype=np.float64)
    cdef int i, j
    cdef DTYPE_t diff, total

    for i in range(n_samples):
        total = 0.0
        for j in range(n_features):
            diff = X_train[i, j] - x_test[j]
            total += diff * diff
        distances[i] = sqrt(total)

    return distances


✅ Result: My Cython KNN is 20x faster than pure Python and matches scikit-learn in accuracy.

🧠 Models Implemented
Model	Description
Gaussian Naive Bayes	Simple probabilistic baseline
KNN (Cython, scikit-learn, pure Python)	Custom KNN vs library comparison
Decision Tree	Tuned max depth and splits
Random Forest	1000 trees with optimized depth
SVM	Polynomial kernel with probability estimates
Logistic Regression	GridSearchCV for best regularization parameter
MLP Neural Network	Hidden layer size 256, batch size 64, experimented with epochs & learning rates
📊 Metrics & Visualization

Accuracy, Precision, Recall, F1 Score

Confusion Matrix

ROC Curve and AUC

Train vs Test Accuracy comparison

Effect of dataset size on MLP performance

Feature correlation heatmaps

Example:

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

⏱️ Speed Comparison (KNN)
| Implementation      | ⏱ Time (s) | ✅ Accuracy |
|--------------------|------------|------------|
| scikit-learn KNN    | 0.0846     | 0.9561     |
| Cython KNN          | 0.0205     | 0.9561     |
| Pure Python KNN     | 0.0202     | 0.9561     |

Cython provides a ~20x speedup over pure Python while maintaining identical predictions.

🧪 MLP Experiments

I explored MLP performance across varying training sizes, epochs, and learning rates:

Training fractions: 10%, 30%, 50%, 70%, 90%, 100%

Epochs: 50, 100, 200

Learning rates: 0.001, 0.01, 0.1

The experiments helped me understand overfitting, underfitting, and dataset sufficiency in neural networks.
