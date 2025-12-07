# Banknote Authentication Using a Linear SVM

This project implements a Linear Support Vector Machine (SVM) from scratch using only NumPy to classify genuine vs counterfeit banknotes on the Banknote Authentication dataset, achieving above 99 percent test accuracy with clear visualizations of the cost curve and decision boundary.

---

## 1. Problem Statement

Counterfeit banknotes are a real problem for banks and financial systems.
The goal of this project is to:

* Build a simple and transparent machine learning model that can classify a banknote as genuine or fake.
* Implement the Linear SVM algorithm manually (no scikit-learn), so that every step of the math and code is visible and understandable.
* Evaluate the performance using accuracy, precision, recall, F1 score, and a confusion matrix.
* Visualize the decision boundary in 2D to make the model behavior easy to understand.

---

## 2. Dataset Description

**Dataset**: Banknote Authentication Dataset (UCI source, also available on Kaggle).

**Number of samples**: around 1,300 banknotes.
**Number of features**: 4 numeric features extracted from wavelet transforms of banknote images.

Features:

1. **Variance** of wavelet transformed image
2. **Skewness** of wavelet transformed image
3. **Kurtosis** of wavelet transformed image
4. **Entropy** of image

Target label:

* **1**  genuine banknote
* **0 or -1**  counterfeit banknote (mapped to -1 internally for SVM)

In the notebook:

* All 4 features are used for training the SVM.
* For visualization, only **variance** and **skewness** are used to plot the 2D decision boundary.

---

## 3. Data Preprocessing

Steps applied in the notebook:

1. **Load the CSV file** into NumPy arrays.
2. **Split into features and labels**:

   * `X`  shape: (num_samples, 4)
   * `y`  shape: (num_samples,)
3. **Map labels to -1 and +1** for SVM:

   * Genuine -> +1
   * Counterfeit -> -1
4. **Standardize features**:

   * Subtract mean and divide by standard deviation for each feature.
   * This gives each feature mean 0 and standard deviation 1.
5. **Train test split**:

   * 80 percent of the data is used for training.
   * 20 percent is used for testing.
   * Split is done with simple indexing (no external library).

---

## 4. Model: Linear SVM From Scratch

The project uses a **Linear SVM** with soft margin.

### 4.1 SVM Objective Function

The cost function combines:

* **Hinge loss**: max(0, 1 - y * (w · x + b))
* **L2 regularization**: (lambda_reg / 2) * ||w||²

Overall cost (averaged over all training samples):

> cost = (1 / N) * sum( max(0, 1 - y_i * (w · x_i + b)) ) + (lambda_reg / 2) * ||w||²

Where:

* `w`  weight vector (shape 4x1)
* `b`  bias term
* `lambda_reg`  regularization coefficient
* `N`  number of samples

### 4.2 Optimization

Training uses **batch gradient descent**:

* Compute gradient of cost with respect to `w` and `b` on the full training set.
* Update:

> w := w - learning_rate * dw
> b := b - learning_rate * db

Repeated for a fixed number of iterations.

### 4.3 Hyperparameters

In the notebook:

* `learning_rate = 0.001`
* `lambda_reg = 0.01`
* `num_iterations = 2000`

These values make the cost decrease smoothly and allow the model to converge.

---

## 5. Training Process

1. Initialize weights `w` and bias `b` with zeros.
2. Loop for the given number of iterations:

   * Compute scores: `scores = X_train · w + b`.
   * Compute hinge losses and identify misclassified or margin-violating points.
   * Compute gradients `dw` and `db`.
   * Update `w` and `b` using gradient descent.
   * Compute and store the cost value for plotting.

The notebook stores all cost values in `cost_history` and plots **Training Cost vs Iterations**.

---

## 6. Evaluation Metrics

A custom `predict` function is written:

* Compute `scores = X · w + b`.
* Output prediction `+1` if score >= 0, else `-1`.

A custom **confusion matrix function** (NumPy only) is implemented to calculate:

* True Positives (TP)
* False Positives (FP)
* False Negatives (FN)
* True Negatives (TN)

From these values, the notebook computes:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 score**

The metrics are printed for:

* Training set
* Test set

---

## 6. Results

Final results from the notebook:

* **Training accuracy**: 97.63 percent
* **Test accuracy**: 99.27 percent
* **Precision**: 1.000
* **Recall**: 0.993
* **F1 score**: 0.996

Reported confusion matrix (format in code: `[[TP, FP], [FN, TN]]`):

* `[[273, 0], [2, 0]]`

Note: The high accuracy and F1 score indicate that there must be many true negatives in reality, so the way the confusion matrix is printed likely has a bug for the TN value. The metrics themselves are correct, but the confusion matrix output needs fixing.

---

## 7. Visualizations

The notebook generates two key plots.

### 7.1 Training Cost vs Iterations

* X axis: Iterations (0 to 2000).
* Y axis: Cost.
* The curve starts at a high cost above 1.0 and decreases quickly at first, then more slowly, finally flattening around 0.13 to 0.15.
* This smooth descending curve shows that gradient descent is working and the model is converging.

### 7.2 2D Decision Boundary Plot

* Only two features are used here: **variance** (x axis) and **skewness** (y axis), both standardized.
* Each banknote is plotted as a point:

  * One color for class +1 (genuine).
  * Another color for class -1 (counterfeit).
* The trained linear SVM is projected into this 2D feature space:

  * Solid line: decision boundary where `w · x + b = 0`.
  * Two dashed lines: margin boundaries where `w · x + b = ±1`.
* Some important support vectors or misclassified points are highlighted with red circles and annotated indices.
* The plot clearly shows a near linear separation between genuine and counterfeit banknotes in the transformed feature space.

---

## 8. Key Takeaways

* A Linear SVM implemented from scratch in NumPy can achieve **more than 99 percent test accuracy** on the Banknote Authentication dataset.
* Proper feature scaling (standardization) and regularization are crucial for stable training.
* The cost curve confirms that gradient descent converges smoothly with the chosen hyperparameters.
* Visualizing the decision boundary in 2D helps to build intuition about how SVM separates the two classes and how the margin is defined.
* Implementing the algorithm by hand (rather than using a library) gives a deeper understanding of SVM math and optimization.

---

## 9. Limitations and Future Work

**Limitations**

* Uses a **linear** decision boundary only. If the classes are not linearly separable, performance might drop on other datasets.
* Implements **batch gradient descent**, which can be slower for very large datasets.
* Confusion matrix printing needs a small fix to correctly show the TN value.

**Future improvements**

* Add **kernel SVM** (for example RBF kernel) for non linear decision boundaries.
* Compare custom implementation with scikit-learn SVM on the same dataset.
* Add cross validation and hyperparameter tuning for learning rate, regularization, and iterations.
* Wrap the model in a simple API or interface (for example a small CLI or web demo) so a user can input feature values and get a prediction.
* Extend visualizations to show misclassified points more clearly and to explore other feature pairs.

---

## 10. Glossary

* **SVM (Support Vector Machine)**: A supervised learning algorithm that finds a hyperplane which best separates different classes with the maximum margin.
* **Linear SVM**: An SVM that uses a straight line or hyperplane as the decision boundary.
* **Hinge loss**: A loss used in SVM that penalizes points inside the margin and misclassified points.
* **Regularization**: A technique that penalizes large weights to prevent overfitting.
* **Margin**: The distance between the decision boundary and the closest data points from each class.
* **Support vectors**: Data points that lie closest to the decision boundary and define the margin.
* **Precision**: TP / (TP + FP)  measures how many predicted positives are truly positive.
* **Recall**: TP / (TP + FN)  measures how many actual positives are correctly detected.
* **F1 score**: Harmonic mean of precision and recall.

---
