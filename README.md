
# Logistic Regression from Scratch (NumPy)

I built logistic regression from scratch using only NumPy and used it to tell whether a breast tumor is **malignant** (cancer) or **benign** (safe), based on 30 measurements from a biopsy.

## What is logistic regression?

It is a model that gives a probability between 0 and 1. If the probability is above a threshold (0.5 by default), it predicts class 1, otherwise class 0. It learns by gradient descent: it makes a prediction, checks how wrong it is, and slowly fixes its weights.

## Dataset

- **Wisconsin Breast Cancer dataset** from scikit-learn.
- 569 tumors, 30 features each (radius, texture, perimeter, area, concavity and more).
- 212 malignant (label 0) and 357 benign (label 1).
- Split: 455 for training, 114 for testing (80/20).

## What I did

1. Loaded the data and made a scatter plot of two features (mean radius and mean texture).
2. Split the data and standardized the features. The scaler was fitted on the training set only, so no test information leaked.
3. Wrote my own `LogisticRegression` class with these methods: sigmoid, cost (binary cross-entropy), gradient, fit, predict_proba and predict.
4. Trained it with gradient descent (learning rate 0.1, 1000 iterations).
5. Plotted the loss curve.
6. Evaluated it on the test set.
7. Trained a second model on 2 features and plotted its decision boundary.
8. Tried different thresholds to see the effect on recall and precision.
9. Compared my model with scikit-learn's LogisticRegression.

## Results

| Metric | Value |
|---|---|
| Test accuracy | 97.4% |
| Recall (malignant) | 97.6% |
| Precision (malignant) | 95.4% |
| Final training loss | 0.057 (started at 0.693) |

**Confusion matrix** (rows = true, columns = predicted):

| | Predicted malignant | Predicted benign |
|---|---|---|
| **Actually malignant** | 41 | 1 |
| **Actually benign** | 2 | 70 |

Only 1 cancer case was missed out of 42.

**Compared with scikit-learn:** my model got 97.4% and scikit-learn got 98.2%, a difference of less than 1%.

## Threshold experiment

In medicine, missing a cancer is worse than a false alarm. So I checked how the threshold changes the results (for the malignant class):

| Threshold | Recall | Precision |
|---|---|---|
| 0.3 | 0.9524 | 1.0000 |
| 0.4 | 0.9524 | 0.9756 |
| 0.5 | 0.9762 | 0.9535 |
| 0.6 | 0.9762 | 0.8913 |
| 0.7 | 0.9762 | 0.8913 |

A higher threshold on the benign probability catches more cancers, but it also gives more false alarms.

## Tech Stack

Python, NumPy, Matplotlib, scikit-learn (only for loading data, splitting, scaling and metrics)

## How to Run

1. Open `logistic_regression_from_scratch.ipynb` in Google Colab.
2. Click **Runtime, Run all**.
3. The dataset loads automatically from scikit-learn.

## What I Learned

- How sigmoid, cross-entropy loss and gradient descent work together.
- Why features need to be standardized, and why the scaler must be fitted on training data only.
- Why recall matters more than accuracy in a medical problem.

## Author

**Areeha Shafique**
GitHub: [Areeha548](https://github.com/Areeha548)
