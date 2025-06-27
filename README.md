# Classification-with-logistic-Regression
Build a binary classifier using logistic regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import seaborn as sns

# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1, 
                         n_informative=2, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict probabilities and classes
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Calculate ROC curve and find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Plotting
plt.figure(figsize=(12, 4))

# Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot 2: ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal threshold = {optimal_threshold:.2f}')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.tight_layout()
plt.show()

# Print evaluation metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Optimal Threshold: {optimal_threshold:.2f}")

# Sigmoid function explanation and visualization
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate data for sigmoid plot
x_sigmoid = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x_sigmoid)

plt.figure(figsize=(6, 4))
plt.plot(x_sigmoid, y_sigmoid)
plt.title('Sigmoid Function')
plt.xlabel('Input')
plt.ylabel('Sigmoid Output')
plt.grid(True)
plt.show()

print("""
Sigmoid Function Explanation:
The sigmoid function, f(z) = 1 / (1 + e^(-z)), maps any real number to the range (0,1).
In logistic regression, it converts the linear combination of features (z = wX + b) into probabilities.
- Inputs < 0 produce outputs < 0.5
- Inputs > 0 produce outputs > 0.5
- The threshold (e.g., 0.5) determines the binary classification decision.
The optimal threshold ({:.2f} in this case) balances true positives and false positives, as shown in the ROC curve.""".format(optimal_threshold))
