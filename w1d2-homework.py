from sklearn.datasets import load_digits



def choose_paradigm(has_labels: bool, goal: str) -> str:
    if has_labels :
        if goal == "predict_category" :
            return "Classification"
        elif goal == "predict_value" :
            return "Regression" 
        else :
            return "unknown goal"   
    else :
        if goal == "discover_groups" :
            return "Clustering"
        elif goal == "compress_data" :
            return "Dimensionality Reduction"
        else :
            return "unknown goal"

"""
result = choose_paradigm(True, "predict_category")
print(f"Result: {result}")

result = choose_paradigm(True, "predict_value")
print(f"Result: {result}")

result = choose_paradigm(False, "discover_groups")
print(f"Result: {result}")

result = choose_paradigm(False, "compress_data")
print(f"Result: {result}")
"""

params = [
    (True, "predict_category"),
    (True, "predict_number"),
    (False, "discover_groups"),
    (False, "compress_data")
]

print("1. === Paradigm Decision ===")
for has_labels, goal in params:
    result = choose_paradigm(has_labels, goal)
    print(f"has_labels: {has_labels}, goal: '{goal}' => Result: {result}")

print("2. === Supervised Classification Task ===")

print("# 1. Load the digits dataset")
digits = load_digits()
print(f"Data shape: {digits.data.shape}")
print(f"Target shape: {digits.target.shape}") # it means that we have 1797 samples and each label has 64 features (8x8 pixel values) and the target is the digit label (0-9) for each sample.

# Structure of the dataset
X = digits.data # Features (pixel values)
y = digits.target # Target labels (digits 0-9)

print("# 2. Split the dataset into training and testing sets")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, 
                                    test_size=0.2, # 20% of the data for testing
                                    random_state=42 # Fix randomness for reproducibility
)

print("# 3. Scale the features")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("# 4. Train a simple model (e.g., Logistic Regression)")
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("# 5. Evaluate the model")
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("# 6. Visualize some predictions")
import numpy as np

report = classification_report(y_test, y_pred, output_dict=True)
precisions = {k: v["precision"] for k, v in report.items() if k.isdigit()}
lowest_class = min(precisions, key=precisions.get)
print("Lowest precision class:", lowest_class)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("# Digit 9 has the lowest precision because it is sometimes misclassified as digits like 4 and 5.")

print("3. === Unsupervised Clustering Task ===")

print("# 1. Use same dataset but DROP labels")

from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data # Use only features (pixel values)
print("*** Dataset loaded without labels completed ***")

print("# 2. Apply K-Means clustering")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)
print("*** K-Means model training completed ***")

print("# 3. Get cluster labels")
cluster_labels = kmeans.labels_
print(cluster_labels)
print("*** Cluster labels obtained ***")

print("# 4. Cluster size distribution")

import numpy as np

for i in range(10):
    cluster_size = np.sum(cluster_labels == i)
    print(f"Cluster {i}: {cluster_size} samples")
print("*** Cluster size distribution calculated ***")

print("# 5. Find largest cluster")

largest_cluster = np.argmax([np.sum(cluster_labels == i) for i in range(10)])
print(f"Largest cluster: Cluster {largest_cluster}")
print("*** Largest cluster identified ***")

print("# 5. Visualize 5 random samples from the largest cluster")

import matplotlib.pyplot as plt

# Get indices of largest cluster
indices = np.where(cluster_labels == largest_cluster)[0]

# Pick first 5 samples
samples = X[indices[:5]]

for i, sample in enumerate(samples):
    plt.subplot(1, 5, i+1)
    plt.imshow(sample.reshape(8, 8), cmap='gray')
    plt.axis('off')

plt.show()

print("4. === Comparison Analysis (Comments) ===")
print("KMean clusters do not perfectly match the digit classes because it is an unsupervised method.")
print("It groups data based on similarity rather than true labels.")

print("Supervised classification uses true labels to learn patterns, while unsupervised clustering finds inherent groupings without labels.")
print("It can predict exact classes and provided clear accuracy metrics.")

print("Clustering do not required labels and can discover hidden patterns data.")
print("It can be useful for exploratory data analysis but may not align with true classes.")