import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def analyze_baseline_model(X_train, X_test, y_train, y_test, y_pred):
    # 1. Data Distribution Analysis
    plt.figure(figsize=(10, 5))

    # Training Data Distribution
    plt.subplot(121)
    train_dist = Counter(y_train)
    plt.bar(['Real (0)', 'Fake (1)'], [train_dist[0], train_dist[1]])
    plt.title('Training Data Distribution')
    plt.ylabel('Number of Articles')

    # Test Data Distribution
    plt.subplot(122)
    test_dist = Counter(y_test)
    plt.bar(['Real (0)', 'Fake (1)'], [test_dist[0], test_dist[1]])
    plt.title('Test Data Distribution')

    plt.tight_layout()
    plt.show()

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 3. Print detailed statistics
    print("\n=== Dataset Statistics ===")
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")

    print("\n=== Class Distribution ===")
    print("Training set:")
    print(f"Real news (0): {train_dist[0]} ({train_dist[0] / len(y_train) * 100:.2f}%)")
    print(f"Fake news (1): {train_dist[1]} ({train_dist[1] / len(y_train) * 100:.2f}%)")

    print("\n=== Model Performance Metrics ===")
    print("Test Set Performance:")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

    # 4. Performance Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()