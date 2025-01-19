import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F


def analyze_lstm_performance(model, train_metrics, X_test_tensor, y_test_tensor, X_val_tensor, y_val_tensor):
    """
    Visualize LSTM model performance from collected metrics

    Parameters:
    - model: trained LSTM model
    - train_metrics: dictionary containing:
        - 'epoch_losses': list of training losses
        - 'train_accuracies': list of training accuracies
        - 'val_accuracies': list of validation accuracies
    - X_test_tensor: test data
    - y_test_tensor: test labels
    - X_val_tensor: validation data
    - y_val_tensor: validation labels
    """
    plt.figure(figsize=(15, 10))

    # 1. Training Metrics Plot
    plt.subplot(221)
    plt.plot(train_metrics['epoch_losses'], label='Training Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy Plot
    plt.subplot(222)
    plt.plot(train_metrics['train_accuracies'], label='Training Accuracy', marker='o', color='blue')
    plt.plot(train_metrics['val_accuracies'], label='Validation Accuracy', marker='o', color='red')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 3. Confusion Matrix
    plt.subplot(223)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs, 1)
        cm = confusion_matrix(y_test_tensor.numpy(), test_predicted.numpy())

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.colorbar()

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    classes = ['Real', 'Fake']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)

    # 4. Prediction Confidence Distribution
    plt.subplot(224)
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = F.softmax(test_outputs, dim=1)
        confidence, predicted = torch.max(test_probs, 1)

        # Separate confidences for correct and incorrect predictions
        correct_mask = predicted == y_test_tensor
        correct_conf = confidence[correct_mask].numpy()
        incorrect_conf = confidence[~correct_mask].numpy()

    plt.hist(correct_conf, alpha=0.5, label='Correct Predictions',
             bins=20, color='green')
    plt.hist(incorrect_conf, alpha=0.5, label='Incorrect Predictions',
             bins=20, color='red')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print Performance Summary
    print("\n=== Model Performance Summary ===")
    print(f"\nConfusion Matrix:")
    print("                  Predicted")
    print("                  Real    Fake")
    print(f"Actual Real:     {cm[0][0]}     {cm[0][1]}")
    print(f"      Fake:     {cm[1][0]}     {cm[1][1]}")

    # Calculate final metrics
    test_acc = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Average Prediction Confidence: {confidence.mean():.4f}")