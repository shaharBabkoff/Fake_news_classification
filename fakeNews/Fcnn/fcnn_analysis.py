import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn.functional as F


def analyze_fcnn_model(model, train_metrics, X_test_tensor, y_test_tensor):
    """
    Visualize FCNN model performance and training metrics
    
    Parameters:
    - model: trained FCNN model
    - train_metrics: dict containing 'losses' and 'accuracies' lists from training
    - X_test_tensor: test data tensor
    - y_test_tensor: test labels tensor
    """
    plt.style.use('default')
    
    # 1. Training Progress Plot
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(131)
    plt.plot(train_metrics['losses'], label='Train Loss', marker='o')
    plt.plot(train_metrics['val_losses'], label='Validation Loss', marker='o')

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot training accuracy
    plt.subplot(132)
    plt.plot(train_metrics['accuracies'], marker='o', color='green')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # 3. Confusion Matrix
    plt.subplot(133)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        cm = confusion_matrix(y_test_tensor.numpy(), predicted.numpy())
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    classes = ['Real', 'Fake']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    
    plt.tight_layout()
    plt.show()

    # 4. Prediction Confidence Distribution
    plt.figure(figsize=(10, 5))
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    # Separate confidences for correct and incorrect predictions
    correct_mask = predicted == y_test_tensor
    correct_conf = confidence[correct_mask].numpy()
    incorrect_conf = confidence[~correct_mask].numpy()
    
    plt.hist(correct_conf, alpha=0.5, label='Correct Predictions', 
             bins=20, color='green')
    plt.hist(incorrect_conf, alpha=0.5, label='Incorrect Predictions', 
             bins=20, color='red')
    plt.title('Model Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. Print Performance Metrics
    print("\n=== Model Performance Summary ===")
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  Real    Fake")
    print(f"Actual Real:     {cm[0][0]}     {cm[0][1]}")
    print(f"      Fake:     {cm[1][0]}     {cm[1][1]}")
    
    # Calculate and print classification report
    report = classification_report(
        y_test_tensor.numpy(),
        predicted.numpy(),
        target_names=['Real', 'Fake']
    )
    print("\nClassification Report:")
    print(report)

    # Print final metrics
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {confidence.mean():.4f}")
