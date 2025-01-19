import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd


def analyze_logistic_regression(grid_search, X_train, y_train, y_test, y_pred, best_logistic_regression, vectorizer):
    # Set the figure style
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['axes.grid'] = True

    # 1. Cross-validation results for different hyperparameters
    plt.figure(figsize=(15, 5))

    # Create subplot for hyperparameter analysis
    plt.subplot(131)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    c_values = sorted(set(cv_results['param_C']))
    penalties = sorted(set(cv_results['param_penalty']))

    # Create matrix of scores
    scores_matrix = np.zeros((len(c_values), len(penalties)))
    for i, c in enumerate(c_values):
        for j, penalty in enumerate(penalties):
            mask = (cv_results['param_C'] == c) & (cv_results['param_penalty'] == penalty)
            scores_matrix[i, j] = cv_results[mask]['mean_test_score'].values[0]

    plt.imshow(scores_matrix, aspect='auto')
    plt.colorbar(label='CV Score')

    # Add text annotations
    for i in range(len(c_values)):
        for j in range(len(penalties)):
            plt.text(j, i, f'{scores_matrix[i, j]:.3f}',
                     ha='center', va='center', color='white')

    plt.xticks(range(len(penalties)), penalties)
    plt.yticks(range(len(c_values)), c_values)
    plt.xlabel('Penalty')
    plt.ylabel('C value')
    plt.title('Cross-validation Scores')

    # 2. Confusion Matrix
    plt.subplot(132)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm)
    plt.colorbar(label='Count')

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center', color='white')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 3. Feature Importance
    plt.subplot(133)
    feature_importance = np.abs(best_logistic_regression.coef_[0])
    top_features_idx = np.argsort(feature_importance)[-10:]
    top_features = vectorizer.get_feature_names_out()[top_features_idx]
    top_importance = feature_importance[top_features_idx]

    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Top 10 Most Important Features')

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("\n=== Model Performance Summary ===")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Score: {grid_search.best_score_:.3f}")
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")