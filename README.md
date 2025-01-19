# Fake News Detection Using Deep Learning: A Comparative Analysis of Classification Models
## 1. Project Overview and Dataset Description
This project addresses the challenge of fake news detection through automated classification of news titles. The goal is to develop and compare various machine learning models that can effectively distinguish between legitimate and fake news headlines.
### Dataset Characteristics
- **Features**: News article titles processed using TF-IDF vectorization
- **Feature Dimensionality**: 9,000 numerical features after TF-IDF transformation
- **Label**: Binary classification (0 for legitimate news, 1 for fake news)
   
- **Data preprossesing**:
  - Null value handling through empty string replacement
  - Removes non-alphabetic characters
  - Converts text to lowercase
  - Tokanization text to words
  - Text normalization using Porter Stemming
  - Remove English stopwords (common words like "the", "is", "at")
  - TF-IDF vectorization with a maximum of 9,000 features
  - **Data Split**:
    - Split the train.csv to 80% train and 20% test
    - define validation from test.csv
## 2. Baseline Model Implementation
A baseline model was established using scikit-learn's DummyClassifier with the 'most_frequent' strategy to provide a minimum performance benchmark.
### Baseline Results
based on the train and test sets
- **Test Accuracy**: 50.07%  
- **Precision**: 25% (weighted average)
- **Recall**: 50% (weighted average)
- **Class Distribution Analysis**: The near-50% accuracy indicates a relatively balanced dataset

The baseline model's performance highlights the need for better approaches, as its accuracy is only slightly higher than random guessing. This shows the importance of using smarter models to make better predictions.
# Baseline Model Implementation and Analysis

### Implementation Details
```python
# Create and train baseline model
baseline_model = DummyClassifier(strategy='most_frequent')
baseline_model.fit(X_train, y_train)

# Generate predictions
y_pred = baseline_model.predict(X_test)
```

### Visualization
![image](https://github.com/user-attachments/assets/ba2a06a0-fe49-4663-903c-2817f7ccd9d3)
![image](https://github.com/user-attachments/assets/721281b2-4eba-45ca-9c8c-54abda0d57ec)


## 3. Logistic Regression Implementation
The logistic regression model was implemented using scikit-learn's LogisticRegression class with hyperparameter optimization through GridSearchCV. The hyperparameter search space included regularization strengths (C) ranging from 0.001 to 100 and both L1 and L2 regularization penalties. Five-fold cross-validation was employed to ensure robust model selection.
### Model Configuration
- **Hyperparameter Search Space**:
  - Regularization strength (C): [0.001, 0.01, 0.1, 1, 10, 100]
  - Regularization type: L1 and L2 (Lasso and Ridge)
- **Optimal Parameters**: C=10 with L2 regularization (these parameters were found to be optimal for enhancing accuracy)
### Performance Metrics
based on the train and test sets
- **Test Accuracy**: 93.39%
- **Precision**: 0.94 (weighted average)
- **Recall**: 0.93 (weighted average)
  # Logistic Regression Implementation and Analysis

## Implementation Details
The preprocessing pipeline included:
1. Text vectorization using the previously established vectorizer
2. Train-test split maintained from the baseline model
3. Grid search cross-validation for hyperparameter tuning
4. Model training with optimized parameters
5. Performance evaluation on the test set

### Performance Analysis
![image](https://github.com/user-attachments/assets/6f47f7b2-7dd9-44ef-82d3-96991ba28d40)

# Logistic Regression Implementation and Analysis

### Implementation Code
```python
# Find optimal hyperparameters using grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2']  # Regularization type
}
# Create a Grid Search with cross-validation
grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), 
                         param_grid, cv=5, scoring='accuracy')

# Fit the Grid Search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
```

### Model Training Code
```python
# Train a Logistic Regression model with the best hyperparameters
best_logistic_regression = LogisticRegression(solver='liblinear', **best_params)
best_logistic_regression.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_logistic_regression.predict(X_test)
```

## Model Optimization Process
### Initial Challenges
Initial implementation revealed several challenges, one of them was:
- High-dimensional feature space: The text vectorization process created a large number of features, which initially led to computational inefficiency and potential overfitting.
### Optimization Step
To address these challenge, the following optimizations were implemented
- Comprehensive hyperparameter tuning using GridSearchCV with cross-validation to find the optimal balance between model complexity and performance. The grid search explored different regularization strengths (C values from 0.001 to 100) to prevent overfitting.

## 4. Basic Neural Network Implementation
A fully connected neural network (FCNN) was implemented using PyTorch with a simple architecture.
### Architecture
```
Input Layer (9000 features)
    ↓
Hidden Layer (64 units, ReLU activation)
    ↓
Output Layer (2 units)
```
### Training Configuration
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 5
### Performance Analysis
- **Training Accuracy Progression**:
  - Epoch 1: 87.11%
  - Epoch 5: 99.16%
- **Final Test Accuracy**: 92.67%
### Model Behavior
- Rapid convergence observed
- High training accuracy suggests potential overfitting
- Comparable test performance to logistic regression
## 5. Advanced Neural Network Implementation (LSTM)
An LSTM-based architecture was implemented to capture sequential patterns in the text data.
### Architecture
```
Input Layer
    ↓
LSTM Layer (hidden_size=128)
    ↓
Dropout Layer (p=0.3)
    ↓
Linear Layer (Output)
```
### Training Configuration
- **Batch Size**: 64
- **Optimizer**: Adam (learning rate = 0.001)
- **Dropout Rate**: 0.3
- **Epochs**: 10
### Performance Metrics
- **Final Training Accuracy**: 99.78%
- **Validation Accuracy**: 91.92%
- **Test Accuracy**: 92.31%
- **Class-wise Performance**:
  - Class 0: Precision = 0.92, Recall = 0.92, F1 = 0.92
  - Class 1: Precision = 0.92, Recall = 0.92, F1 = 0.92
### Model Evolution
- Training accuracy improved consistently over epochs
- Dropout helped maintain generalization
- Final performance comparable to simpler models
## Comparative Analysis
### Performance Comparison
1. Baseline: 50.07% (Test Accuracy)
2. Logistic Regression: 93.39% (Test Accuracy)
3. FCNN: 92.67% (Test Accuracy)
4. LSTM: 92.31% (Test Accuracy)
### Key Findings
- All advanced models significantly outperformed the baseline
- Logistic regression achieved the highest test accuracy
- Neural networks showed similar performance despite architectural differences
- Complexity increase did not yield proportional performance gains
### Model Trade-offs
- **Logistic Regression**: Best test performance, simpler architecture
- **FCNN**: Good performance, faster training
- **LSTM**: Similar performance, higher computational cost
## Conclusions
The project demonstrates that while deep learning models can effectively classify fake news titles, simpler models like logistic regression can achieve comparable or better performance for this specific task. The similar performance across different architectures suggests that the distinguishing features between fake and legitimate news titles may be effectively captured by linear relationships in the TF-IDF space.
Future work could explore:
- Feature engineering improvements
- Ensemble methods
- More sophisticated text preprocessing
- Attention mechanisms
- Transfer learning with pre-trained language models
