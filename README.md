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
    Split the train.csv to 80% train and 20% test
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
![image](https://github.com/user-attachments/assets/2a67d5c9-ba3b-410d-9c6c-5c8456d9f686)
![image](https://github.com/user-attachments/assets/6ae19a74-8038-4a3c-8b66-400fba266d0b)


## 3. Logistic Regression Implementation
The logistic regression model was implemented using scikit-learn's LogisticRegression class with hyperparameter optimization through GridSearchCV. The hyperparameter search space included regularization strengths (C) ranging from 0.001 to 100 and both L1 and L2 regularization penalties. Five-fold cross-validation was employed to ensure robust model selection.
### Model Configuration
- **Hyperparameter Search Space**:
  - Regularization strength (C): [0.0001, 0.001, 0.01]
  - Regularization type: L1 and L2 (Lasso and Ridge)
- **Optimal Parameters**: C=10 with L2 regularization (these parameters were found to be optimal for enhancing accuracy)
### Performance Metrics
based on the train and test sets
- **Accuracy**: 90.86%
- **Precision**: 92.15% (weighted average)
- **Recall**: 90.86% (weighted average)
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
    'C': [0.0001, 0.001, 0.01],  # Regularization parameter
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

## Model Architecture
A simple fully connected neural network was implemented using PyTorch, consisting of one hidden layer with 64 neurons and ReLU activation function. The architecture was designed to handle the high-dimensional input from the text vectorization while maintaining computational efficiency.

### Performance Metrics
based on the train and test sets
- **Test Accuracy**: 92.24%
- **Precision**: 92.24% (weighted average)
- **Recall**: 92.24% (weighted average)
### Network Structure
```python
class FCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.output = nn.Linear(64, output_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.output(x)  # Logits (raw scores)
        return x
```

## Implementation Details

### Data Preparation
The input data was converted from sparse matrices to PyTorch tensors and organized into batches for efficient processing:

```python
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Training Configuration
The model was trained using the following hyperparameters:
- Optimizer: Adam with learning rate 0.001
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 5

```python
# Initialize model components
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Training Process
The training loop included performance tracking for both loss and accuracy:

```python
# Training loop with metric tracking
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
```

### Initial Challenges
Initial implementation revealed several challenges:
1. Memory management: Converting sparse matrices to dense PyTorch tensors significantly increased memory usage due to the high-dimensional input space.
2. Training stability: Initial training attempts showed unstable learning patterns with fluctuating loss values.

### Optimization Steps
To address these challenges, the following optimizations were implemented:
1. For Memory Management:
- Implemented batch processing using DataLoader with batch_size=32, allowing the model to process data in smaller chunks instead of loading all data into memory at once
- Used a simplified network architecture with only one hidden layer (64 neurons) to reduce the model's memory footprint while maintaining performance


2. For Training Stability:

- Employed the Adam optimizer with a carefully tuned learning rate of 0.001, known for its adaptive learning rate properties that help maintain stable training
- Implemented CrossEntropyLoss as the loss function, which provides stable gradients for classification tasks
- Added systematic tracking of training metrics (losses and accuracies) to monitor and verify training stability across epochs

## Visualization Results
![image](https://github.com/user-attachments/assets/611565b4-a8e2-4518-bcf3-2cfd3bf3e3b5)

# LSTM Model for Fake News Detection

## Summary
This project implements an LSTM (Long Short-Term Memory) neural network for fake news detection. The model processes news article titles to classify them as legitimate or fake news. The LSTM architecture was chosen for its ability to capture sequential patterns in text data, improving upon the previous FCNN implementation.

## Model Architecture
```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output
```

## Data Preprocessing
- Input: TF-IDF vectorized news titles (9,000 features)
- Data split: 80% training, 20% testing
- Additional validation set from test.csv
- Batch size: 64

```python
def prepare_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader
```

## Training Process
- Optimizer: Adam (learning rate = 0.001)
- Loss Function: CrossEntropyLoss
- Epochs: 10
- Early stopping implemented

```python
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(1))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
```

## Model Performance
### Final Results
- Training Accuracy: 91.95%
- Validation Accuracy: 92.02%
- Test Accuracy: 91.95%

## Implementation Challenges and Solutions

### 1. Input Dimensionality
**Challenge**: LSTM required 3D input (batch_size, sequence_length, features)
**Solution**: Implemented dynamic input reshaping using unsqueeze()

### 2. Overfitting
**Challenge**: High training accuracy (99.9%) but lower validation (85.3%)
**Solutions**:
- Added dropout (0.3)
- Increased batch size to 64
- Implemented early stopping

## Visualizations
![image](https://github.com/user-attachments/assets/96384bd6-a3cd-4baa-8b36-506676a35f93)


## Comparison with all Models

### Performance Analysis
| Model               | Test Accuracy | Key Advantage                    | Trade-off                      |
|--------------------|---------------|----------------------------------|--------------------------------|
| Baseline           | 50.07%        | Simple benchmark                 | Poor performance               |
| Logistic Regression| 90.86%        | Simple, effective               | Limited feature learning       |
| FCNN              | 92.24%        | Best performance, faster training| More parameters than logistic  |
| LSTM              | 91.95%        | Captures sequential patterns     | Higher computational cost      |

### Architecture Comparison
1. **Baseline vs Advanced Models**:
   - All advanced models showed ~40% improvement
   - Demonstrates the value of machine learning approaches

2. **FCNN vs LSTM**:
   - FCNN achieved slightly better accuracy (+0.29%)
   - LSTM shows comparable performance despite different architecture
   - FCNN offers better efficiency for this task

3. **Logistic Regression vs Neural Networks**:
   - Neural networks showed modest improvements
   - Complexity increase yielded minor accuracy gains

## Key Findings
 **Model Performance**:
   - FCNN achieved the highest test accuracy (92.24%)
   - LSTM performed comparably (91.95%)
   - Logistic regression was competitive (90.86%)
   - All significantly outperformed baseline (50.07%)



## Conclusions
The project demonstrates that while both simple and complex models can effectively classify fake news titles, the FCNN achieved the best performance with a test accuracy of 92.24%. The LSTM model, despite its more sophisticated architecture, performed slightly lower at 91.95%. This suggests that for this specific task:

1. The distinguishing features between fake and legitimate news titles may be effectively captured by simpler neural architectures
2. Sequential patterns captured by LSTM may not provide significant advantages for this classification task
3. The additional computational cost of LSTM might not be justified given the minimal performance difference

