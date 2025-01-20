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
- **Optimal Parameters**: C=0.01 with L1 regularization (these parameters were found to be optimal for enhancing accuracy)
### Performance Metrics

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
- Comprehensive hyperparameter tuning using GridSearchCV with cross-validation to find the optimal balance between model complexity and performance. The grid search explored different regularization strengths to prevent overfitting.


## 4. Basic Neural Network Implementation

## Model Architecture
A simple fully connected neural network was implemented using PyTorch, consisting of one hidden layer with 64 neurons and ReLU activation function. The architecture was designed to handle the high-dimensional input from the text vectorization while maintaining computational efficiency.

### Performance Metrics

- **Test Accuracy**: 91.97%
- **Precision**: 92.01% (weighted average)
- **Recall**: 91.07% (weighted average)
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
![Fcnn_Graphs 1](https://github.com/user-attachments/assets/865106d0-9812-4460-877e-63892b995b7e)


# 5. LSTM Model for Fake News Detection

## Summary
This project implements an LSTM-based neural network for detecting fake news. The model analyzes news article titles to classify them as legitimate or fake. The architecture leverages embedding layers, multi-layer LSTMs, and dropout regularization to effectively handle sequential text data and mitigate overfitting.

## Key Updates
- *Vocabulary Size*: Limited to 40,000 tokens for efficient embedding.
- *Preprocessing*: Texts are cleaned (stopwords removed, lowercased), tokenized, and padded to a fixed sequence length of 18.
- *Dynamic Class Balancing*: Added class weights to the loss function to address label imbalances.
- *Scheduler*: Learning rate adjusted dynamically using ReduceLROnPlateau to prevent plateauing.

## Model Architecture
The model includes:
1. *Embedding Layer*: Converts token IDs to dense vector representations.
2. *LSTM Layers*: Two stacked LSTM layers with hidden dimension 64 to capture temporal dependencies.
3. *Dropout*: Applied both within the LSTM layers and before the fully connected layer.
4. *Fully Connected Layer*: Maps LSTM outputs to binary logits.

### Performance Metrics

- **Accuracy**: 92.21%
- **Precision**: 92.28% (weighted average)
- **Recall**: 92.21% (weighted average)




### Network Structure


```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, seq_len):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(seq_len * hidden_dim, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
```
### Training Configuration

- *Optimizer*: Adam with weight decay.(lr=0.00025)
- *Loss Function*: CrossEntropyLoss with dynamic class weights.
- *Batch Size*: 32.
- *Epochs*: 10.
- *Device*: CPU (can be adapted for GPU).

```python
criterion = CrossEntropyLoss(weight=class_weights, reduction="mean")
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
```
## Training Process

```python
for epoch in range(max_epochs):
    train_auc, train_accuracy, train_loss = trainer.train(
        current_epoch_nr=epoch)
    train_aucs.append(train_auc)
    train_accs.append(train_accuracy)
    train_losses.append(train_loss)

    val_auc, val_accuracy, val_loss = trainer.evaluate(
        current_epoch_nr=epoch, scheduler=scheduler)
    val_aucs.append(val_auc)
    val_accs.append(val_accuracy)
    val_losses.append(val_loss)
```

## Visualizations
Training progress visualized using:
1. AUC per epoch.
2. Accuracy trends for training and validation.
3. Loss curves for both datasets.

![Advanced_Graphs 1](https://github.com/user-attachments/assets/37a6d45b-c4e2-46cd-98fd-16a7f380249e)
## Results
### Final Test Performance
- *Accuracy*: Achieved using classification_report.
- *Precision*: Weighted for imbalanced datasets.
- *Recall*: Weighted metrics ensure robust evaluation.




## Comparison with all Models

### Performance Analysis
| Model               | Test Accuracy | Key Advantage                    | Trade-off                      |
|--------------------|---------------|----------------------------------|--------------------------------|
| Baseline           | 50.07%        | Simple benchmark                 | Poor performance               |
| Logistic Regression| 90.86%        | Simple, effective               | Limited feature learning       |
| FCNN              | 91.97%        | Best performance, faster training| More parameters than logistic  |
| LSTM              |  92.21%        | Captures sequential patterns     | Higher computational cost      |

### Architecture Comparison
## Baseline vs Advanced Models

- All advanced models showed ~40% improvement.
- Demonstrates the value of machine learning approaches.

## FCNN vs LSTM

- LSTM achieved better accuracy (+0.24%).
- LSTM offers better sequential pattern capture, though at a higher computational cost.

# Key Findings

## Model Performance

- **LSTM** achieved the highest test accuracy (92.21%).
- **FCNN** performed comparably (91.97%).
- **Logistic Regression** was competitive (90.86%).
- All significantly outperformed baseline (50.07%).

# Conclusions

The project demonstrates that while both simple and complex models can effectively classify fake news titles, the **LSTM** achieved the best performance with a test accuracy of 92.21%. The **FCNN** model, despite its simpler architecture, performed slightly lower at 91.97%. This suggests that for this specific task:

1. The distinguishing features between fake and legitimate news titles may benefit from sequential pattern analysis provided by **LSTM**.
2. Simpler architectures like **FCNN** are competitive and may be preferred in resource-constrained scenarios.
3. **Logistic Regression** remains a strong baseline for tasks where simplicity is critical.






