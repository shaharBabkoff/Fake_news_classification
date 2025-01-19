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
## 3. Logistic Regression Implementation
A logistic regression model was implemented with TF-IDF features and hyperparameter optimization through grid search.
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
### Error Analysis
- **Validation Performance**: 66.96% accuracy
- **Observation**: Significant drop in validation accuracy suggests overfitting
- **Potential Improvements**: Feature selection or stronger regularization could help generalization
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
