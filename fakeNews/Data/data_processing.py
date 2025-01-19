import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#nltk.download('stopwords')


def apply_stemming(content, stemmer):
    """
- Takes text content and applies several cleaning steps:
- Removes all non-alphabetic characters using regex ([^a-zA-Z])
- Converts text to lowercase
- Splits text into words
- Applies Porter Stemming (reduces words to their root form)
- Removes English stopwords (common words like "the", "is", "at")
- Joins words back into a single string
 """
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def get_training_data():
    """
- Loads training data from 'train.csv'
- Fills any null values with empty strings
- Applies stemming to the 'title' column
- Splits data into features (X) and labels (y)
- Uses train_test_split with:
    80% training, 20% testing split
    Stratified sampling (maintains label distribution)
    Random seed of 42 for reproducibility
-Applies TF-IDF vectorization:
    Maximum of 9000 features
    Converts text data into numerical format
- Returns processed training and test sets
    """
    train_df = pd.read_csv('../Data/train.csv')
    train_df = train_df.fillna('')
    stemmer = PorterStemmer()
    train_df['title'] = train_df['title'].apply(lambda title: apply_stemming(title, stemmer))

    X = train_df['title'].values
    y = train_df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    vectorizer = TfidfVectorizer(max_features=9000)
    vectorizer.fit(X_train)  # Only fit on training data

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test, vectorizer  # Return the vectorizer


# def get_validation_data(vectorizer):
#     """
# - Loads validation data from 'test.csv'
# - Applies the same preprocessing steps as training data
# - Uses TF-IDF vectorization with 9000 features
# - Loads true labels from 'final_compare.csv'
#     """
#     validation_df = pd.read_csv('../Data/test.csv')
#     validation_df = validation_df.fillna('')
#     stemmer = PorterStemmer()
#     validation_df['title'] = validation_df['title'].apply(lambda content: apply_stemming(content, stemmer))
#
#     X_validation = validation_df['title'].values
#     # Use the same vectorizer that was fit on training data
#     X_validation = vectorizer.transform(X_validation)
#
#     submit_df = pd.read_csv('../Data/final_compare.csv')
#     y_validation = submit_df['label']
#
#     return X_validation, y_validation
#

def get_training_validation_test_data():
    """
    Loads and splits data into training, validation, and test sets
    Returns processed and vectorized data
    """
    # Load the data (using your existing code)
    train_df = pd.read_csv('../Data/train.csv')
    train_df = train_df.fillna('')

    # Apply stemming (using your existing preprocessing)
    stemmer = PorterStemmer()
    train_df['title'] = train_df['title'].apply(lambda title: apply_stemming(title, stemmer))

    # Split features and labels
    X = train_df['title'].values
    y = train_df['label'].values

    # First split: separate training set (60%) from temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,  # 40% for validation and test combined
        stratify=y,
        random_state=42
    )

    # Second split: divide temp into validation (20%) and test (20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,  # Split temp set in half
        stratify=y_temp,
        random_state=42
    )

    # Vectorize the text (using your existing vectorizer)
    vectorizer = TfidfVectorizer(max_features=9000)
    vectorizer.fit(X_train)  # Fit only on training data

    # Transform all sets
    X_train = vectorizer.transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer