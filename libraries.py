# === Data Handling & Text Utilities ===
import pandas as pd                 # For data manipulation and analysis
import re                           # For regular expressions used in text preprocessing
import nltk                         # Natural Language Toolkit for text processing
import contractions                 # For expanding contractions (e.g., "don't" -> "do not")
import matplotlib.pyplot as plt     # For creating visualizations like plots
from wordcloud import WordCloud     # For generating word clouds
from collections import Counter     # For counting frequency of words/n-grams
from nltk.util import ngrams        # For generating n-grams from tokenized text

# === Scikit-learn Modules ===
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold        # Splitting data & hyperparameter tuning
from sklearn.preprocessing import LabelEncoder, label_binarize                                              # Encoding labels and preparing for ROC curve
from sklearn.multiclass import OneVsRestClassifier                                                          # Plotting ROC Curve by label 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer                                # Converting text to feature vectors
from sklearn.linear_model import LogisticRegression                                                         # Logistic Regression classifier
from sklearn.svm import LinearSVC                                                                           # Support Vector Machine classifier (linear kernel)
from sklearn.ensemble import RandomForestClassifier                                                         # Random Forest classifier
from sklearn.naive_bayes import MultinomialNB                                                               # Naive Bayes classifier for discrete features
from sklearn.neighbors import KNeighborsClassifier                                                          # K-Nearest Neighbors classifier
from sklearn.tree import DecisionTreeClassifier                                                             # Decision Tree classifier
from sklearn.metrics import (                                                                               # Metrics and visualizations for model evaluation
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay, accuracy_score
)

# === NLTK Resources ===
from nltk.corpus import stopwords                                                                # Common stopwords (e.g., "the", "and", "is")
from nltk.stem import WordNetLemmatizer, PorterStemmer                                           # Lemmatizer for reducing words to their base form

import random                                                                                    # To revert from static state to random state 

pd.set_option('display.max_columns', None)                                                       # Show all columns
pd.set_option('display.width', 1000)                                                             # Set a wider display width
pd.set_option("display.float_format", "{:.6f}".format)                                           # Control the formatting at display time, not just during storage.
