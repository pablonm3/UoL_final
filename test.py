from deap import base, creator, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
import pandas as pd

# Load your data
# Assuming df is your DataFrame and it has columns 'text' and 'label'
df = pd.read_csv('datasets/Sentiment140.csv', encoding='latin-1')

print(df.columns)
