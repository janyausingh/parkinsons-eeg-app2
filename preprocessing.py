
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    if 'class' not in df.columns:
        raise ValueError("Label column 'class' not found in data.")
    X = df.drop(columns=['class'])
    y = df['class']
    return X, y

def scale_data(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
