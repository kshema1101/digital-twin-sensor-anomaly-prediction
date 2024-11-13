from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

def preprocess_data(file_name):
    # read the data
    data = pd.read_csv(file_name)
    # remove the missing values
    data = data.dropna()
    #normalize temp and vibration data
    scaler = MinMaxScaler()
    data[["temperature","vibration"]] =  scaler.fit_transform(data[["temperature", "vibration"]])
    #split the data into train and test
    X = data[["temperature","vibration"]]
    y = data["fault_label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

def train_model_RF(X_train, y_train):
    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("Confusion Matrix: ", confusion)
    print("Classification Report: ", classification_rep)
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("../synthetic_data.csv")
    model = train_model_RF(X_train, y_train)
    evaluate_model(model, X_test, y_test)