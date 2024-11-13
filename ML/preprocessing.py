import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    # Load the synthetic data
    data = pd.read_csv(file_path)
    
    # Check for missing values
    if data.isnull().values.any():
        data = data.dropna()

    # Normalize the temperature and vibration data
    scaler = MinMaxScaler()
    data[['temperature', 'vibration']] = scaler.fit_transform(data[['temperature', 'vibration']])
    
    return data
def feature_engineering(data):
    data["temp_rolling_mean"] = data["temperature"].rolling(window =10).mean()
    data["vib_rolling_mean"] = data["vibration"].rolling(window =10).mean()
    data["temp_diff"] = data["temperature"].diff()
    data["vib_diff"] = data["vibration"].diff()
    data = data.dropna()
    return data
def split_data(data):
    X = data[["temperature","vibration","temp_rolling_mean","vib_rolling_mean","temp_diff","vib_diff"]]
    y = data["fault_label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    data = preprocess_data('synthetic_data.csv')
    data = feature_engineering(data)
    X_train, X_test, y_train, y_test = split_data(data)
    print(data.head())  # Preview the normalized data
