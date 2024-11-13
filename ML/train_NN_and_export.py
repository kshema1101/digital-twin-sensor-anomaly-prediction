import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

def preprocessing(file_name):
    data = pd.read_csv(file_name)
    data = data.dropna()
    scaler = MinMaxScaler()
    data[["temperature", "vibration"]] = scaler.fit_transform(data[["temperature", "vibration"]])
    X = data[["temperature", "vibration"]]
    y = data["fault_label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
    # convert data to pytorch tensor
    X_train_tensor = torch.Tensor(X_train.values)
    X_test_tensor = torch.Tensor(X_test.values)
    y_train_tensor = torch.Tensor(y_train.values)
    y_test_tensor = torch.Tensor(y_test.values)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


class FaultDetectionNN(nn.Module):
    def __init__(self):
        super(FaultDetectionNN,self).__init__()
        self.layer1 = nn.Linear(2,16) # input 2 features temp and vibration
        self.layer2 = nn.Linear(16,8) # hidden layer
        self.output = nn.Linear(8,1) # output layer
        self.sigmoid = nn.Sigmoid() # activation function

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
    
#train the model
def train_model(model, X_train, X_test, y_train, y_test, epochs=15, lr =0.1 ):
    criterion = nn.BCELoss() #binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        #forward pass
        output = model(X_train)
        loss = criterion(output, y_train.unsqueeze(1))

        #backward pass
        optimizer.zero_grad() #clear the gradients
        loss.backward()
        optimizer.step()
# evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval() #set model to evaluation mode
    with torch.no_grad(): #stop calculation of gradient
        y_pred = model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = ((y_pred_class == y_test.unsqueeze(1)).sum().item() )/ y_test.size(0)
        print("Accuracy: ", accuracy)
  

def export_to_onnx(model, file_name):
    dummy_input = torch.randn(1, 2)  # Example input size, adjust as necessary
    torch.onnx.export(model, dummy_input, file_name,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
    print("Model exported to ONNX format")  


if __name__ == "__main__":
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocessing("../synthetic_data.csv")
    model = FaultDetectionNN()
    train_model(model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    evaluate_model(model, X_test_tensor, y_test_tensor)
    #save the model
    torch.save(model.state_dict(), "fault_detection_model.pth")
    print("Model saved successfully")

    # Export the model to ONNX format
    export_to_onnx(model, "fault_detection_model.onnx")
    print("Model exported to ONNX format.")