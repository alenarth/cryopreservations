import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

class HepatocyteDMSOPredictor(nn.Module):
    def __init__(self):
        super(HepatocyteDMSOPredictor, self).__init__()
        self.fc1 = nn.Linear(1, 32)  # Aumentei os neurônios para capturar mais padrões
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()

    def prepare_data(self):
        X = self.data['% DMSO'].values.reshape(-1, 1)
        y = self.data['% APÓS O DESCONGELAMENTO'].values.reshape(-1, 1)

        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)  # Escala a saída também

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), \
               torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    def train(self, epochs=700, lr=0.0005):  # Aumentei o número de épocas e reduzi a LR
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model = HepatocyteDMSOPredictor()
        
        criterion = nn.L1Loss()  # MAE para lidar melhor com discrepâncias
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            predictions = self.model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                self.model.eval()
                test_loss = criterion(self.model(X_test), y_test)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Test Loss: {test_loss.item()}")

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        X = self.scaler_X.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        prediction = self.model(X).item()
        prediction = self.scaler_y.inverse_transform([[prediction]])[0][0]  # Inverte a escala para obter o valor real
        return prediction