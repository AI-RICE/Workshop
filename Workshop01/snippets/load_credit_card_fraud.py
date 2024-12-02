# Credit Card Fraud Detection Problem
#
# Goal:
# Classify credit card transactions as either "Fraud" or "Not Fraud" using a neural network.
# This is a binary classification problem with a highly imbalanced dataset, as fraudulent
# transactions represent only a small fraction of all transactions.
#
# Challenges:
# 1. Class Imbalance: The dataset is highly imbalanced, with legitimate transactions far
#    outnumbering fraudulent ones, which can lead to biased models that underperform in
#    detecting fraud.
# 2. Feature Scaling: Since the features vary in scale, standardization is necessary to
#    optimize the performance of the neural network.
#
# Solution Approach:
# 1. Oversampling: To address class imbalance, we oversample the minority class (fraud cases)
#    to create a balanced dataset, allowing the model to learn effectively from both classes.
# 2. Feature Scaling: Standardize features for more stable and efficient training.
# 3. Neural Network: A neural network is trained to classify transactions as "Fraud" or "Not
#    Fraud," using balanced data to ensure fair performance on both classes.
#
# Expected Outcome:
# The model will classify new transactions as either "Fraud" or "Not Fraud." Performance is
# evaluated based on accuracy on test data, with a focus on effective fraud detection.

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from sklearn.utils import resample

# Load the credit card fraud dataset
data = pd.read_csv('creditcard.csv')

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.LongTensor(y.values)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Define the neural network
class FraudNet(nn.Module):
    def __init__(self):
        super(FraudNet, self).__init__()
        self.layer1 = nn.Linear(X_tensor.shape[1], 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 2)  # 2 classes: Fraud or Not Fraud
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# Initialize the model, loss function, and optimizer
model = FraudNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'\nTest Accuracy: {accuracy:.2%}')


# Example prediction function
def predict_fraud(features):
    with torch.no_grad():
        # Convert input list to DataFrame with original column names
        features_df = pd.DataFrame([features], columns=X.columns)
        features_scaled = scaler.transform(features_df)
        features_tensor = torch.FloatTensor(features_scaled)
        output = model(features_tensor)
        _, predicted = torch.max(output.data, 1)
        return "Fraud" if predicted.item() == 1 else "Not Fraud"
        from sklearn.metrics import classification_report
        print(classification_report(y, yp))



def predict_fraud(features):
    with torch.no_grad():
        # Convert input list to DataFrame with original column names
        features_df = pd.DataFrame([features], columns=X.columns)
        features_scaled = scaler.transform(features_df)
        features_tensor = torch.FloatTensor(features_scaled)
        output = model(features_tensor)
        _, predicted = torch.max(output.data, 1)
        print(classification_report(y, yp))


# Example usagec
sample_features = X.iloc[0].tolist()  # Example transaction features
print(f"\nPredicted class for sample transaction:")
print(predict_fraud(sample_features))
