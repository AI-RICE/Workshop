import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data
y = iris.target

# Convert to PyTorch tensors and scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.LongTensor(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)


# Define the neural network
class IrisNet(nn.Module):

    def __init__(self):
        super(IrisNet, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 6)
        self.layer3 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# Initialize the model, loss function, and optimizer

model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100

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


# Example prediction

def predict_iris(features):
    with torch.no_grad():
        features_scaled = scaler.transform([features])
        features_tensor = torch.FloatTensor(features_scaled)
        output = model(features_tensor)
        _, predicted = torch.max(output.data, 1)
        return iris.target_names[predicted.item()]


# Example usage

sample_features = [5.1, 3.5, 1.4, 0.2]  # Sample iris measurements
print(f"\nPredicted class for {sample_features}:")
print(predict_iris(sample_features))