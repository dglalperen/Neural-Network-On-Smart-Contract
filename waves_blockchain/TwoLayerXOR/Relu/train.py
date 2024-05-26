import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TwoLayerXORNet(nn.Module):
    def __init__(self):
        super(TwoLayerXORNet, self).__init__()
        # Define network layers with increased capacity
        self.layer1 = nn.Linear(2, 4)  # Increase to 4 neurons in the first layer
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)  # No activation on the output layer
        return x


# XOR data and labels
X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = torch.Tensor([[0], [1], [1], [0]])

# Model initialization
model = TwoLayerXORNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Using Adam optimizer

# Training loop
epochs = 5000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Backward and optimize
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
# Evaluate the model
model.eval()
with torch.no_grad():
    logits = model(X)  # raw model outputs
    probabilities = torch.sigmoid(logits)  # convert logits to probabilities
    predicted_labels = (
        probabilities > 0.5
    ).float()  # threshold probabilities to get binary labels
    print("Probabilities:", probabilities)
    print("Predicted labels:", predicted_labels)
    print("True labels:", Y)
    accuracy = (predicted_labels == Y).float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")


# Extract weights and biases
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
