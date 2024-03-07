import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the XOR class
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        # Define network layers
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


if __name__ == "__main__":
    # Initialize the network, loss function, and optimizer
    xor_net = XORNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(xor_net.parameters(), lr=0.1)

    # XOR input and output
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

    # Lists for storing loss and accuracy values
    losses = []
    outputs = []

    # Train the network
    epochs = 40000
    for epoch in range(epochs):  # Increased epochs for better convergence
        optimizer.zero_grad()
        output = xor_net(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        # Store loss
        losses.append(loss.item())

        # Print loss
        if epoch % 1000 == 0:
            print(f'Epoch [{epoch}/{epochs}] Loss: {loss.item()}')

    # Store final output
    with torch.no_grad():
        for inp in X:
            outputs.append(xor_net(inp).item())

    # Plotting the loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting the output after training
    plt.figure(figsize=(10, 5))
    plt.plot(outputs, 'ro', label='Final Outputs')
    plt.title('Network Outputs after Training')
    plt.xlabel('Input Configurations')
    plt.ylabel('Output Value')
    plt.xticks(range(4), ['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
    plt.legend()
    plt.show()

    torch.save(xor_net.state_dict(), 'xor_net.pth')
