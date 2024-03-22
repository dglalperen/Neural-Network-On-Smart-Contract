import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from TwoLayerXORNet import TwoLayerXORNet


if __name__ == "__main__":
    # Initialize the network, loss function, and optimizer
    xor_net = TwoLayerXORNet()
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

    torch.save(xor_net.state_dict(), 'two_layer_xor_net.pth')