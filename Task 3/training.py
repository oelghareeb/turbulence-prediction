import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Training the model
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs1, inputs2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")


# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    truths = []
    with torch.no_grad():
        for inputs1, inputs2, labels in test_loader:
            outputs = model(inputs1, inputs2)
            predictions.append(outputs.numpy())
            truths.append(labels.numpy())

    predictions = np.concatenate(predictions)
    truths = np.concatenate(truths)

    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of truths: {truths.shape}")

    return predictions, truths


def plot_results(x1, y1, predictions, truths, x2, y2):
    # Ensure correct shapes for predictions and truths
    pred_case1 = predictions[:len(x1)].reshape(len(x1))  # For the first case
    truth_case1 = truths[:len(y1)].reshape(len(y1))  # For the first case

    pred_case2 = predictions[len(x1):].reshape(len(x2))  # For the second case
    truth_case2 = truths[len(y1):].reshape(len(y2))  # For the second case

    # Plotting the results
    fig = plt.figure(figsize=(12, 12))

    # Plot for Case 1
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # Case 1: Predictions
    ax1.tricontourf(x1, y1, pred_case1, levels=50, vmin=0, vmax=1)
    ax1.set_aspect(1)
    ax1.set_title('Predictions for Case 1')

    # Case 1: Truths
    ax2.tricontourf(x1, y1, truth_case1, levels=50, vmin=0, vmax=1)
    ax2.set_aspect(1)
    ax2.set_title('Truth for Case 1')

    # Plotting for Case 2
    fig2 = plt.figure(figsize=(12, 12))
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)

    # Case 2: Predictions
    ax3.tricontourf(x2, y2, pred_case2, levels=50, vmin=0, vmax=1)
    ax3.set_aspect(1)
    ax3.set_title('Predictions for Case 2')

    # Case 2: Truths
    ax4.tricontourf(x2, y2, truth_case2, levels=50, vmin=0, vmax=1)
    ax4.set_aspect(1)
    ax4.set_title('Truth for Case 2')

    plt.show()
