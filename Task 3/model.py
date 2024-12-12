import torch
import torch.nn as nn

# Neural network model definition in PyTorch
class TurbulenceModel(nn.Module):
    def __init__(self):
        super(TurbulenceModel, self).__init__()
        self.hidden1 = nn.Linear(2, 20)  # Input layer for Ux, Uy
        self.hidden2 = nn.Linear(2, 20)  # Another input layer for Ux, Uy
        self.hidden3 = nn.Linear(40, 20) # After concatenation
        self.hidden4 = nn.Linear(20, 20)
        self.hidden5 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 1)   # Output layer for turbulence property

    def forward(self, input1, input2):
        x1 = torch.relu(self.hidden1(input1))
        x2 = torch.relu(self.hidden2(input2))
        x = torch.cat((x1, x2), dim=1)  # Concatenate
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        x = torch.relu(self.hidden5(x))
        output = self.output(x)
        return output
