import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from data_processing import load_combined_array, preprocess_data
from model import TurbulenceModel
from training import train_model, evaluate_model, plot_results


# Main execution flow
def main():
    dataset = 'komegasst'
    case1 = ['PHLL_case_1p0']
    case2 = ['PHLL_case_0p5']

    # Load data
    x1 = load_combined_array(case1, 'Cx')
    y1 = load_combined_array(case1, 'Cy')
    x2 = load_combined_array(case2, 'Cx')
    y2 = load_combined_array(case2, 'Cy')

    Ux1 = load_combined_array(case1, 'Ux')
    Uy1 = load_combined_array(case1, 'Uy')
    p1 = load_combined_array(case1, 'p')

    Ux2 = load_combined_array(case2, 'Ux')
    Uy2 = load_combined_array(case2, 'Uy')
    p2 = load_combined_array(case2, 'p')

    # Combine data
    df1 = pd.DataFrame(np.column_stack((Ux1, Uy1, p1)), columns=['Ux', 'Uy', 'p'])
    df2 = pd.DataFrame(np.column_stack((Ux2, Uy2, p2)), columns=['Ux', 'Uy', 'p'])

    # Preprocess data
    df_features1, df_labels1, scaler_features1, scaler_labels1 = preprocess_data(df1[['Ux', 'Uy']], df1['p'],
                                                                                 ['Ux', 'Uy'], 'p')
    df_features2, df_labels2, scaler_features2, scaler_labels2 = preprocess_data(df2[['Ux', 'Uy']], df2['p'],
                                                                                 ['Ux', 'Uy'], 'p')

    # Merge data for training
    merge_x = np.concatenate((x1, x2))
    merge_y = np.concatenate((y1, y2))

    # Convert data into PyTorch tensors
    inputs1 = torch.tensor(df_features1, dtype=torch.float32)
    inputs2 = torch.tensor(df_features2, dtype=torch.float32)
    labels1 = torch.tensor(df_labels1, dtype=torch.float32)
    labels2 = torch.tensor(df_labels2, dtype=torch.float32)

    # Concatenate inputs and labels from both datasets
    inputs = torch.cat((inputs1, inputs2), dim=0)
    labels = torch.cat((labels1, labels2), dim=0)

    # Create a DataLoader for batching
    dataset = TensorDataset(inputs, inputs, labels)  # both inputs1 and inputs2 as inputs
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize the model
    model = TurbulenceModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5E-4)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=10)

    # Evaluate the model
    predictions, truths = evaluate_model(model, train_loader)

    # Visualize the results
    plot_results(x1, y1, predictions, truths, x2, y2)


# Run the main function
if __name__ == '__main__':
    main()
