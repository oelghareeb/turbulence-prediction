# Turbulence Prediction Neural Network

This repository contains a neural network model for predicting turbulence properties based on input features derived from fluid dynamics data. The project leverages PyTorch for building and training the model and provides a modular structure for data preprocessing, model definition, training, and visualization.

## Project Structure

The repository is organized into the following files:

1. **`data_processing.py`**
   - Handles data loading and preprocessing.
   - Functions for combining multiple data files and scaling features and labels.

2. **`model.py`**
   - Defines the neural network architecture for the turbulence prediction task.

3. **`training.py`**
   - Contains functions for training and evaluating the model.
   - Includes utilities for plotting predictions versus ground truths.

4. **`main.py`**
   - Orchestrates the entire pipeline: data loading, preprocessing, training, evaluation, and visualization.

## Dataset

The dataset used in this project is from the **ML Turbulence Dataset** on Kaggle. It consists of the following fields:

- `Ux`: X-component of velocity
- `Uy`: Y-component of velocity
- `Cx`, `Cy`: Coordinate fields
- `p`: Pressure values (used as labels)

Ensure the dataset is downloaded and placed in the appropriate directory (e.g., `/kaggle/input/ml-turbulence-dataset/`).

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/oelghareeb/turbulence-prediction.git
   cd turbulence-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download the **ML Turbulence Dataset** from Kaggle.
   - Extract the data into the `/kaggle/input/ml-turbulence-dataset/` directory.

## Usage

### Training the Model

Run the main script to train the model and visualize the results:
```bash
python main.py
```

### Output Visualization

- Predictions and ground truths are plotted for different test cases.
- Ensure `matplotlib` is installed to visualize the results.

## Model Architecture

The neural network consists of the following layers:
- Input layers for `Ux` and `Uy` features.
- Hidden layers with ReLU activations.
- A concatenation layer to combine processed features.
- Output layer predicting turbulence properties.

## Results

- Tricontour plots for predicted and ground truth turbulence properties are generated for visualization.
- Example plots include predictions and truths for two different test cases.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For questions or feedback, feel free to contact [oelghareeb@gmail.com].

