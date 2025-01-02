# Stock Price Prediction

This repository contains two Jupyter notebooks for predicting stock prices using different machine learning models: LSTM without Attention and Transformer.

## Notebooks

### 1. stockprediction_LSTMwithoutAttention.ipynb

This notebook demonstrates how to predict stock prices using an LSTM model without attention mechanism.

#### Steps:

1. **Download TSMC Stock Data**:
    - Use Yahoo Finance to download TSMC stock data.
    - Calculate the average price from the high and low prices.

2. **Data Preprocessing**:
    - Normalize the data using MinMaxScaler.
    - Apply Exponential Moving Average (EMA) smoothing.
    - Create sequences for training, validation, and testing.

3. **Define LSTM Model**:
    - Define an LSTM model with multiple layers and dropout for regularization.

4. **Train the Model**:
    - Train the LSTM model using the training data.
    - Validate the model using the validation data.
    - Plot training and validation loss.

5. **Evaluate the Model**:
    - Predict stock prices using the test data.
    - Plot the actual vs predicted prices.
    - Calculate and display the Mean Absolute Error (MAE).

### 2. stockprediction_Transformer.ipynb

This notebook demonstrates how to predict stock prices using a Transformer model.

#### Steps:

1. **Download TSMC Stock Data**:
    - Use Yahoo Finance to download TSMC stock data.
    - Calculate the average price from the high and low prices.

2. **Data Preprocessing**:
    - Normalize the data using MinMaxScaler.
    - Apply Exponential Moving Average (EMA) smoothing.
    - Create sequences for training, validation, and testing.

3. **Define Transformer Model**:
    - Define a Transformer model with positional encoding.

4. **Train the Model**:
    - Train the Transformer model using the training data.
    - Validate the model using the validation data.
    - Plot training and validation loss.

5. **Evaluate the Model**:
    - Predict stock prices using the test data.
    - Plot the actual vs predicted prices.
    - Calculate and display the Mean Absolute Error (MAE).

## Requirements

- Python 3.10.13
- yfinance
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- seaborn
- tqdm

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/stock-price-prediction.git
    cd stock-price-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebooks:
    ```sh
    jupyter notebook stockprediction_LSTMwithoutAttention.ipynb
    jupyter notebook stockprediction_Transformer.ipynb
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](http://_vscodecontentref_/2) file for details.