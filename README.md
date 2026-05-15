# Stock Direction Classifier
A feed forward neural network built with PyTorch that tries to predict whether Apple's (or any company's) stock prices will go up or down the next day.
> Pssst... spoiler: It DOES NOT work

## Objective
The goal of this project was to explore whether a simple deep learning model could beat random guessing at predicting daily stock direction, using only technical indicators as input features. Made for fun & giggles.

## Dataset (customizable)
- **Source**: Yahoo finance via `yfinance`
- **Ticker**: AAPL
- **Period**: January 2010 - January 2014 (about ~3500 training days)
- **Column used**: Close

## Features
| Feature | Description |
|---|---|
| `log_return` | Logarithmic daily return |
| `ma_5_ratio` | Ratio of price to its 5-day moving average |
| `ma_20_ratio` | Ratio of price to its 20-day moving average |
| `volatility_5` | Standard deviation of log returns over 5 days |
| `volatility_20` | Standard deviation of log returns over 20 days |
| `momentum_5` | Cumulative log return over 5 days |
| `momentum_20` | Cumulative log return over 20 days |
| `rsi` | Relative Strength Index (14-day window) |

The moving averages are expressed as ratios rather than raw prices, so that the values remain scale-independent across different time periods.

## Model Architecture
![StockDirectionClassifierArchitecture](StockDirectionClassifier.png "Logo Title Text 1")

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 50
- **Batch size**: 32

## Data Pipeline
1. Chronological split: 70% train / 20% validation / 10% test (no shuffling to respect time ordering)
2. StandardScaler fitted only on the training set, then applied to validation and test to prevent data leakage
3. Conversion to PyTorch tensors and loaded into DataLoaders

## Results
- **Baseline accuracy** (predict "Up" every day): ~52%
- **Model test accuracy:** ~52%
