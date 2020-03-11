# Stock Price Prediction

This project predicts Apple's stock price by using Recurrent Neural Netowrk (RNN) + Long Shrot Term Memory (LSTM).

## Getting Started

Use any python IDE to open the project. I personally use Spyder from Anaconda.You can download both Anaconda or Spyder from the following links:
* [Anaconda](https://www.anaconda.com/distribution/) - The Data Science Platform for Python/R
* [Spyder](https://www.spyder-ide.org/) - The Scientific Python Development Environment

### Installation

Before running the program, type the following command to install the libraries that the project depends on

```
pip install numpy, pandas, matplotlib, sklearn, tensorflow
```
Or simply type the following:

```
pip install -r requirements.txt
```

## Running the tests

The description of each function is located on top of them. Please read them before running to understand the overall structure of the project.
This project predicts stock prices from its Open, High, Low, and Close using combinations of different models and hyperparameters.
The following is the graph of Actual v.s Prediction:

![Actual V.S Prediction](/data/Actual_vs_Prediction_Graph.png)

To explore more result, run the main.py file. The output will be showing more details.

## Deployment

Download any stock price charts from online (Ex: Kaggle) and insert the data to the model in order to test its accuracy.
* [Kaggle](https://www.kaggle.com/) - The Machine Learning and Data Science Community

## Built With

* [Python](https://www.python.org/) - The Programming Language
* [Tensorflow](https://www.tensorflow.org/) - The end-to-end open source machine learning platform

## Author

* **CSY** - [csy0522](https://github.com/csy0522)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
