import sys
import math
from pathlib import Path
import numpy as np
from PBP_net import PBP_net


def load_data(datafile_path):
    """Load dataset
    Needs to be txt format with d columns
    X: first d-1 columns
    y: last column
    """

    datafile_path = Path(datafile_path)
    if datafile_path.exists():
        data = np.loadtxt(datafile_path)
    else:
        print("Dataset file does not exist: {}".format(datafile_path))
        sys.exit(1)
    X = data[:, 0: data.shape[1]-1]
    y = data[:, data.shape[1]-1]
    return X, y


def create_data_split(X, y, proportion=0.9):
    """Create data split
    Default: Create the train and test sets with 90% and 10% of the data
    """

    np.random.seed(1)
    permutation = np.random.choice(range(X.shape[0]),
                                   X.shape[0], replace=False)
    size_train = int(np.round(X.shape[0]*proportion))
    index_train = permutation[0:size_train]
    index_test = permutation[size_train:]

    return X[index_train, :], y[index_train], X[index_test, :], y[index_test]


def main():
    """Main entry point for script"""
    X, y = load_data("boston_housing.txt")
    X_train, y_train, X_test, y_test = create_data_split(X, y, 0.9)

    # We construct the network with one hidden layer with two-hidden layers
    # with 50 neurons in each one and normalizing the training features to have
    # zero mean and unit standard deviation in the trainig set.
    n_hidden_units = 50
    n_epochs = 40
    net = PBP_net.PBP_net(X_train, y_train, [n_hidden_units, n_hidden_units],
                          normalize=True, n_epochs=n_epochs)

    # We make predictions for the test set
    m, v, v_noise = net.predict(X_test)

    # We compute the test RMSE
    rmse = np.sqrt(np.mean((y_test - m)**2))
    print("RMSE: {}".format(rmse))

    # We compute the test log-likelihood
    test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) -
                      0.5 * (y_test - m)**2 / (v + v_noise))
    print("Log likelihood: {}".format(test_ll))


if __name__ == "__main__":
    main()
