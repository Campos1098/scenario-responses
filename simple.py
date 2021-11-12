import numpy as np

import train_parser
import test_parser

import math
import warnings


warnings.filterwarnings ("ignore")


def test(train_player, train_opponent, train_cutoff, test_player, test_opponent, cutoff, ignore_context):
    parser = train_parser.Parser(train_player, train_opponent, "charting-m-points.csv",
                               "charting-m-matches.csv")
    
    x, y = parser.parse()
    x = np.array(x)

    if ignore_context:
        x = x[:, :79]

    y = np.array(y)

    x_train = x[:train_cutoff]
    y_train = y[:train_cutoff]

    guess = int(np.mean(y_train) >= 0.5)

    parser = test_parser.Parser(test_player, test_opponent, "charting-m-points.csv",
                               "charting-m-matches.csv", cutoff)
    
    x_test, y_test = parser.parse()
    x_test = np.array(x_test)

    if ignore_context:
        x_test = x_test[:, :79]

    y_test = np.array(y_test)

    y_pred = np.zeros(len(y_test))

    for i in range(len(x_test)):
        indices = np.where((x_train == x_test[i]).all(axis = 1))[0]

        if len(indices) == 0:
            y_pred[i] = guess

        else:
            y_pred[i] = int(np.mean(y_train[indices]) >= 0.5)

    result = " ".join(test_player.split("_")) + " vs. " + " ".join(test_opponent.split("_")) + ": " + str(np.round_(np.mean(y_pred == y_test) * 100, 1)) + "%"

    return result


def self_test(player, opponent, valid_offset, indices_file, ignore_context):
    # Retrieve rally data
    parser = train_parser.Parser(player, opponent, "charting-m-points.csv",
                               "charting-m-matches.csv")
    
    x, y = parser.parse()
    x = np.array(x)

    if ignore_context:
        x = x[:, :79]

    y = np.array(y)

    x_train = x[:valid_offset]
    y_train = y[:valid_offset]

    guess = int(np.mean(y_train) >= 0.5)
    
    with open(indices_file, "r") as f:
        indices = f.readline()
        indices = indices[1: -1]
        indices = indices.split(", ")
        indices = [int(index) for index in indices]
    f.close()

    test_indices = indices[math.ceil(len(indices) / 2) :]

    x_test = x[test_indices]
    y_test = y[test_indices]

    y_pred = np.zeros(len(y_test))

    for i in range(len(x_test)):
        indices = np.where((x_train == x_test[i]).all(axis = 1))[0]

        if len(indices) == 0:
            y_pred[i] = guess

        else:
            y_pred[i] = int(np.mean(y_train[indices]) >= 0.5)

    result = " ".join(player.split("_")) + " vs. " + " ".join(opponent.split("_")) + ": " + str(np.round_(np.mean(y_pred == y_test) * 100, 1)) + "%"

    return result


print(test("Roger_Federer", "Novak_Djokovic", 2186, "Novak_Djokovic", "Roger_Federer", "20151117", True))
print(test("Roger_Federer", "Novak_Djokovic", 2186, "Novak_Djokovic", "Rafael_Nadal", "20151117", True))
print(self_test("Roger_Federer", "Novak_Djokovic", 2186, "rf_nd_indices.txt", True))
print(test("Roger_Federer", "Novak_Djokovic", 2186, "Roger_Federer", "Rafael_Nadal", "20151117", True))
print(test("Roger_Federer", "Novak_Djokovic", 2186, "Rafael_Nadal", "Roger_Federer", "20151117", True))
print(test("Roger_Federer", "Novak_Djokovic", 2186, "Rafael_Nadal", "Novak_Djokovic", "20151117", True))