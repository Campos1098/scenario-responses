import numpy as np

import train_parser
import test_parser

import math
import warnings

import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings ("ignore")


class Tennis_NN(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.lin_stack = nn.Sequential(
            nn.Linear(111, lr, True),
            nn.BatchNorm1d(lr),
            nn.ReLU(),
            nn.Linear(lr, 2, True),
            nn.Sigmoid()
        )

    def forward(self, xb):
        return self.lin_stack(xb)


# Returns the model and its associated optimiser
def get_model(lr, num):
    model = Tennis_NN(num)

    return model, optim.SGD(model.parameters(), lr = lr, momentum = 0.9)


# Returns the training and validation data
def get_data(train_ds,  bs):
    return (
        DataLoader(train_ds, batch_size = bs, shuffle = False)
    )


# Computes the accuracy of the model
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def loss_batch(model, loss_func, xb, yb, opt = None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, x_train, y_train, x_valid, y_valid):
    train = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    valid = np.zeros(epochs)
    valid_loss = np.zeros(epochs)
    inc_count = 0
    max_acc = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        train[epoch] = accuracy(model(x_train), y_train)
        train_loss[epoch] = loss_func(model(x_train), y_train)
        valid[epoch] = accuracy(model(x_valid), y_valid)
        valid_loss[epoch] = loss_func(model(x_valid), y_valid)

        if valid[epoch] > max_acc:
            max_acc = valid[epoch]
            inc_count = 0
            torch.save(model, "test")
        else: 
            inc_count += 1

        if inc_count == 300:
            print("EARLY STOPPING: " + str(epoch - 300) + " " + str(valid[epoch - 300]))
            inc_count = 0
            break

    model = torch.load("test")
    model.eval()
    print("SAVED: " + str(accuracy(model(x_valid), y_valid)))


def train_test(player, opponent, train_offset, valid_offset, test_offset, indices_file):
    # Retrieve rally data
    parser = train_parser.Parser(player, opponent, "charting-m-points.csv",
                               "charting-m-matches.csv")
    x, y = parser.parse()

    # Splitting the data and transofmring them into the appropriate types
    x = np.array(x)
    y = np.array(y)

    x_train = x[train_offset:valid_offset]
    y_train = y[train_offset:valid_offset]

    with open(indices_file, "r") as f:
        indices = f.readline()
        indices = indices[1: -1]
        indices = indices.split(", ")
        indices = [int(index) for index in indices]
    f.close()

    valid_indices = indices[: math.ceil(len(indices) / 2)]
    test_indices = indices[math.ceil(len(indices) / 2) :]

    x_valid = x[valid_indices]
    x_test = x[test_indices]
    
    y_valid = y[valid_indices]
    y_test = y[test_indices]

    x_train, x_valid, x_test, y_train, y_valid, y_test  = map(torch.tensor, (x_train, x_valid, x_test, 
                                                                             y_train, y_valid, y_test))

    x_train = x_train.float()
    x_valid = x_valid.float()
    x_test = x_test.float()
    y_train = y_train.long()
    y_valid = y_valid.long()
    y_test = y_test.long()

    # Setting some model training parameters
    lr = 0.0097  # learning rate
    epochs = 3000  # how many epochs to train for
    bs = 64  # batch size
    loss_func = F.cross_entropy
    n = len(x_train)

    model, opt = get_model(lr, 66)

    # Loading the data
    train_ds = TensorDataset(x_train, y_train)
    train_dl = get_data(train_ds, bs)

    # Train the model
    model.train()
    fit(epochs, model, loss_func, opt, train_dl, x_train, y_train, x_valid, y_valid)

    # Reporting post-training model performance
    model = torch.load("test")
    model.eval()
    
    return accuracy(model(x_test), y_test).item()


def opt(num, x_train, y_train, x_valid, y_valid):
    # Setting some model training parameters
    lr = num  # learning rate
    epochs = 3000  # how many epochs to train for
    bs = 64  # batch size
    loss_func = F.cross_entropy
    n = len(x_train)

    model, opt = get_model(lr, 217)

    # Loading the data
    train_ds = TensorDataset(x_train, y_train)
    train_dl = get_data(train_ds, bs)

    # Train the model
    model.train()
    fit(epochs, model, loss_func, opt, train_dl, x_train, y_train, x_valid, y_valid)

    # Reporting post-training model performance
    model = torch.load("test")
    model.eval()
    out = model(x_valid)
    print(accuracy(model(x_valid), y_valid).item())
    
    return accuracy(model(x_valid), y_valid).item()


def self_test(model_path, player, opponent, train_offset, valid_offset, test_offset, indices_file):
    model = torch.load(model_path)

    # Retrieve rally data
    parser = train_parser.Parser(player, opponent, "charting-m-points.csv",
                               "charting-m-matches.csv")
    x, y = parser.parse()

    # Splitting the data and transofmring them into the appropriate types
    x = np.array(x)
    y = np.array(y)

    with open(indices_file, "r") as f:
        indices = f.readline()
        indices = indices[1: -1]
        indices = indices.split(", ")
        indices = [int(index) for index in indices]
    f.close()

    test_indices = indices[math.ceil(len(indices) / 2) :]

    x_test = x[test_indices]
    y_test = y[test_indices]

    x_test, y_test  = map(torch.tensor, (x_test, y_test))

    x_test = x_test.float()
    y_test = y_test.long()

    # Reporting post-training model performance
    model.eval()

    acc = accuracy(model(x_test), y_test)
    
    result = " ".join(player.split("_")) + " vs. " + " ".join(opponent.split("_")) + ": " + str(np.round_(acc.item() * 100, 1)) + "%"

    return result


def test(model_path, player, opponent, cutoff):
    model = torch.load(model_path)

    # Retrieve rally data
    parser = test_parser.Parser(player, opponent, "charting-m-points.csv",
                               "charting-m-matches.csv", cutoff)
    x, y = parser.parse()

    # Splitting the data and transofmring them into the appropriate types
    x = np.array(x)
    y = np.array(y)

    x, y = map(torch.tensor, (x, y))

    x = x.float()
    y = y.long()

    model.eval()

    acc =  accuracy(model(x), y)

    result = " ".join(player.split("_")) + " vs. " + " ".join(opponent.split("_")) + ": " + str(np.round_(acc.item() * 100, 1)) + "%"

    return result


# model - the model to generate action probabilities with
# scenario - the scenario to generate action probabilities for
def evaluate_scenario(model, scenario):
    # Possible stroke and direction actions that can be taken
    stroke = ["Forehand groundstroke", "Backhand groundstroke", "Forehand slice", "Backhand slice",
              "Forehand volley", "Backhand volley", "Standard overhead/smash", "Backhand overhead/smash",
              "Forehand drop shot", "Backhand drop shot", "Forehand lob", "Backhand lob", "Forehand half-volley",
              "Backhand half-volley", "Forehand swinging volley", "Backhand swinging volley"]
    modifier = ["", "(approach shot), ", "(stop volley), ", "(approach shot, stop volley), "]
    direction = ["to the opponents right", "down the middle of the court", "to the opponents left"]
    
    print("The success probabilities for all actions within this scenario are:")
    model.eval()

    opt_shot = [(0, 0, 0), 0]

    # Evluate each possible action that the player can take
    for i in range(len(stroke)):
        for j in range(len(direction)):
            scenario[0][18 + i] = 1
            scenario[0][36 + j] = 1
            if i in [4, 5, 12, 13, 14, 15]:
                for k in range(len(modifier)):
                    scenario[0][34 + k] = 1
                    out = model(scenario).detach().numpy()
                    p = np.max(out, 1) * 100
                    if p[0] > opt_shot[1]:
                        opt_shot[0] = (i, k, j)
                        opt_shot[1] = p[0]
                    print(stroke[i] + ", " + modifier[k] + direction[j] + ": " + str(np.round(p[0], decimals = 2)) + "%")
                    scenario[0][21 + i] = 0
                    scenario[0][35 + j] = 0
                    scenario[0][34 + k] = 0
            else:
                for k in range(len(modifier) - 2):
                    scenario[0][34 + k] = 1
                    out = model(scenario).detach().numpy()
                    p = np.max(out, 1) * 100
                    if p[0] > opt_shot[1]:
                        opt_shot[0] = (i, k, j)
                        opt_shot[1] = p[0]
                    print(stroke[i] + ", " + modifier[k] + direction[j] + ": " + str(np.round(p[0], decimals = 2)) + "%")
                    scenario[0][21 + i] = 0
                    scenario[0][35 + j] = 0
                    scenario[0][34 + k] = 0
        print("")

    print("The optimal shot for this scenario is:")
    print(stroke[opt_shot[0][0]] + ", " + modifier[opt_shot[0][1]] + 
          direction[opt_shot[0][2]] + ": " + 
          str(np.round(opt_shot[1], decimals = 2)) + "%")


def opt_outer():
    # Retrieve rally data
    parser = train_parser.Parser("Roger_Federer", "Novak_Djokovic", "charting-m-points.csv",
                                "charting-m-matches.csv")
    x, y = parser.parse()

    # Splitting the data and transofmring them into the appropriate types
    x = np.array(x)
    # x = x[:, :79]
    y = np.array(y)

    x_train = x[0:2186]
    y_train = y[0:2186]

    with open("./ELEC4712-3/thesis/indices/rf_nd_indices.txt", "r") as f:
        indices = f.readline()
        indices = indices[1: -1]
        indices = indices.split(", ")
        indices = [int(index) for index in indices]
    f.close()

    valid_indices = indices[: math.ceil(len(indices) / 2)]
    test_indices = indices[math.ceil(len(indices) / 2) :]

    x_valid = x[valid_indices]
    x_test = x[test_indices]

    y_valid = y[valid_indices]
    y_test = y[test_indices]

    x_train, x_valid, x_test, y_train, y_valid, y_test  = map(torch.tensor, (x_train, x_valid, x_test, 
                                                                                y_train, y_valid, y_test))

    x_train = x_train.float()
    x_valid = x_valid.float()
    x_test = x_test.float()
    y_train = y_train.long()
    y_valid = y_valid.long()
    y_test = y_test.long()

    with open("./ELEC4712-3/thesis/indices/lr_results.txt", "a") as f:
        x = np.arange(1e-4, 1e-1, 0.0004)
        for i in range(214, len(x)):
            f.write(str(x[i]) + " " + str(opt(x[i], x_train, y_train, x_valid, y_valid)) + "\n")
    f.close()


print(test("p_o_rf_nd", "Novak_Djokovic", "Roger_Federer", "20151117"))
print(test("p_o_rf_nd", "Novak_Djokovic", "Rafael_Nadal", "20151117"))
print(self_test("p_o_rf_nd", "Roger_Federer", "Novak_Djokovic", 0, 2186, 0, "rf_nd_indices.txt"))
print(test("p_o_rf_nd", "Roger_Federer", "Rafael_Nadal", "20151117"))
print(test("p_o_rf_nd", "Rafael_Nadal", "Roger_Federer", "20151117"))
print(test("p_o_rf_nd", "Rafael_Nadal", "Novak_Djokovic", "20151117"))

# evaluate_scenario(model, torch.tensor([[
#  1, 0, 0, 0, 0, 0, 0, 0, 0,  # Current player position (0 - 8)
#  1, 0, 0, 0, 0, 0, 0, 0, 0,  # Previous player position (9 - 17)
#  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Player stroke type (18 - 33)
#  0, 0,  # Player shot modifier (34 - 35)
#  0, 0, 0,  # Player shot direction (36 - 38)
#  0, 0, 0, 0, 1, 0, 0, 0, 0,  # Current opponent position (39 - 47)
#  0, 0, 0, 1, 0, 0, 0, 0, 0,  # Previous opponent position (48 - 56)
#  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Opponent stroke type (57 - 72)
#  1, 0, 0,  # Opponent shot modifier (73 - 75)
#  0, 0, 1,  # Opponent shot direction (76 - 78)
#  -0.4184169507977564,  # Rally legnth (79)
#  1, 0, 0,  # Court surface (80 - 82)
#  0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Game score (83 - 100)
#  1, 0, 0, 0, 0, 0, 0, 0, 0,  # Set score (101 - 109)
#  0  # Best of (110)
# ]]))
