import random
from datetime import datetime
import numpy
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import wandb
from sklearn.metrics import balanced_accuracy_score, accuracy_score

batch_size = 32
big_batch_size = 1000
output_size = 2
test_accuracy_by_small_batch = False
max_data_to_log = batch_size
model_path_to_load = "models/fy4i7bah_MLP.pth"
model_config = {
    'model_name': 'MLP',
    'model_type': 3,
    'hidden_layer_num': 5,
    'hidden_size1': 43,
    'hidden_size2': 78,
    'hidden_size3': 80,
    'hidden_size4': 47,
    'hidden_size5': 13,
    'epoch_num': 15,
    'learning_rate': 0.0008952
}


reload_model = False
test_interval = 50
toy = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading dataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size, big_train_dataset, big_test_dataset = ld.get_data_set(
    batch_size, big_batch_size, toy)


# Special matrix multiplication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,
                                                                                  out_channels,
                                                                                  device=device)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels, device=device), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, exrnn_input_size, exrnn_output_size, config):
        super(ExRNN, self).__init__()

        self.hidden_size = config['hidden_size']
        self.sigmoid = torch.sigmoid
        self.hidden_activation = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(exrnn_input_size + self.hidden_size, self.hidden_size, device=device)
        self.hidden2output = nn.Linear(self.hidden_size, exrnn_output_size, device=device)

    def name(self):
        return "RNN"

    def forward(self, x, exrnn_hidden_state):
        # Implementation of RNN cell
        combined = torch.cat((x, exrnn_hidden_state), dim=1)
        h_t = self.hidden_activation(self.in2hidden(combined))
        exrnn_output = self.hidden2output(h_t)

        return exrnn_output, h_t

    def init_hidden(self, exrnn_batch_size):
        return torch.zeros(exrnn_batch_size, self.hidden_size, device=device)


# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, exgru_input_size, exgru_output_size, config):
        super(ExGRU, self).__init__()
        self.hidden_size = config['hidden_size']

        # GRU Cell weights
        self.w_z = nn.Linear(exgru_input_size + self.hidden_size, self.hidden_size, device=device)
        self.w_r = nn.Linear(exgru_input_size + self.hidden_size, self.hidden_size, device=device)
        self.w = nn.Linear(exgru_input_size + self.hidden_size, self.hidden_size, device=device)
        self.hidden2output = nn.Linear(self.hidden_size, exgru_output_size, device=device)

        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

    def name(self):
        return "GRU"

    def forward(self, x, exgru_hidden_state):
        # Implementation of GRU cell
        combined = torch.cat((x, exgru_hidden_state), dim=1)
        z_t = self.sigmoid(self.w_z(combined))
        r_t = self.sigmoid(self.w_r(combined))
        point_wise = torch.mul(r_t, exgru_hidden_state)
        combined2 = torch.cat((x, point_wise), dim=1)
        h_hat_t = self.tanh(self.w(combined2))

        exgru_hidden = torch.mul(1 - z_t, exgru_hidden_state) + torch.mul(z_t, h_hat_t)
        exgru_output = self.hidden2output(exgru_hidden)

        return exgru_output, exgru_hidden

    def init_hidden(self, exgru_batch_size):
        return torch.zeros(exgru_batch_size, self.hidden_size, device=device)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(ExMLP, self).__init__()

        self.hidden_layer_num = config['hidden_layer_num']
        self.ReLU = torch.nn.ReLU()
        self.sigmoid = torch.sigmoid

        self.layers = []
        self.layers.append(MatMul(input_size, config['hidden_size1']))
        # Token-wise MLP network weights
        for i in range(1, self.hidden_layer_num):
            self.layers.append(MatMul(config[f'hidden_size{i}'], config[f'hidden_size{i + 1}']))

        self.layers.append(MatMul(config[f'hidden_size{self.hidden_layer_num}'], output_size))
        self.layers = nn.ModuleList(self.layers)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation
        for i in range(self.hidden_layer_num):
            x = self.layers[i](x)
            x = self.ReLU(x)

        x = self.layers[-1](x)
        # NO SIGMOID

        return x


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(ExRestSelfAtten, self).__init__()
        config = dict(config)
        config['hidden_size0'] = input_size
        self.hidden_layer_num = config['hidden_layer_num']
        self.attention_location = config['attention_location']
        self.attention_size = config['attention_size']
        self.input_size = input_size
        self.output_size = output_size
        self.middle_hidden_size = config[f'hidden_size{self.attention_location - 1}']
        self.sqrt_hidden_size = np.sqrt(float(self.middle_hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(3)
        self.sigmoid = torch.sigmoid

        # Token-wise MLP + Restricted Attention network implementation

        self.W_q = MatMul(self.middle_hidden_size, self.middle_hidden_size, use_bias=False)
        self.W_k = MatMul(self.middle_hidden_size, self.middle_hidden_size, use_bias=False)
        self.W_v = MatMul(self.middle_hidden_size, self.middle_hidden_size, use_bias=False)

        self.layers = []
        self.layers.append(MatMul(input_size, config['hidden_size1']))
        for i in range(1, self.hidden_layer_num):
            self.layers.append(MatMul(config[f'hidden_size{i}'], config[f'hidden_size{i + 1}']))

        self.layers.append(MatMul(config[f'hidden_size{self.hidden_layer_num}'], output_size))
        self.layers = nn.ModuleList(self.layers)

    def name(self):
        return "MLP_attention"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation

        for i in range(self.attention_location - 1):
            x = self.layers[i](x)
            x = self.ReLU(x)

        # -----------------------------------------ATTENTION START---------------------------------------

        # generating x in offsets between -attention_size and attention_size
        # with zero padding at the ends
        padded = pad(x, (0, 0, self.attention_size, self.attention_size, 0, 0))

        x_nei = []
        for k in range(-self.attention_size, self.attention_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, self.attention_size:-self.attention_size, :]

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer

        query = self.W_q(x)
        keys = self.W_k(x_nei)
        values = self.W_v(x_nei)

        query = torch.unsqueeze(query, dim=2)
        keys_transpose = torch.transpose(keys, dim0=2, dim1=3)
        dist = torch.matmul(query, keys_transpose)
        dist = dist / self.sqrt_hidden_size
        attention_weights = self.softmax(dist)
        v_out = torch.matmul(attention_weights, values)
        x = torch.squeeze(v_out)

        # -----------------------------------------ATTENTION END---------------------------------------

        for i in range(self.attention_location - 1, self.hidden_layer_num):
            x = self.layers[i](x)
            x = self.ReLU(x)

        x = self.layers[-1](x)  # Mapping to output in length 2

        return x, attention_weights


# prints portion of the review (20-30 first words), with the sub-scores each word obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, numpy_subs, is_actually_pos, is_actually_neg):
    n_first_word = 20
    sub_positive = numpy_subs[:, 0]
    sub_negative = numpy_subs[:, 1]

    print("___________________________Words and sub-scores:___________________________")
    for i in range(min(n_first_word, len(rev_text))):
        print(
            f"Word \"{rev_text[i]}\": \n pos: {sub_positive[i]} \n neg: {sub_negative[i]}")

    summed = numpy.sum(numpy_subs, axis=0)

    print(f"Final scores: \npos: {summed[0]}\nneg: {summed[1]}")

    softmx = torch.nn.Softmax(dim=0)(torch.from_numpy(summed))

    print(f"After softmax: \npos: {softmx[0]}\nneg: {softmx[1]}")

    argmax = lambda x, y: "positive" if x > y else "negative"

    print(
        f"Our prediction is: {argmax(softmx[0], softmx[1])}\n   actual label:   "
        f"{argmax(is_actually_pos, is_actually_neg)}")


def get_config(run_recurrent, use_RNN,
               attention_size):
    sweep_config = {}
    sweep_config['method'] = 'random'
    sweep_config['metric'] = {'name': 'balanced_accuracy', 'goal': 'maximize'}
    now = datetime.now()
    time_str = now.strftime("%d-%m-%Y__%H-%M-%S")

    if run_recurrent:
        if use_RNN:
            sweep_config['name'] = f"RNN_{time_str}"
            param_dict = {
                'model_name': {'value': 'RNN'},
                'model_type': {'value': 1},
                'hidden_size': {'distribution': 'int_uniform', 'min': 64, 'max': 128}
            }
        else:
            sweep_config['name'] = f"GRU_{time_str}"
            param_dict = {
                'model_name': {'value': 'GRU'},
                'model_type': {'value': 2},
                'hidden_size': {'distribution': 'int_uniform', 'min': 64, 'max': 128}
            }
    else:
        if attention_size == 0:
            sweep_config['name'] = f"MLP_{time_str}"
            param_dict = {
                'model_name': {'value': 'MLP'},
                'model_type': {'value': 3},
                'hidden_layer_num': {'distribution': 'int_uniform', 'min': 1, 'max': 5},
                'hidden_size1': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size2': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size3': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size4': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size5': {'distribution': 'int_uniform', 'min': 1, 'max': 80}

            }
        else:
            sweep_config['name'] = f"MLP_attention_{time_str}"
            param_dict = {
                'model_name': {'value': 'MLP_attention'},
                'model_type': {'value': 4},
                # In order to change the attention size easily in the future:
                'attention_size': {'distribution': 'int_uniform', 'min': 5, 'max': 5},
                'hidden_layer_num': {'distribution': 'int_uniform', 'min': 5, 'max': 5},
                'hidden_size1': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size2': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size3': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size4': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'hidden_size5': {'distribution': 'int_uniform', 'min': 1, 'max': 80},
                'attention_location': {'distribution': 'int_uniform', 'min': 1, 'max': 5}
            }
    sweep_config['parameters'] = param_dict
    sweep_config['parameters'].update(
        {"learning_rate": {'distribution': 'uniform', 'min': 0.0001, 'max': 0.001},
         'epoch_num': {'distribution': 'int_uniform', 'min': 3, 'max': 9}})
    return sweep_config


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        # ---------------------------------
        #  re-calculating settings
        # ---------------------------------
        run_recurrent = False
        if reload_model:
            config = model_config

        if config['model_name'] == 'RNN':
            run_recurrent = True
            model = ExRNN(input_size, output_size, config)
        elif config['model_name'] == 'GRU':
            run_recurrent = True
            model = ExGRU(input_size, output_size, config)
        elif config['model_name'] == 'MLP':
            model = ExMLP(input_size, output_size, config)
        else:
            model = ExRestSelfAtten(input_size, output_size, config)
        attention_size = 0 if 'attention_size' not in dict(config).keys() else config['attention_size']
        # ---------------------------------
        #  done re-calculating
        # ---------------------------------

        print("Using model: " + model.name())

        if reload_model:
            print("Reloading model")
            x = torch.load(model_path_to_load)
            model.load_state_dict(x)

        wandb.watch(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        train_loss = 1.0
        test_loss = 1.0

        # training steps in which a test step is executed every test_interval
        all_itr = 0
        num_epochs = config['epoch_num']
        for epoch in range(num_epochs):

            itr = 0  # iteration counter within each epoch

            for labels, reviews, reviews_text in train_dataset:  # getting training batches

                itr += 1
                all_itr += 1

                if (itr + 1) % test_interval == 0:
                    test_iter = True
                    if test_accuracy_by_small_batch:
                        labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch
                    else:
                        labels, reviews, reviews_text = next(iter(big_test_dataset))
                else:
                    test_iter = False
                # TRAINING:

                # Recurrent nets (RNN/GRU)

                if run_recurrent:
                    hidden_state = model.init_hidden(int(labels.shape[0]))

                    for i in range(num_words):
                        # the slicing in the row below means "take the i row of the reviews" or
                        # "take the ith word from every review in the batch"
                        output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE

                else:

                    # Token-wise networks (MLP / MLP + Attention)

                    sub_score = []
                    if attention_size > 0:
                        # MLP + attention
                        sub_score, attention_weights = model(reviews)
                    else:
                        # MLP
                        sub_score = model(reviews)

                    output = torch.mean(sub_score, 1)

                # cross-entropy loss

                loss = criterion(output, labels)

                # optimize in training iterations

                if not test_iter:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # averaged losses
                if test_iter:
                    test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                else:
                    train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

                # 0 means its negative review, 1 means its positive review
                if test_iter:
                    y_pred = [1 if output[i][0] > output[i][1] else 0 for i in range(output.shape[0])]
                    y_true = [1 if labels[i][0] > labels[i][1] else 0 for i in range(labels.shape[0])]
                    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                    accuracy = accuracy_score(y_true, y_pred, normalize=True)
                    wandb.log({"epoch": epoch,
                               "test_loss": test_loss,
                               "train_loss": train_loss,
                               "accuracy": accuracy,
                               "balanced_accuracy": balanced_accuracy},
                              step=all_itr)
                    # ------------------------------------------------------
                    #                   LOGGING REVIEWS
                    # ------------------------------------------------------
                    columns = ["Review", "Predicted Sentiment", "True Sentiment"]
                    y_pred_text = ["positive" if pred == 1 else "negative" for pred in y_pred]
                    y_true_text = ["positive" if pred == 1 else "negative" for pred in y_true]
                    data = [[reviews_text[k], y_pred_text[k], y_true_text[k]] for k in range(
                        min(max_data_to_log, len(reviews_text)))]
                    table = wandb.Table(data=data, columns=columns)
                    wandb.log({"examples": table}, step=all_itr)

                    # ------------------------------------------------------
                    #                   /LOGGING REVIEWS
                    # ------------------------------------------------------

                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Step [{itr + 1}/{len(train_dataset)}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Test Loss: {test_loss:.4f}"
                    )

                    if not run_recurrent:
                        numpy_subs = sub_score.cpu().detach().numpy()
                        labels = labels.cpu().detach().numpy()
                        print_review(reviews_text[0], numpy_subs[0, :, :], labels[0, 0],
                                     labels[0, 1])

                    # saving the model
                    if not reload_model:
                        torch.save(model.state_dict(), f"models/{config._settings.run_id}_{model.name()}.pth")


def main(run_recurrent, use_RNN, attention_size):
    sweep_id = wandb.sweep(get_config(run_recurrent, use_RNN, attention_size), project="test_ex2_tests",
                           entity="malik-noam-idl")
    wandb.agent(sweep_id, train, count=1)


if __name__ == '__main__':
    keepRunning = "True"
    while keepRunning == "True":
        run_recurrent1 = random.choice([True, False])
        use_RNN1 = random.choice([True, False])
        attention_size1 = random.choice([0, 5])
        main(run_recurrent=run_recurrent1, use_RNN=use_RNN1, attention_size=attention_size1)
        with open("keepRunning.txt", "r") as file:
            keepRunning = file.readline()
