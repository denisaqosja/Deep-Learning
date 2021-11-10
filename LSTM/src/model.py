import torch.nn as nn
import torch

class LSTM(nn.Module):
    """
    To obtain the data as a sequence, process each image in rows
    """

    def __init__(self, input_dim, hidden_dim, output_dim=10, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm_layer = nn.LSTM(input_dim, self.hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, output_dim)

    def init_states(self, batch_size):
        """Initialize cell and hidden states"""
        hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_dim)
        cell_state = torch.randn(self.num_layers, batch_size, self.hidden_dim)

        return hidden_state, cell_state

    def forward(self, input):
        b_size, channels, height, width = input.shape

        """sequential input of size: batch_size, sequence length, input_dim=rows=28"""
        input_sequential = input.view(b_size, channels * height, width)

        h_state, c_state = self.init_states(b_size)
        lstm_output, (hidden, cell) = self.lstm_layer(input_sequential, (h_state, c_state))

        """ouput is of size: batch_size, sequence_length, hidden_dim"""
        y = self.classifier(lstm_output[:, -1, :])

        return y



class LSTM_scratch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.input_dim = input_dim

        self.lstm_layer_1 = LSTM_layer(self.input_dim, hidden_dim)
        self.lstm_layer_2 = LSTM_layer(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(in_features=hidden_dim, out_features=output_dim)


    def forward(self, input):
        b_size, channels, height, width = input.shape
        seq_length = channels * height
        input_sequential = input.view(b_size, seq_length, self.input_dim)

        output_tensor = self.lstm_layer_1.forward(input_sequential)
        output_tensor = self.lstm_layer_2.forward(output_tensor)

        """classification, using the last element of the sequence at the last layer"""
        y = self.classifier(output_tensor[:, -1, :])

        return y

class LSTM_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.init_layers(input_dim, hidden_dim)

    def init_layers(self, input_dim, hidden_dim):
        # forget gate
        self.linear_I_forget = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear_H_forget = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.sigmoid_forget = nn.Sigmoid()
        # input gate
        self.linear_I_input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear_H_input = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.sigmoid_input = nn.Sigmoid()
        # cell memory gate
        self.linear_I_cell = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear_H_cell = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.tanh = nn.Tanh()
        # output gate
        self.linear_I_output = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear_H_output = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.sigmoid_output = nn.Sigmoid()

    def forward(self, input_sequential):
        b_size, seq_length, rows = input_sequential.shape
        hidden_state, cell_state = self.init_states(b_size)
        output_list = []

        for t_step in range(seq_length):
            input_rows = input_sequential[:, t_step, :]

            out_fg = self.forget_gate(input_rows, hidden_state)
            out_ig = self.input_gate(input_rows, hidden_state) * self.cell_memory(input_rows, hidden_state)

            new_cell_state = cell_state * out_fg + out_ig
            new_hidden_state = self.tanh(new_cell_state) * self.output_gate(input_rows, hidden_state)

            hidden_state = new_hidden_state
            cell_state = new_cell_state
            output_list.append(new_hidden_state)

            output_tensors = torch.stack(output_list, dim=1)

        return output_tensors

    def forget_gate(self, input, hidden_state):
        input_fg = self.linear_I_forget(input)
        hidden_fg = self.linear_H_forget(hidden_state)

        output_fg = self.sigmoid_forget(input_fg + hidden_fg)
        return output_fg

    def input_gate(self, input, hidden_state):
        input_ig = self.linear_I_input(input)
        hidden_ig = self.linear_H_input(hidden_state)

        output_ig = self.sigmoid_input(input_ig + hidden_ig)
        return output_ig

    def output_gate(self, input, hidden_state):
        input_og = self.linear_I_output(input)
        hidden_og = self.linear_H_output(hidden_state)

        output_og = self.sigmoid_output(input_og + hidden_og)
        return output_og

    def cell_memory(self, input, hidden_state):
        input_mem = self.linear_I_cell(input)
        hidden_mem = self.linear_H_cell(hidden_state)

        output_mem = self.tanh(input_mem + hidden_mem)
        return output_mem

    def init_states(self, batch_size):
        hidden_state = torch.randn(batch_size, self.hidden_dim)
        cell_state = torch.randn(batch_size, self.hidden_dim)

        return hidden_state, cell_state


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
