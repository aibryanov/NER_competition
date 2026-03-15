import numpy as np
import torch.nn as nn


def init_embedding(input_embedding):
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def init_linear(input_linear):
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    for layer_index in range(input_lstm.num_layers):
        for weight_name in (f"weight_ih_l{layer_index}", f"weight_hh_l{layer_index}"):
            weight = getattr(input_lstm, weight_name)
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

        if input_lstm.bias:
            for bias_name in (f"bias_ih_l{layer_index}", f"bias_hh_l{layer_index}"):
                bias = getattr(input_lstm, bias_name)
                bias.data.zero_()
                bias.data[input_lstm.hidden_size : 2 * input_lstm.hidden_size] = 1

    if not input_lstm.bidirectional:
        return

    for layer_index in range(input_lstm.num_layers):
        for weight_name in (f"weight_ih_l{layer_index}_reverse", f"weight_hh_l{layer_index}_reverse"):
            weight = getattr(input_lstm, weight_name)
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

        if input_lstm.bias:
            for bias_name in (f"bias_ih_l{layer_index}_reverse", f"bias_hh_l{layer_index}_reverse"):
                bias = getattr(input_lstm, bias_name)
                bias.data.zero_()
                bias.data[input_lstm.hidden_size : 2 * input_lstm.hidden_size] = 1
