import cPickle as pickle
from lasagne.init import Normal
from lasagne.layers import InputLayer
from lasagne.layers import LSTMLayer
from lasagne.layers import DenseLayer
from lasagne.layers import get_all_layers
from lasagne.layers import get_all_params
from lasagne.nonlinearities import tanh, softmax


def save_weights(weights, filename):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_weights(layer, filename):
    with open(filename, 'rb') as f:
        src_params_list = pickle.load(f)

    dst_params_list = get_all_params(layer)
    # assign the parameter values stored on disk to the model
    for src_params, dst_params in zip(src_params_list, dst_params_list):
        dst_params.set_value(src_params)


def build_model(input_shape, num_hidden, num_output, grad_clipping):
    l_in = InputLayer(input_shape, name='l_in')
    l_lstm1 = LSTMLayer(
        l_in, name='l_lstm1',
        num_units=num_hidden, grad_clipping=grad_clipping, nonlinearity=tanh,
    )
    l_lstm2 = LSTMLayer(
        l_lstm1, name='l_lstm2',
        num_units=num_hidden, grad_clipping=grad_clipping, nonlinearity=tanh,
        only_return_final=True,
    )

    l_out = DenseLayer(
        l_lstm2, name='l_out', W=Normal(),
        num_units=num_output, nonlinearity=softmax
    )

    layers = get_all_layers(l_out)
    return {layer.name: layer for layer in layers}


if __name__ == '__main__':
    print('testing build_model')
    build_model((None, 32, 64), 128, 10, 10.)
    print('done')
