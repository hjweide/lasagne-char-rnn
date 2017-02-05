import char_rnn
import theano_funcs
import utils

import numpy as np
from lasagne.layers import get_all_param_values
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def train_char_rnn():
    weights_fpath = 'weights2.pickle'  # weights will be stored here
    text_fpath = 'parsed2.txt'  # path to the input file
    max_epochs = 1000
    lr = 0.01
    grad_clipping = 100.
    num_hidden = 512
    batch_size = 128
    sample_every = 1000  # sample every n batches
    # sequence length during training, number of chars to draw for sampling
    train_seq_length, sample_seq_length = 20, 200
    text, vocab = utils.parse(text_fpath)

    # encode each character in the vocabulary as an integer
    encoder = LabelEncoder()
    encoder.fit(list(vocab))
    vocab_size = len(vocab)

    layers = char_rnn.build_model(
        (None, train_seq_length, vocab_size),  # input_shape
        num_hidden, vocab_size, grad_clipping
    )

    # optionally load a pre-trained model
    #print('loading model weights from %s' % (weights_fpath))
    #char_rnn.load_weights(layers['l_out'], weights_fpath)

    # phrases to use during sampling
    phrases = ['I should go to bed now']

    print('compiling theano function for training')
    train = theano_funcs.create_train_func(layers, lr=lr)
    print('compiling theano function for sampling')
    sample = theano_funcs.create_sample_func(layers)

    try:
        for epoch in range(1, 1 + max_epochs):
            print('epoch %d' % (epoch))

            # sample from the model and update the weights
            train_losses = []
            seq_iter = utils.sequences(
                text, batch_size, train_seq_length, vocab_size, encoder
            )
            for i, (X, y) in tqdm(enumerate(seq_iter), leave=False):
                if X is not None and y is not None:
                    loss = train(X, y)
                    train_losses.append(loss)

                # continuously sample from the model
                if ((i + 1) % sample_every) == 0:
                    print(' loss = %.6f' % (np.mean(train_losses)))
                    phrase = np.random.choice(phrases)
                    generated_phrase = utils.sample(
                        sample, phrase,
                        train_seq_length, sample_seq_length,
                        vocab_size, encoder
                    )
                    print('%s%s' % (phrase, generated_phrase))

    except KeyboardInterrupt:
        print('caught ctrl-c, stopping training')

    # write the weights to disk so we can try out the model
    print('saving weights to %s' % (weights_fpath))
    weights = get_all_param_values(layers['l_out'])
    char_rnn.save_weights(weights, weights_fpath)
    print('done')


if __name__ == '__main__':
    train_char_rnn()
