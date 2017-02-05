import char_rnn
import theano_funcs
import utils

from sklearn.preprocessing import LabelEncoder


def generate_samples():
    weights_fpath = 'weights.pickle'  # weights from which to initialize
    text_fpath = 'parsed.txt'  # training data text file, to build vocabulary

    grad_clipping = 100.
    num_hidden = 512
    train_seq_length, sample_seq_length = 20, 200
    text, vocab = utils.parse(text_fpath)

    # need to build the same encoder as during training, could pickle
    encoder = LabelEncoder()
    encoder.fit(list(vocab))
    vocab_size = len(vocab)

    layers = char_rnn.build_model(
        (None, train_seq_length, vocab_size),  # input_shape
        num_hidden, vocab_size, grad_clipping
    )

    print('loading model weights from %s' % (weights_fpath))
    char_rnn.load_weights(layers['l_out'], weights_fpath)

    print('compiling theano function for sampling')
    sample = theano_funcs.create_sample_func(layers)

    try:
        while True:
            # prompt the user for a phrase to initialize the sampling
            phrase = raw_input('start a phrase of at least %d chars:\n' % (
                train_seq_length)
            )
            if len(phrase) < train_seq_length:
                print('len(phrase) = %d, need len(phrase) >= %d' % (
                    len(phrase), train_seq_length)
                )
                continue
            generated_phrase = utils.sample(
                sample, phrase,
                train_seq_length, sample_seq_length,
                vocab_size, encoder
            )
            print('%s\n' % (generated_phrase))
    except KeyboardInterrupt:
        print('caught ctrl-c')
    print('done')


if __name__ == '__main__':
    generate_samples()
