import theano
import theano.tensor as T
from lasagne.layers import get_output
from lasagne.layers import get_all_params
from lasagne.updates import adagrad
from theano.tensor.nnet import categorical_crossentropy


def create_train_func(layers, lr=0.01):
    # dims: batch, sequence, vocabulary
    X = T.tensor3('X')
    X_batch = T.tensor3('X_batch')

    # dims: target
    y = T.ivector('y')
    y_batch = T.ivector('y_batch')

    y_hat = get_output(layers['l_out'], X, deterministic=False)

    train_loss = T.mean(categorical_crossentropy(y_hat, y), axis=0)
    params = get_all_params(layers['l_out'], trainable=True)

    updates = adagrad(train_loss, params, lr)

    train_func = theano.function(
        inputs=[theano.In(X_batch), theano.In(y_batch)],
        outputs=train_loss,
        updates=updates,
        givens={
            X: X_batch,
            y: y_batch,
        },
    )

    return train_func


def create_sample_func(layers):
    X = T.tensor3('X')
    X_batch = T.tensor3('X_batch')

    y_hat = get_output(layers['l_out'], X, deterministic=True)

    sample_func = theano.function(
        inputs=[theano.In(X_batch)],
        outputs=y_hat,
        updates=None,
        givens={
            X: X_batch,
        },
    )

    return sample_func


def test_create_train_func():
    import numpy as np
    from char_rnn import build_model
    batch_size, sequence_length, vocab_size = 16, 32, 64

    layers = build_model((None, None, vocab_size), 128, vocab_size, 10.)
    train_func = create_train_func(layers)

    X = np.zeros((batch_size, sequence_length, vocab_size), dtype=np.float32)
    X[:, :, 0] = 1.
    y = np.random.randint(0, vocab_size, batch_size).astype(np.int32)
    print('testing train_func')
    loss = train_func(X, y)
    print('loss = %.6f' % (loss))
    print('done')


if __name__ == '__main__':
    test_create_train_func()
