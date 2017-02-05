import numpy as np


def parse(fpath):
    with open(fpath, 'r') as f:
        text = f.read()

    vocab = set(text)

    return text, vocab


# build the batches for training: X[i] --> y[i]
# ['H' 'e' 'l' 'l' 'o' ' ' 'd' 'a' 'r'] --> 'k'
# ['e' 'l' 'l' 'o' ' ' 'd' 'a' 'r' 'k'] --> 'n'
# ['l' 'l' 'o' ' ' 'd' 'a' 'r' 'k' 'n'] --> 'e', and so on
def sequences(text, batch_size, sequence_length, vocab_size, encoder):
    for i in range(0, len(text), batch_size):
        substring = text[i:i + batch_size + sequence_length]
        # when we run out of text, reduce the batch size
        if i + batch_size + sequence_length >= len(text):
            effective_batch_size = len(text) - sequence_length - i
        else:
            effective_batch_size = batch_size

        if effective_batch_size <= 0:
            X, y = None, None
        else:
            X = np.zeros(
                (effective_batch_size, sequence_length, vocab_size),
                dtype=np.float32
            )
            y = np.zeros(effective_batch_size, dtype=np.int32)

        if X is not None and y is not None:
            for j in range(effective_batch_size):
                sequence = substring[j:j + sequence_length]
                target = substring[j + sequence_length]
                tokens = encoder.transform([s for s in sequence])
                # convert from sequence of tokens to one-hot encoding
                X[j, np.arange(tokens.shape[0]), tokens] = 1.
                y[j] = encoder.transform(target)

        yield X, y


def sample(infer, text, sequence_length, num_samples, vocab_size, encoder):
    substring = text[len(text) - sequence_length:]
    X = np.zeros((1, sequence_length, vocab_size), dtype=np.float32)
    tokens = encoder.transform([s for s in substring])
    # initialize X as the last characters of the given phrase
    X[0, np.arange(tokens.shape[0]), tokens] = 1.
    samples = []
    for i in range(num_samples):
        probs = infer(X)
        # move all characters one place back and insert the newly sampled char
        X = np.roll(X, -1, axis=1)
        tokens = probs.argmax(axis=1)
        #tokens = np.random.choice(np.arange(vocab_size), p=probs.ravel())
        X[:, -1] = np.zeros(vocab_size)
        X[:, -1, tokens] = 1.

        # keep track of the sampled chars for display
        samples.append(encoder.inverse_transform(tokens[0]))

    return ''.join(samples)


def test_sequences():
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    text, vocab = parse('parsed.txt')
    encoder.fit(list(vocab))
    seq_iter = sequences(text, 128, 200, len(vocab), encoder)
    for (X, y) in seq_iter:
        if X is not None and y is not None:
            for i in range(X.shape[0]):
                print('X = %s, y = %s' % (
                    encoder.inverse_transform(X[i].argmax(axis=1)),
                    encoder.inverse_transform(int(y[i]))
                ))


if __name__ == '__main__':
    test_sequences()
