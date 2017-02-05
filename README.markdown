The relevant blog post is here: [http://hjweide.github.io/char-rnn](http://hjweide.github.io/char-rnn)

This implementation is largely based on [https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py](https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py).
See [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for a thorough explanation of how char-rnn works.

This implementation of char-rnn can be used to train on any text file.  My goal, however, was to train it on the entire history of my Facebook conversations.  If you have your own text file, you can skip to step 5 below.

1. Follow [these instructions](https://www.facebook.com/help/302796099745838) to get a copy of all your Facebook data.  You may want to do this first, because it can take a while for them to send you the download link.  When the download is complete, unzip the archive.

2. Clone and install this [Facebook chat parser]([Facebook chat parser](https://github.com/ownaginatious/fbchat-archive-parser)).
``` 
    git clone https://github.com/ownaginatious/fbchat-archive-parser     python setup.py develop
```

3. Run the parser on the ```messages.htm``` file from the extracted archive:
```
    fbcap html/messages.htm > messages.txt
```

4. Use this [snippet of code](https://gist.github.com/hjweide/2ef77fc69297daa5e00777c59c64161e) to strip out all messages not written by you.  Set the name appearing in your Facebook chats as the ```name``` variable, and run ```python parse_messages.py```.  You may need to write a more sophisticated parser if you want more control about which messages you want to extract, or if you had a name change, for example.

5. Set the ```text_fpath``` in ```train_char_rnn.py``` to the text file containing the training data.  If you used the snippet mentioned above, this will already be appropriately set as ```parsed.txt```.

6. Observe the sequences generated during training.  Once you are happy that the model has reached reasonable convergence, end the training with ```ctrl-c```.  

7. Set the ```text_fpath``` in ```generate_samples.py```, and run ```python generate_samples.py``` to continually supply phrases and samples from the model to amuse yourself.
