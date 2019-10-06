import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque

class Reader(object):
    def __init__(self, batch_size, nb_sentences, nb_epochs,
                 tf_session, iterator):
        self.tf_session = tf_session
        self.batch_size = batch_size
        self.iterator = iterator
        self.pbar = tqdm(total=nb_sentences*nb_epochs)
        self.iter = 1

    def generate_batch(self):
        '''
        Generate a batch of padded sentences.
        Returns:
            x: batch of padded input sentences
            y: batch of padded output sentences
            shifted_y: y shifted to the right
        '''
        in_sentences = []
        out_sentences = []
        len_list_x = []
        len_list_y = []
        for _ in range(self.batch_size):
            in_sentence, out_sentence = self.tf_session.run(self.iterator)
            in_sentence = in_sentence.tolist()
            out_sentence = out_sentence.tolist()
            in_sentences.append(in_sentence)
            len_list_x.append(len(in_sentence))
            out_sentences.append(out_sentence)
            len_list_y.append(len(out_sentence))
            self.iter += 1
            self.pbar.update(1)
        # Padding
        max_len_list_x = max(len_list_x)
        max_len_list_y = max(len_list_y)
        x = np.array([in_sentences[i] + [-1] * \
            (max_len_list_x - len_list_x[i]) for i in range(self.batch_size)])
        y = np.array([out_sentences[i] + [-1] * \
            (max_len_list_y - len_list_y[i]) for i in range(self.batch_size)])
        shifted_y = np.array([out_sentences[i][:-1] + [-1] * \
            (max_len_list_y - len_list_y[i]) for i in range(self.batch_size)])

        return x, y, shifted_y
