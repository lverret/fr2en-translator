import numpy as np
import pickle
from models import GRU, LSTM, Transformer

# Load data
with open('./data/europarl/id2word_fr.pickle', 'rb') as f:
    id2word_fr = pickle.load(f)
with open('./data/europarl/input_sentences.pickle', 'rb') as f:
    input_sentences = pickle.load(f)
with open('./data/europarl/id2word_en.pickle', 'rb') as f:
    id2word_en = pickle.load(f)
with open('./data/europarl/output_sentences.pickle', 'rb') as f:
    output_sentences = pickle.load(f)
with open('./data/europarl/word2id_en.pickle', 'rb') as f:
    word2id_en = pickle.load(f)
with open('./data/europarl/word2id_fr.pickle', 'rb') as f:
    word2id_fr = pickle.load(f)

n = input_sentences.shape[0]
n_train = int(0.9*n)
perm = np.random.permutation(n)
train_in = input_sentences[perm[0:n_train]]
train_out = output_sentences[perm[0:n_train]]
val_in = input_sentences[perm[n_train:n]].values
val_out = output_sentences[perm[n_train:n]].values

model = Transformer(
    in_voc=(id2word_fr, word2id_fr), out_voc=(id2word_en, word2id_en),
    hidden_size=50, lr=1e-3, batch_size=128, beam_size=10, nb_epochs=10,
    nb_heads=4, pos_enc=True, nb_layers=1)

model.fit(input_sentences, output_sentences)
model.save("./model/transformer.ckpt")
