from collections import Counter
import pandas as pd
import os
import pickle

keep_most_common_fr = 20000
keep_most_common_en = 10000
word2id_fr = {}
id2word_fr = {keep_most_common_fr: 'UNK', keep_most_common_fr + 1: 'EOS'}
freq_dict_fr = {}
word2id_en = {}
id2word_en = {keep_most_common_en: 'UNK', keep_most_common_en + 1: 'EOS'}
freq_dict_en = {}
filename_fr = "./wmt14_en_fr/tmp/train.tags.en-fr.tok.fr"
filename_en = "./wmt14_en_fr/tmp/train.tags.en-fr.tok.en"

print("counting fr")

with open(filename_fr) as f:
    for k, line in enumerate(f):
        split = line.lower().split()
        for word in split:
            if word not in freq_dict_fr:
                freq_dict_fr[word] = 1
            else:
                freq_dict_fr[word] += 1


print("counting en")

with open(filename_en) as f:
    for k, line in enumerate(f):
        split = line.lower().split()
        for word in split:
            if word not in freq_dict_en:
                freq_dict_en[word] = 1
            else:
                freq_dict_en[word] += 1

k = Counter(freq_dict_fr)
high_fr = dict(k.most_common(keep_most_common_fr))
k = Counter(freq_dict_en)
high_en = dict(k.most_common(keep_most_common_en))

sentences = []
to_remove = []

print("to ids fr")

with open(filename_fr) as f:
    id = 0
    for k, line in enumerate(f):
        split = line.lower().split()
        # remove too long and too short sentences
        if len(split) > 50 or len(split) < 3 or len(split) > 0 and split[0] == '(':
            to_remove.append(k)
            continue
        ids = []
        for word in split:
            if word in high_fr:
                if word not in word2id_fr:
                    word2id_fr[word] = id
                    id2word_fr[id] = word
                    id += 1
                ids.append(word2id_fr[word])
            else:
                ids.append(keep_most_common_fr)
        ids.append(keep_most_common_fr + 1)
        sentences.append(ids)

input_sentences = pd.Series(sentences)

print("to ids en")
sentences = []
i = 0
next_to_remove = to_remove[i]

with open(filename_en) as f:
    id = 0
    for k, line in enumerate(f):
        if k == next_to_remove:
            i += 1
            if i < len(to_remove):
                next_to_remove = to_remove[i]
            continue
        split = line.lower().split()
        ids = []
        for word in split:
            if word in high_en:
                if word not in word2id_en:
                    word2id_en[word] = id
                    id2word_en[id] = word
                    id += 1
                ids.append(word2id_en[word])
            else:
                ids.append(keep_most_common_en)
        ids.append(keep_most_common_en + 1)
        sentences.append(ids)

output_sentences = pd.Series(sentences)

if not os.path.exists("../data/europarl"):
    os.makedirs("../data/europarl")

with open('../data/europarl/word2id_fr.pickle', 'wb') as f:
    pickle.dump(word2id_fr, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/europarl/id2word_fr.pickle', 'wb') as f:
    pickle.dump(id2word_fr, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/europarl/word2id_en.pickle', 'wb') as f:
    pickle.dump(word2id_en, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('../data/europarl/id2word_en.pickle', 'wb') as f:
    pickle.dump(id2word_en, f, protocol=pickle.HIGHEST_PROTOCOL)

input_sentences.to_pickle("../data/europarl/input_sentences.pickle")
output_sentences.to_pickle("../data/europarl/output_sentences.pickle")
