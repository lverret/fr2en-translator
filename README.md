# French to English Translator with Transformers

This project is a TensorFlow implementation of a French to English neural machine translation system that makes use of the recently proposed [Transformer architecture](https://arxiv.org/abs/1706.03762). The repository also contains implementations of GRU and LSTM architectures for building seq2seq models.

## Requirements
* TensorFlow 1.x
* Pandas 0.25.x
* (optional) NLTK 3.4.5 (for computing bleu score)

## Data

For training, we used the [Europarl Corpus](https://www.statmt.org/europarl/) that consists of 2,218,201 english sentences and 2,190,579 french sentences.

### Preprocessing

We made use of the [Moses](https://github.com/moses-smt/mosesdecoder) library for tokenizing the sentences. Moreover, we lowercased the text and removed the empty sentences and their correspondences. The preprocessing script can be run with the following commands:
```
cd preprocessing
./preprocessed.sh
```
This creates a folder `../data` that contains pickle files of input and output sentences and source and target vocabularies of size respectively 20k and 10k.

## Training
The implementation favors simplicity and ease of use. A Transformer model can be trained in only a couple of lines of code:
```python
model = Transformer(in_voc=(id2word_fr, word2id_fr), out_voc=(id2word_en, word2id_en),
                    hidden_size=50, lr=1e-3, batch_size=128, beam_size=10, nb_epochs=5,
                    nb_heads=4, pos_enc=True, nb_layers=1)

model.fit(input_sentences, output_sentences)
model.save("./model/transformer.ckpt")
```
The full training script can be run with:
```
python3 train.py
```

## Translating
Once the model is trained, it can be used to translate a French sentence to its corresponding English sentence. We implemented the beam search algorithm for finding the most likely sequence. A trained model can be downloaded [here](https://drive.google.com/drive/folders/10g6mujHEVfwq5lCuNUKalFy8YoXUyTib?usp=sharing) (~4 hours on Google Colab).
```python
model = Transformer(in_voc=(id2word_fr, word2id_fr), out_voc=(id2word_en, word2id_en),
                    hidden_size=50, lr=1e-3, batch_size=128, beam_size=10, nb_epochs=5,
                    nb_heads=4, pos_enc=True, nb_layers=1)

model.load("./model/transformer.ckpt")
model.translate("jacques chirac est décédé à l&apos; âge de 86 ans .", nb_translations=5)
```
The snippet outputs:
```
Translation 1: "of course , jacques chirac died at the age of 86 ." with score -0.4987
Translation 2: "of course , jacques chirac died in the age of 86 ." with score -0.5307
Translation 3: "of jacques chirac died at the age of 86 ." with score -0.5564
Translation 4: "of course , jacques chirac died during the age of 86 ." with score -0.6169
Translation 5: "of course , jacques chirac died of the age of 86 ." with score -0.6329
```
Apparently, it is obvious for the machine.
