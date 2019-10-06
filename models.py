import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import operator
import layers
from data import Reader

class Model(object):
    '''
    Generic class for building neural seq2seq model for translation task.

    Arguments
    ---------
    in_voc: Tuple of dictionnaries (id2word and word2id) of the source
    language.
    out_voc: Tuple of dictionnaries (id2word and word2id) of the target
    language.
    hidden_size: Size of the word embeddings.
    drop_p: Dropout rate (between 0 and 1).
    lr: Learning rate.
    batch_size: Size of the batches.
    nb_epochs: Number of epochs.
    beam_size: Beam size.
    nb_layers: Number of layers.
    dtype: Type for the parameter matrices.
    name: Path to folder where the model will be saved every epoch.
    '''
    def __init__(self, in_voc, out_voc, hidden_size, drop_p, lr, batch_size,
                 nb_epochs, beam_size, nb_layers, dtype, folder):
        self.in_id2word = in_voc[0]
        self.out_id2word = out_voc[0]
        self.in_word2id = in_voc[1]
        self.out_word2id = out_voc[1]
        self.input_dim = len(self.in_id2word)
        self.output_dim = len(self.out_id2word)
        self.hidden_size = hidden_size
        self.drop_p = drop_p
        self.lr = lr
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.beam_size = beam_size
        self.drop_p = drop_p
        self.nb_layers = nb_layers
        self.dtype = dtype
        self.folder = folder
        self._define_placeholders()

    def fit(self, train_data, target_data, valid_data=None):
        '''
        Fit a seq2seq model on pairs of sequences of source and target
        languages.

        Arguments
        ---------
        train_data: pd.Series of sentences of the source language.
        target_data: pd.Series of sentences of the target language.
        valid_data: tuple of pd.Series input and output sentences.
        '''
        # Construct a tf.Dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: zip(train_data, target_data),
            (tf.int32, tf.int32), output_shapes=(None))
        dataset = dataset.repeat(self.nb_epochs)
        dataset = dataset.shuffle(buffer_size=10000)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        # Init
        c_train = np.zeros((self.batch_size, self.hidden_size))
        epoch = 1
        avg_train_loss = 0
        nb_batch = 0
        # Sentence reader
        reader = Reader(batch_size=self.batch_size,
                        nb_sentences=train_data.shape[0],
                        nb_epochs=self.nb_epochs,
                        tf_session=self.session,
                        iterator=next_element)
        while True:
            try:
                x, y, shifted_y = reader.generate_batch()
                fd = {self.input: x,
                      self.gt: y,
                      self.shifted_gt: shifted_y,
                      self.c_t: c_train}
                # Run backprop
                _, train_loss = self.session.run([self.optimizer,
                                                  self.avg_loss],
                                                 feed_dict=fd)
                avg_train_loss += train_loss
                nb_batch += 1
                # Print
                if reader.iter >= epoch * train_data.shape[0] - self.batch_size:
                    if self.folder is not None:
                        self.save(self.folder + "epoch" + str(epoch) + "/" + \
                                  str(self.model_name) + ".ckpt")
                    reader.pbar.write("Epoch %d" %epoch)
                    reader.pbar.write(
                        " Training loss: %.6f" %(avg_train_loss / nb_batch))
                    if valid_data is not None:
                        bleu = self.evaluate(valid_data)
                        reader.pbar.write(" Validation bleu score: %.6f" %bleu)
                    epoch += 1
                    avg_train_loss = 0
                    nb_batch = 0
            except tf.errors.OutOfRangeError:
                reader.pbar.close()
                break

    def translate(self, sentence, nb_translations=1):
        '''
        Implement the beam search algorithm for finding the most likely
        translation sentence.

        Arguments
        ---------
        sentence: sentence to translate (string) or list of encoded numerical
        tokens (list)
        '''
        to_encode = type(sentence) is str
        if to_encode:
            enc_sent = []
            for word in sentence.split():
                if word in self.in_word2id:
                    enc_sent.append(self.in_word2id[word])
                else:
                    enc_sent.append(self.input_dim - 2)
            enc_sent.append(self.input_dim - 1)
        else:
            enc_sent = sentence
        # Feed dict
        fd = {self.c_t: np.zeros((1, self.hidden_size)),
              self.input: np.array(enc_sent).reshape(1, -1),
              self.shifted_gt: np.zeros((1, 1)), # not used
              self.gt: np.zeros((1, 1)), # not used
              self.k_value: self.beam_size,
              self.test: True}
        # Feed a dummy word embedding (vector of zeros) for generating the
        # first token of the translation sentence
        top_k, proba = self.session.run([self.top_k,
                                         self.proba],
                                        feed_dict=fd)
        new_input = top_k[0, -2, :]
        best_probas = np.log(proba[0, -2, new_input])
        best_seqs = [[new_input[i]] for i in range(self.beam_size)]
        # Generate the sequence
        eos_token = self.output_dim - 1
        max_len = 2 * len(enc_sent)
        len_tr = 1
        to_keep = []
        while len(to_keep) < nb_translations and len_tr < max_len:
            all_seq = []
            all_proba = []
            for i in range(len(best_seqs)):
                best_seq = best_seqs[i]
                best_proba = best_probas[i]
                fd[self.shifted_gt] = np.array(best_seq).reshape(1, -1)
                top_k, proba = self.session.run([self.top_k,
                                                 self.proba],
                                                feed_dict=fd)
                new_input = top_k[0, -1, :]
                proba = np.log(proba[0, -1, new_input])
                # Append beam size best tokens for each pre-sequence
                for j in range(self.beam_size):
                    if best_seq[-1] == eos_token:
                         proba[j] = -1000
                    all_seq.append(best_seq + [new_input[j]])
                    norm_prob = (best_proba * len_tr + proba[j]) / (len_tr + 1)
                    all_proba.append(norm_prob)
            # Keep the "beam size" best ones
            ind = np.argsort(all_proba)[-self.beam_size:][::-1]
            best_seqs = [all_seq[indi] for indi in ind]
            best_probas = [all_proba[indi] for indi in ind]
            len_tr += 1
            # Append the best first sentence that ends with EOS token
            while best_seqs[0][-1] == eos_token and len(best_seqs[0]) > 0 \
                  and len(to_keep) < nb_translations:
                to_keep.append((best_probas.pop(0), best_seqs.pop(0)))
        # Keep the best ones
        translations = []
        for k, (p, translation) in \
            enumerate(sorted(to_keep, key=operator.itemgetter(0),
                      reverse=True)):
            if to_encode:
                translation = translation[:-1] # remove <EOS> token
                translation = ' '.join(
                    [self.out_id2word[id] for id in translation])
            translations.append(translation)
            print("Translation %d: \"%s\" with score %f" %(k+1, translation, p))
        if to_encode:
            return translations
        else:
            return translations[0]

    def evaluate(self, valid_data):
        '''
        Compute the bleu score on validation data.

        Arguments
        ---------
        valid_data: tuple of pd.Series input and output sentences.
        '''
        import nltk # for bleu score
        input_valid_sentences, output_valid_sentences = valid_data
        tot_bleu = 0
        nb_valid_sentences = input_valid_sentences.shape[0]
        for k in tqdm(range(nb_valid_sentences)):
            valid_in_sent = input_valid_sentences[k]
            valid_out_sent = output_valid_sentences[k]
            pred = self.translate(valid_in_sent)
            bleu = nltk.translate.bleu_score.sentence_bleu(
                [valid_out_sent], pred, weights=[1/3, 1/3, 1/3])
            tot_bleu += bleu
        return tot_bleu / nb_valid_sentences

    def save(self, model_path):
        '''
        Store a TensorFlow graph.

        Arguments
        ---------
        model_path: Name of the file where the graph will be stored.
        '''
        self.saver.save(self.session, model_path)
        print("Model saved in path: %s" % model_path)

    def load(self, model_path):
        '''
        Load a TensorFlow graph.

        Arguments
        ---------
        model_path: Name of the file where the graph is stored.
        '''
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        meta_file = model_path + '.meta'
        self.saver = tf.train.import_meta_graph(meta_file)
        model_dir = os.path.dirname(meta_file) + '/'
        self.saver.restore(session, tf.train.latest_checkpoint(model_dir))
        self.session = session
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name("input:0")
        self.output = graph.get_tensor_by_name("output:0")
        self.gt = graph.get_tensor_by_name("gt:0")
        self.shifted_gt = graph.get_tensor_by_name("shifted_gt:0")
        self.test = graph.get_tensor_by_name("mode:0")
        self.k_value =  graph.get_tensor_by_name("k_value:0")
        self.c_t = graph.get_tensor_by_name("c_t:0")
        self.avg_loss = graph.get_tensor_by_name("loss:0")
        self.top_k =  graph.get_tensor_by_name("top_k:0")
        self.optimizer = tf.get_collection("optimizer")[0]


    def _define_placeholders(self):
        '''
        Define the inputs of the graph.
        '''
        # Define the input indices placeholder (batch_size, Ti)
        self.input = tf.placeholder(dtype=tf.int32,
                                    shape=(None, None),
                                    name='input')
        # Define the actual output indices placeholder (batch_size, To)
        self.gt = tf.placeholder(dtype=tf.int32,
                                 shape=(None, None),
                                 name='gt')
        # Same as gt but shifted to the right (batch_size, To-1)
        self.shifted_gt = tf.placeholder(dtype=tf.int32,
                                         shape=(None, None),
                                         name='shifted_gt')
        # Initializer for RNN (batch_size, d)
        self.c_t = tf.placeholder(dtype=tf.float32,
                                  shape=(None, self.hidden_size),
                                  name='c_t')
        # Test boolean placeholder
        self.test = tf.placeholder_with_default(input=False,
                                                shape=(),
                                                name='mode')
        # Top k words for beam search
        self.k_value = tf.placeholder_with_default(input=20,
                                                   shape=(),
                                                   name='k_value')

    def _compute_scores(self, h_t, W_out):
        '''
        Compute a score for each word.

        Arguments
        ---------
        h_t: Vector representation of a sequence (size (batch size, T)).
        W_out: Output embedding layer (type layers.EmbeddingLayer).
        '''
        output = tf.einsum('btd,nd->btn', h_t, W_out.matrix)
        self.output = tf.identity(output, name='output')

    def _compute_loss(self):
        '''
        Compute the loss based on the computed output scores.
        '''
        # Top k scores
        _, top_k = tf.math.top_k(self.output, k=self.k_value)
        self.top_k = tf.identity(top_k, name='top_k')
        # Proba
        self.proba = tf.nn.softmax(self.output, axis=-1)
        # Compute loss
        self.labels = tf.one_hot(self.gt, self.output_dim,
                            on_value=1.0, off_value=0.0, axis=-1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.labels, logits=self.output)
        # Mask the padded values
        mask = tf.cast(tf.sign(self.gt + 1), self.dtype)
        loss *= mask
        non_zero = tf.reduce_sum(mask, axis=1)
        loss = tf.reduce_sum(loss, axis=1) / non_zero
        self.avg_loss = tf.reduce_mean(loss, name='loss')

    def _define_optimizer(self):
        '''
        Define the optimizer.
        '''
        # Use the Adam optimizer for training
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) \
                                 .minimize(self.avg_loss)
        # Add optimizer state to collection
        tf.add_to_collection("optimizer", self.optimizer)

    def _init_session(self):
        '''
        Initialize the TensorFlow session.
        '''
        # Define the session
        self.session = tf.Session(
                            config=tf.ConfigProto(allow_soft_placement=True))
        # Initialize all the variables
        init_variables = tf.global_variables_initializer()
        self.session.run(init_variables)
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def _dropout(self, h_t):
        '''
        Dropout layer.
        '''
        # Dropout for training
        h_t = tf.cond(tf.equal(self.test, False),
                      lambda: tf.nn.dropout(h_t, 1 - self.drop_p),
                      lambda: h_t)
        return h_t

        def _build(self):
            raise NotImplementedError

class Transformer(Model):
    '''
    Class that implements the Transformer architecture [1].
    [1] Vaswani, Ashish, et al. "Attention is all you need."

    Arguments
    ---------
    causality: boolean that tells the model to mask or not the subsequent
    words of the current word in the sequence.
    pos_enc: boolean that tells to add positional encodings of the words or not.
    nb_heads: Number of heads for multi-head attention layers.
    '''
    def __init__(self, in_voc, out_voc, hidden_size=100, drop_p=0, lr=1e-3,
                 batch_size=256, nb_epochs=5, beam_size=3, nb_layers=1,
                 causality=True, pos_enc=False, nb_heads=1, dtype=tf.float32,
                 name=None):
        super(Transformer, self).__init__(
            in_voc, out_voc, hidden_size, drop_p, lr, batch_size,
            nb_epochs, beam_size, nb_layers, dtype, name)
        self.causality = causality
        self.pos_enc = pos_enc
        self.nb_heads = nb_heads
        self.model_name = 'transformer'
        self._build()
        self._compute_loss()
        self._define_optimizer()
        self._init_session()

    def _build(self):
        # Input embedding
        h_enc = layers.EmbeddingLayer(
            input_size=self.input_dim,
            output_size=self.nb_heads*self.hidden_size, name='W_enc',
            dtype=self.dtype, pos_enc=self.pos_enc)(
                ids=self.input)
        # Dropout
        h_enc = self._dropout(h_enc)
        # Encoder
        encoder = [h_enc]
        for _ in range(self.nb_layers):
            # Multi-Head Attention
            encoder.append(self._add_multi_head_attention_layer(
                keys=encoder[-1], queries=encoder[-1], values=encoder[-1],
                key_seq=self.input, value_seq=self.input))
        # Output embedding
        h_dec = layers.EmbeddingLayer(
            input_size=self.output_dim,
            output_size=self.nb_heads*self.hidden_size, name='W_dec',
            dtype=self.dtype, pad_zero=True, pos_enc=self.pos_enc)(
                ids=self.shifted_gt)
        # Dropout
        h_dec = self._dropout(h_dec)
        # Decoder
        decoder = [h_dec]
        for k in range(self.nb_layers):
            # Self Attention
            h_dec = layers.SelfAttentionLayer(
                input_size=self.nb_heads*self.hidden_size,
                hidden_size=self.hidden_size, key_seq=self.gt,
                value_seq=self.gt, nb_heads=self.nb_heads, causality=True,
                dtype=self.dtype)(
                    keys=decoder[-1], queries=decoder[-1], values=decoder[-1])
            # Dropout
            h_dec = self._dropout(h_dec)
            # Add & Norm
            h_dec = layers.LayerNorm(
                hidden_size=self.nb_heads*self.hidden_size, dtype=self.dtype)(
                    x=h_dec+decoder[-1])
            # Multi-Head Attention
            decoder.append(self._add_multi_head_attention_layer(
                keys=encoder[k], queries=h_dec, values=encoder[k],
                key_seq=self.input, value_seq=self.gt))
        # Output linear transformation layer
        W_out = layers.EmbeddingLayer(
            input_size=self.output_dim,
            output_size=self.nb_heads*self.hidden_size,
            name='W_out', dtype=self.dtype)
        self._compute_scores(decoder[-1], W_out)

    def _add_multi_head_attention_layer(self, keys, queries, values, key_seq,
                                        value_seq):
        # Multi-Head Attention
        x = layers.SelfAttentionLayer(
            input_size=self.nb_heads*self.hidden_size,
            hidden_size=self.hidden_size, key_seq=key_seq, value_seq=value_seq,
            nb_heads=self.nb_heads, causality=False, dtype=self.dtype)(
                keys=keys, queries=queries, values=values)
       	# Dropout
        x = self._dropout(x)
        # Add & Norm
        h = layers.LayerNorm(
            hidden_size=self.nb_heads*self.hidden_size, dtype=self.dtype)(
                x=x+queries)
        # 2-layer Feed Forward
        x = layers.FeedForwardLayer(hidden_size=self.nb_heads*self.hidden_size,
            activation=tf.nn.relu, dtype=self.dtype)(x=h)
        x = layers.FeedForwardLayer(hidden_size=self.nb_heads*self.hidden_size,
            activation=lambda x: x, dtype=self.dtype)(x=x)
       	# Dropout
        x = self._dropout(x)
        # Add & Norm
        x = layers.LayerNorm(
            hidden_size=self.nb_heads*self.hidden_size, dtype=self.dtype)(x=x+h)
        return x

class RNN(Model):
    '''
    Generic class that allows the implementation of two types of
    Recurrent Neural Network (RNN) model: Gated Recurrent Unit (GRU) [1] or
    Long-Short Term Memory (LSTM) [2].
    [1] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory."
    [2] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
    "Neural machine translation by jointly learning to align and translate."
    '''
    def __init__(self, rnn_layer, in_voc, out_voc, hidden_size=100, drop_p=0,
                 lr=1e-3, batch_size=256, nb_epochs=5, beam_size=3, nb_layers=1,
                 dtype=tf.float32, name=None):
        super(RNN, self).__init__(
            in_voc, out_voc, hidden_size, drop_p, lr, batch_size, nb_epochs,
            beam_size, nb_layers, dtype, name)
        self.rnn_layer = rnn_layer
        self._build()
        self._compute_loss()
        self._define_optimizer()
        self._init_session()

    def _compute_last_hidden_representation(self, h_t):
        '''
        The sentences in a same batch not having the same length, it is
        necessary to select the indices for retrieving the representation vector
        corresponding to the last timestep of each sentence.

        Arguments
        ---------
        h_t: Vector representation of a sequence (size (batch size, T)).
        '''
        mask = tf.sign(self.input + 1)
        sum_mask = tf.reduce_sum(mask, axis=1) - 1
        dim_lat = tf.shape(h_t)[2]
        bs = tf.shape(self.input)[0]
        indices_1 = tf.broadcast_to(tf.reshape(tf.range(bs), (-1, 1)),
                                    shape=(bs, dim_lat))
        indices_2 = tf.broadcast_to(tf.reshape(sum_mask, (-1, 1)),
                                    shape=(bs, dim_lat))
        indices_3 = tf.broadcast_to(tf.reshape(tf.range(dim_lat), (1, -1)),
                                    shape=(bs, dim_lat))
        indices = tf.transpose((indices_1, indices_2, indices_3), [1, 2, 0])
        # Select indices
        last_h = tf.gather_nd(tf.transpose(h_t, [1, 0, 2]), indices)
        return last_h

    def _build(self):
        # Input embedding
        h_enc = layers.EmbeddingLayer(
            input_size=self.input_dim, output_size=self.hidden_size,
            name='W_enc', dtype=self.dtype)(ids=self.input)
        # Encoder part
        encoder = tf.transpose(h_enc, [1, 0, 2]) # (T, bs, nb_lat)
        last_h = []
        for k in range(self.nb_layers):
            encoder = self.rnn_layer(input_size=self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     init=self.c_t,
                                     dtype=self.dtype)(h_t=encoder)
            encoder = self._dropout(encoder)
            last_h.append(self._compute_last_hidden_representation(encoder))
        # Output embedding
        h_dec = layers.EmbeddingLayer(
            input_size=self.output_dim, output_size=self.hidden_size,
            name='W_dec', dtype=self.dtype, pad_zero=True)(ids=self.shifted_gt)
        # Decoder part
        decoder = tf.transpose(h_dec, [1, 0, 2]) # (T, bs, nb_lat)
        for k in range(self.nb_layers):
            decoder = self.rnn_layer(input_size=self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     init=last_h[k],
                                     dtype=self.dtype)(h_t=decoder)
            decoder = self._dropout(decoder)
        decoder = tf.transpose(decoder, [1, 0, 2]) # (bs, T, nb_lat)
        # Output linear transformation layer
        W_out = layers.EmbeddingLayer(
            input_size=self.output_dim, output_size=self.hidden_size,
            name='W_out', dtype=self.dtype)
        self._compute_scores(decoder, W_out)

class GRU(RNN):
    '''
    GRU model.
    '''
    def __init__(self, in_voc, out_voc, hidden_size=100, drop_p=0, lr=1e-3,
                 batch_size=256, nb_epochs=5, beam_size=3, nb_layers=1,
                 dtype=tf.float32, name=None):
        super(GRU, self).__init__(
            layers.GRULayer, in_voc, out_voc, hidden_size, drop_p, lr,
            batch_size, nb_epochs, beam_size, nb_layers, dtype, name)
        self.model_name = "gru"

class LSTM(RNN):
    '''
    LSTM model.
    '''
    def __init__(self, in_voc, out_voc, hidden_size=100, drop_p=0, lr=1e-3,
                 batch_size=256, nb_epochs=5, beam_size=3, nb_layers=1,
                 dtype=tf.float32, name=None):
        super(LSTM, self).__init__(
            layers.LSTMLayer, in_voc, out_voc, hidden_size, drop_p, lr,
            batch_size, nb_epochs, beam_size, nb_layers, dtype, name)
        self.model_name = "lstm"
