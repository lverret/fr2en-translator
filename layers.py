import tensorflow as tf

class Layer(object):
    '''
    Base layer class. Defines basic API for all layer objects.
    '''
    def _call(self, inputs):
        '''
        Defines the computational graph of the layer
        '''
        return inputs

    def __call__(self, **kwargs):
        '''
        Wrapper for _call()
        '''
        outputs = self._call(**kwargs)
        return outputs

class LayerNorm(Layer):
    '''
    Layer Normalization [1] layer.
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
    "Layer normalization."
    '''
    def __init__(self, hidden_size, dtype):
        self.weight = tf.Variable(tf.zeros(hidden_size), dtype=dtype)
        self.bias = tf.Variable(tf.ones(hidden_size), dtype=dtype)

    def _call(self, x):
        u = tf.reduce_mean(x, axis=-1, keepdims=True)
        s = tf.reduce_mean((x - u)**2, axis=-1, keepdims=True)
        x = (x - u) / tf.sqrt(s + 1e-8)
        return self.weight * x + self.bias

class EmbeddingLayer(Layer):
    '''
    Simple Embedding Layer.
    '''
    def __init__(self, input_size, output_size, name, dtype, pad_zero=False,
                 pos_enc=False):
        self.name = name
        self.output_size = output_size
        self.pos_enc = pos_enc
        self.pad_zero = pad_zero
        self.matrix = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(input_size, output_size),
                                stddev=0.01), name=self.name)

    def _call(self, ids):
        # Word embedding
        ids = tf.sign(ids + 1) * ids
        word_emb = tf.nn.embedding_lookup(self.matrix, ids)
        # Add a zero vector in front
        if self.pad_zero:
            zero = tf.zeros((tf.shape(word_emb)[0], 1, self.output_size))
            word_emb = tf.concat([zero, word_emb], axis=1)
        # Positional encoding
        if self.pos_enc:
            bs = tf.shape(word_emb)[0]
            T = tf.shape(word_emb)[1]
            power = tf.range(0.0, self.output_size, 2.0) / self.output_size
            freq = 1 / (10000 ** power)
            positions = tf.reshape(tf.range(0.0, T), (1, T))
            positions = tf.broadcast_to(positions, shape=(bs, T))
            P_e = tf.einsum('bt,d->btd', positions, freq)
            self.Pe = tf.concat([tf.sin(P_e), tf.cos(P_e)], axis=-1)
            word_emb += self.Pe
        return word_emb

class FeedForwardLayer(Layer):
    '''
    Position-wise affine transformation layer.
    '''
    def __init__(self, hidden_size, activation, dtype):
        self.hidden_size = hidden_size
        self.activation = activation
        self.W = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))
        self.b = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size,),
                                stddev=0.01))

    def _call(self, x):
        x = tf.tensordot(x, self.W, axes=[[2], [0]]) + self.b
        return x

class GRULayer(Layer):
    '''
    GRU Layer.
    '''
    def __init__(self, input_size, hidden_size, init, dtype):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.init = init

        # Weights for input vectors of shape (hidden_size, hidden_size)
        self.Wc = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.input_size, self.hidden_size),
                                stddev=0.01))
        self.Wu = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.input_size, self.hidden_size),
                                stddev=0.01))
        self.Wr = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.input_size, self.hidden_size),
                                stddev=0.01))

        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Uc = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))
        self.Uu = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))
        self.Ur = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))

        # Biases for hidden vectors of shape (hidden_size,)
        self.bc = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size,),
                                stddev=0.01))
        self.bu = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,),
                                stddev=0.01))
        self.br = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,),
                                stddev=0.01))

    def _forward_pass(self, c_tm1, x_t):
        '''
        Perform the forward pass of the GRU.

        Arguments
        ---------
        c_tm1: The hidden state at the previous timestep.
        x_t: The input vector at time t.
        '''
        # Definitions of gamma_u and gamma_r (update and forget gates)
        gamma_u = tf.sigmoid(tf.matmul(x_t, self.Wu) + \
                             tf.matmul(c_tm1, self.Uu) + self.bu)
        gamma_r = tf.sigmoid(tf.matmul(x_t, self.Wr) + \
                             tf.matmul(c_tm1, self.Ur) + self.br)

        # Definition of c_t~
        c_t_tilde = tf.tanh(tf.matmul(x_t, self.Wc) + \
                            tf.matmul(tf.multiply(gamma_r, c_tm1), self.Uc) + \
                            self.bc)

        # Compute the next hidden state
        c_t = tf.multiply(1 - gamma_u, c_tm1) + tf.multiply(gamma_u, c_t_tilde)

        return c_t

    def _call(self, h_t):
        h_t = tf.scan(self._forward_pass, h_t, initializer=self.init)
        return h_t # (T, bs, nb_lat)

class LSTMLayer(Layer):
    '''
    LSTM Layer.
    '''
    def __init__(self, input_size, hidden_size, init, dtype):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init = init
        # Weights for input vectors of shape (hidden_size, hidden_size)
        self.Wc = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.input_size, self.hidden_size),
                                stddev=0.01))
        self.Wu = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.input_size, self.hidden_size),
                                stddev=0.01))
        self.Wf = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.input_size, self.hidden_size),
                                stddev=0.01))
        self.Wo = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.input_size, self.hidden_size),
                                stddev=0.01))
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Uc = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))
        self.Uu = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))
        self.Uf = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))
        self.Uo = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size, self.hidden_size),
                                stddev=0.01))
        # Biases for hidden vectors of shape (hidden_size,)
        self.bc = tf.Variable(
            tf.truncated_normal(dtype=dtype,
                                shape=(self.hidden_size,),
                                stddev=0.01))
        self.bu = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,),
                                stddev=0.01))
        self.bf = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,),
                                stddev=0.01))
        self.bo = tf.Variable(
            tf.truncated_normal(dtype=dtype, shape=(self.hidden_size,),
                                stddev=0.01))

    def _forward_pass(self, a_tm1, x_t):
        '''
        Perform the forward pass of the LSTM.

        Arguments
        ---------
        a_tm1: The hidden state at the previous timestep.
        x_t: The input vector at time t.
        '''
        # Definitions of gamma_u, gamma_f and gamma_o
        gamma_u = tf.sigmoid(tf.matmul(x_t, self.Wu) + \
                             tf.matmul(a_tm1, self.Uu) + self.bu)
        gamma_f = tf.sigmoid(tf.matmul(x_t, self.Wf) + \
                             tf.matmul(a_tm1, self.Uf) + self.bf)
        gamma_o = tf.sigmoid(tf.matmul(x_t, self.Wo) + \
                             tf.matmul(a_tm1, self.Uo) + self.bo)
        # Definition of c_t~
        c_t_tilde = tf.tanh(tf.matmul(x_t, self.Wc) + \
                            tf.matmul(a_tm1, self.Uc) + \
                            self.bc)
        # Compute the next hidden state
        c_t = tf.multiply(gamma_u, c_t_tilde) + tf.multiply(gamma_f, a_tm1)
        a_t = tf.multiply(gamma_o, tf.tanh(c_t))
        return a_t

    def _call(self, h_t):
        h_t = tf.scan(self._forward_pass, h_t, initializer=self.init)
        return h_t # (T, bs, nb_lat)

class SelfAttentionLayer(Layer):
    '''
    Self-Attention Layer.
    '''
    def __init__(self, input_size, hidden_size, nb_heads, key_seq, value_seq,
                 causality, dtype):
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.nb_heads = nb_heads
        bs = tf.shape(key_seq)[0]
        Tk = tf.shape(key_seq)[1]
        Tv = tf.shape(value_seq)[1]
        # Parameter matrices
        self.V = tf.Variable(
            tf.truncated_normal(
                dtype=dtype, shape=(input_size, hidden_size, nb_heads),
                stddev=0.01))
        self.K = tf.Variable(
            tf.truncated_normal(
                dtype=dtype, shape=(input_size, hidden_size, nb_heads),
                stddev=0.01))
        self.Q = tf.Variable(
            tf.truncated_normal(
                dtype=dtype, shape=(input_size, hidden_size, nb_heads),
                stddev=0.01))
        # Padded value mask for keys
        mask_pad_key = - 1000 * (1 - tf.sign(key_seq + 1))
        mask_pad_key = tf.reshape(mask_pad_key, (-1, Tk, 1, 1))
        mask_pad_key = tf.cast(mask_pad_key, dtype)
        # Causality mask
        if causality:
            mask_caus = tf.ones(shape=(bs, Tk, Tv))
            mask_caus = - 1000 * (1 - tf.matrix_band_part(mask_caus, 0, -1))
            mask_caus = tf.reshape(mask_caus, (bs, Tk, Tv, 1))
            self.mask = mask_pad_key + mask_caus
        else:
            self.mask = mask_pad_key

    def _call(self, keys, queries, values):
        # Linear transformation
        Kh = tf.tensordot(keys, self.K, axes=[[2], [0]])
        Qh = tf.tensordot(queries, self.Q, axes=[[2], [0]])
        Vh = tf.tensordot(values, self.V, axes=[[2], [0]])
        # Compute scores
        attent_scores = tf.einsum('bidh,bjdh->bijh', Kh, Qh)
        attent_scores += self.mask
        attent_scores /= self.hidden_size ** 0.5
        # Compute weights
        attent_values = tf.nn.softmax(attent_scores, axis=1,
                                           name='attent_weights')
        # Weight with attention values
        h_t = tf.einsum('bijh,bidh->bjdh', attent_values, Vh)
        # Concat the heads
        h_t = tf.concat(tf.split(h_t, self.nb_heads, axis=-1), axis=2)
        return h_t[:, :, :, 0]
