"""
    Author: <mailto:purpuraa@dei.unipd.it Alberto Purpura, stefano.marchesin@dei.unipd.it Stefano Marchesin>
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import tensorflow as tf


class NVSM(object):
    """build the graph for NVSM model"""

    def __init__(self, _word_vocab_size, _corpus_size, _options):
        self.word_vocab_size = _word_vocab_size
        self.corpus_size = _corpus_size
        self.options = _options

        """NETWORK INITIALIZATION"""
        opts = self.options
        self.word_embs = tf.get_variable('word_embs', shape=[self.word_vocab_size, opts.word_size],
                                         initializer=tf.glorot_uniform_initializer(seed=opts.seed), trainable=True)
        #with tf.device('/cpu:0'):
        self.doc_embs = tf.get_variable('doc_embs', shape=[self.corpus_size, opts.doc_size],
                                        initializer=tf.glorot_uniform_initializer(seed=opts.seed), trainable=True)
        self.proj_weights = tf.get_variable('proj_weights', shape=[opts.doc_size, opts.word_size],
                                            initializer=tf.glorot_uniform_initializer(seed=opts.seed), trainable=True)
        # create placeholders
        self.ngram_words = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.ngram_size])
        self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size])
        self.negative_labels = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.neg_samples])
        # embedding lookups
        self.words = tf.nn.embedding_lookup(self.word_embs, self.ngram_words)
        self.true_docs = tf.nn.embedding_lookup(self.doc_embs, self.labels)
        self.negative_docs = tf.nn.embedding_lookup(self.doc_embs, self.negative_labels)

        """FORWARD PASS"""
        self.proj_ngrams = self.ngrams2docs(self.words, opts.l2_norm)
        self.stand_ngrams = self.standardize_batch(self.proj_ngrams)
        # true logits [batch_size]
        self.true_logits = self.compute_true_logits(self.stand_ngrams, self.true_docs)
        # negative logits [batch_size, neg_samples]
        self.neg_logits = self.compute_negative_logits(self.stand_ngrams, self.negative_docs)

        """LOSS OPERATION"""
        self.text_loss = self.text_matching_loss()
        self.reg_loss = self.regularization_loss()
        self.loss = self.text_loss + tf.constant(opts.reg_term) * self.reg_loss

        """OPTIMIZATION OPERATION"""
        optimizer = tf.train.AdamOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ensures we execute the update_ops before performing the train_step
            self.train_op = optimizer.minimize(self.loss)

    def average(self, ngrams):
        """average ngram inputs: [batch_size, emb_size]"""
        return tf.reduce_mean(ngrams, axis=1)

    def norm(self, ngrams):
        """l2 normalize ngrams"""
        return tf.nn.l2_normalize(ngrams, axis=1)

    def projection(self, ngrams):
        """project ngrams from word to doc embeddings space"""
        return tf.transpose(tf.matmul(self.proj_weights, ngrams, transpose_b=True))

    def ngrams2docs(self, words, norm=True):
        """composition function: projection(norm(average(sum(words, concepts))))"""
        if norm:
            return self.projection(self.norm(self.average(words)))
        else:
            return self.projection(self.average(words))

    def standardize_batch(self, batch):
        """standardization operation to reduce internal covariate shift + hard tanh"""
        batch_norm = tf.layers.batch_normalization(batch, axis=1, scale=False, training=True)
        return tf.clip_by_value(batch_norm, clip_value_min=-1.0, clip_value_max=1.0)

    def compute_true_logits(self, ngrams, true_docs):
        """compute true logits"""
        return tf.reduce_sum(tf.multiply(true_docs, ngrams), axis=1)

    def compute_negative_logits(self, ngrams, negative_docs):
        """compute negative logits"""
        return tf.matmul(negative_docs, ngrams[..., None])[..., 0]  # add and remove extra dimension

    def text_matching_loss(self):
        """compute nce loss"""
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(self.true_logits),
            logits=self.true_logits)
        neg_xent = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(self.neg_logits),
            logits=self.neg_logits), axis=1)
        # compute nce loss with scaled negative examples
        nce_loss = tf.reduce_mean(((self.options.neg_samples + 1.0) / (2.0 * self.options.neg_samples)) *
                                  (self.options.neg_samples * true_xent + neg_xent))
        return nce_loss

    def regularization_loss(self):
        """compute regularization loss"""
        reg_loss = tf.nn.l2_loss(self.word_embs) + tf.nn.l2_loss(self.proj_weights) + tf.nn.l2_loss(self.doc_embs)
        reg_loss /= self.options.batch_size
        return reg_loss
