import os
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import pickle

import numpy as np
import tensorflow as tf
from whoosh.index import open_dir

import indexing
from utils import Utils

flags = tf.app.flags

flags.DEFINE_integer("word_embs_size", 300, "The word embedding dimension size.")
flags.DEFINE_integer("doc_embs_size", 256, "The document embedding dimension size.")
flags.DEFINE_integer(
    "epochs", 1,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_integer("negative_samples", 10,
                     "Negative samples per training example.")
flags.DEFINE_integer("num_true", 1,
                     "Number of true labels per training example.")
flags.DEFINE_float("regularization_term", 1e-2, "Regularization parameter.")
flags.DEFINE_integer("batch_size", 51200,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("ngram_size", 16,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_bool("l2_norm_ngrams", True,
                  "Whether to l2 normalize ngram representations.")
flags.DEFINE_bool("include_oov", False,
                  "Whether to include out of vocabulary token into term dictionary.")
flags.DEFINE_integer("seed", 0,
                     "Answer to ultimate question of life, the universe and everything.")
flags.DEFINE_string("corpus", "robust04", "corpus name to use.")
flags.DEFINE_string("queries", "topics.301-450_601-700.txt", "queries to use.")
flags.DEFINE_string("query_field", "title", "query field to use.")
# flags.DEFINE_string("qrels", "robust04.qrel", "qrels file to use.")
# flags.DEFINE_string("stopwords", "indri_stopwords.txt", "stopwords to use.")
flags.DEFINE_string("model_name", "nvsm_robust04", "Model name.")

flags.DEFINE_string("qrels_file", "corpus/robust04/qrels/robust04.qrel", "qrels file to use.")
flags.DEFINE_string("data_folder", "corpus/robust04/robust04/robust04", "path to collection folder")
flags.DEFINE_string("splits_test", "corpus/robust04/splits/test.txt", "path to splits test file")
flags.DEFINE_string("splits_val", "corpus/robust04/splits/validation.txt", "path to splits val file")
flags.DEFINE_string("stopwords", "indri_stopwords.txt", "path to stopwords file")
flags.DEFINE_string("queries_file", "corpus/robust04/queries/topics.301-450_601-700.txt", "queries to use.")

flags.DEFINE_string("output_folder", "output_folder", "output folder.")

FLAGS = flags.FLAGS


class Options(object):
    """options used by the Neural Vector Space Model (NVSM)"""

    def __init__(self):
        # word embeddings dimension
        self.word_size = FLAGS.word_embs_size
        # document embeddings dimension
        self.doc_size = FLAGS.doc_embs_size
        # number of negative samples per example
        self.neg_samples = FLAGS.negative_samples
        # number of true labels per example
        self.num_true = FLAGS.num_true
        # regularization term
        self.reg_term = FLAGS.regularization_term
        # epochs to train
        self.epochs = FLAGS.epochs
        # batch size
        self.batch_size = FLAGS.batch_size
        # ngram size
        self.ngram_size = FLAGS.ngram_size
        # l2 normalization for ngrams
        self.l2_norm = FLAGS.l2_norm_ngrams
        # out of vocabulary token
        self.oov = FLAGS.include_oov
        # seed
        self.seed = FLAGS.seed
        # collection name
        self.corpus = FLAGS.corpus
        # query file
        self.query_file = FLAGS.queries
        # query field
        self.field = FLAGS.query_field
        # qrels file
        # self.qrels_file = FLAGS.qrels
        # model name
        self.model_name = FLAGS.model_name
        # data folder
        self.data_folder = FLAGS.data_folder
        # test and validation qrels and splits paths
        self.splits_test_path = FLAGS.splits_test
        self.splits_val_path = FLAGS.splits_val
        self.stopwords_path = FLAGS.stopwords
        self.qrels_file_path = FLAGS.qrels_file
        self.topics_file_path = FLAGS.queries_file
        self.output_folder_path = FLAGS.output_folder



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


def main(_):
    os.chdir(os.path.dirname(os.path.realpath('__file__')))
    # load options
    opts = Options()
    # set folders
    data_folder_name = opts.data_folder
    index_folder_name = os.path.join(opts.output_folder_path, 'index') # 'corpus/' + opts.corpus + '/index'
    model_folder_name = os.path.join(opts.output_folder_path,'models')#'corpus/' + opts.corpus + '/models'
    quick_folder_name = os.path.join(opts.output_folder_path, 'quick_loads') # 'corpus/' + opts.corpus + '/quick_loads'
    # query_folder_name = 'corpus/' + opts.corpus + '/queries'
    qrels_folder_name = os.path.join(opts.output_folder_path, 'qrels') #'corpus/' + opts.corpus + '/qrels'
    rankings_folder_name = os.path.join(opts.output_folder_path, 'rankings') #'corpus/' + opts.corpus + '/rankings'
    # splits_folder_name = 'corpus/' + opts.corpus + '/splits'
    # set stoplist path
    stoplist_path = opts.stopwords_path # './' + opts.stopwords
    # check if index exists
    if not os.path.exists(index_folder_name):  # index collection
        indexing.main(data_folder_name, index_folder_name, opts.corpus, stoplist_path)
    # load index
    print('loading index')
    ix = open_dir(index_folder_name)
    # load utils functions - set seed value
    utils = Utils(opts.seed)

    if not os.path.exists(os.path.join(rankings_folder_name, opts.model_name)):
        os.makedirs(os.path.join(rankings_folder_name, opts.model_name))
    if not os.path.exists(qrels_folder_name):
        os.makedirs(qrels_folder_name)
    if not os.path.exists(os.path.join(model_folder_name, opts.model_name)):
        os.makedirs(os.path.join(model_folder_name, opts.model_name))
    if os.path.exists(os.path.join(qrels_folder_name, opts.model_name)):
        os.makedirs(os.path.join(qrels_folder_name, opts.model_name))
    if not os.path.exists(quick_folder_name):
        os.makedirs(quick_folder_name)

    if not os.path.exists(quick_folder_name + '/term_dictionary'):  # term dictionary needs to be processed
        print('term dictionary does not exist: create term dictionary')
        # create dictionary
        print('creating term dictionary')
        utils.build_term_dictionary(ix)
        with open(quick_folder_name + '/term_dictionary', 'wb') as td:
            pickle.dump(utils.term_dict, td)
    else:  # dictionary exists
        print('term dictionary exists: load term dictionary')
        # load term dictionary
        with open(quick_folder_name + '/term_dictionary', 'rb') as td:
            utils.term_dict = pickle.load(td)

    # compute index statistics
    print('compute index statistics')
    utils.index_statistics(ix)
    print('term dictionary size: {}'.format(len(utils.term_dict)))

    if not os.path.exists(quick_folder_name + '/corpus'):  # corpus needs to be processed
        print('processing corpus')
        corpus = utils.corpus2idx(ix, opts.oov)
        # store corpus as pickle folder
        with open(quick_folder_name + '/corpus', 'wb') as cf:
            pickle.dump(corpus, cf)
    else:  # corpus exists
        print('loading corpus')
        with open(quick_folder_name + '/corpus', 'rb') as cf:
            # load corpus
            corpus = pickle.load(cf)

    # compute corpus statistics
    print('compute corpus statistics')
    utils.corpus_statistics(corpus)

    # load validation ids
    #with open(splits_folder_name + '/validation.txt', 'r') as val_file:
    with open(opts.splits_val_path, 'r') as val_file:
        val_ids = val_file.read()
    val_ids = val_ids.split('\n')
    # load test ids
    # with open(splits_folder_name + '/test.txt', 'r') as test_file:
    with open(opts.splits_test_path, 'r') as test_file:    
        test_ids = test_file.read()
    test_ids = test_ids.split('\n')

    # load queries
    print('loading queries')
    #qdict = utils.read_trec_queries(query_folder_name + '/' + opts.query_file)  # @smarchesin TODO: make it a choice with flags
    qdict = utils.read_trec_queries(opts.topics_file_path)  # @smarchesin TODO: make it a choice with flags
    # generate query subsets for validation
    val_queries = {}
    for qid, qtext in qdict.items():
        if str(int(qid)) in val_ids:
            val_queries[qid] = qtext
    # generate query subsets for test
    test_queries = {}
    for qid, qtext in qdict.items():
        if str(int(qid)) in test_ids:
            test_queries[qid] = qtext

    if not os.path.exists(os.path.join(qrels_folder_name, 'val_qrels.qrel')) or not os.path.exists(os.path.join(qrels_folder_name, 'test_qrels.qrel')): 
        # read qrels
        # with open(qrels_folder_name + '/' + opts.qrels_file, 'r') as qrels_file:
        with open(opts.qrels_file_path, 'r') as qrels_f:
            qrels = qrels_f.readlines()
        # generate qrels subsets for validation
        val_qrels = [qrel for qrel in qrels if qrel.split('\t')[0] in val_ids]
        # generate qrels subsets for test 
        test_qrels = [qrel for qrel in qrels if qrel.split('\t')[0] in test_ids]
        # write qrels 
        with open(os.path.join(qrels_folder_name,'val_qrels.qrel'), 'w') as out_val, open(os.path.join(qrels_folder_name, 'test_qrels.qrel'), 'w') as out_test:
            # validation qrels
            for qrel in val_qrels:
                out_val.write(qrel)
            # test qrels 
            for qrel in test_qrels:
                out_test.write(qrel)

    # get list of allowed document indexes
    allowed_docs = utils.get_allowed_docs(corpus, opts.ngram_size)
    # get doc labels
    doc_labels = utils.get_doc_labels(ix)
    # start session
    with tf.Graph().as_default(), tf.Session() as sess:
        # declare model parameters
        num_batches = utils.compute_num_batches(corpus, opts.batch_size, opts.ngram_size)
        print('number of batches per epoch: {}'.format(num_batches))
        # add checkpoints to training - one checkpoint per epoch
        save_embeddings_every = num_batches
        print_loss_every = num_batches
        # setup the model
        model = NVSM(len(utils.term_dict), len(corpus), opts)
        # create model saving operation - keeps as many saved models as number of epochs
        saver = tf.train.Saver(max_to_keep=opts.epochs)
        # initialize the variables using global_variables_initializer()
        sess.run(tf.global_variables_initializer())

        print('start training')
        map_per_epoch = list()
        # loop over epochs
        for epoch in range(opts.epochs):
            loss = list()
            loss_at_step = list()
            print('training epoch {}'.format(epoch + 1))
            # loop over batches
            for i in range(num_batches):
                # generate batch data
                batch_data = utils.generate_batch_data(corpus, allowed_docs, opts.batch_size, opts.ngram_size, opts.neg_samples)
                # feed feed_dict
                feed_dict = {model.ngram_words: batch_data[0],
                             model.labels: batch_data[1],
                             model.negative_labels: batch_data[2]}
                # run train_op
                sess.run(model.train_op, feed_dict=feed_dict)
                if (i + 1) % print_loss_every == 0:
                    loss_value, text_loss, reg_loss = sess.run([model.loss, model.text_loss, model.reg_loss], feed_dict=feed_dict)
                    print(loss_value, text_loss, reg_loss)
                    loss.append(loss_value)
                    loss_at_step.append(i + 1)
                    print('loss at step {}: {}'.format(i + 1, loss_value))
                # save embeddings and extract them for validation
                if (i + 1) % save_embeddings_every == 0:
                    model_checkpoint_path = os.path.join(os.getcwd(), model_folder_name + '/' + opts.model_name, opts.model_name + str(epoch + 1) + '.ckpt')
                    save_path = saver.save(sess, model_checkpoint_path)
                    print('model saved in file: {}'.format(save_path))
                    word_embs, proj_weights, doc_embs = sess.run([model.word_embs, model.proj_weights, model.doc_embs])
            print('validation at epoch {}'.format(epoch + 1))
            queries = list()
            query_ids = list()
            # loop over queries and generate rankings
            for qid, qtext in val_queries.items():  # TODO: make field choice as a flag param
                # prepare query for document matching
                proj_query = utils.prepare_query(qid, qtext[opts.field], word_embs, proj_weights, opts.oov)
                if proj_query is not None:  # query contains known tokens
                    queries.append(proj_query)
                    query_ids.append(qid)
            queries = np.array(queries)
            utils.perform_search(doc_labels, doc_embs, query_ids, queries, rankings_folder_name + '/' + opts.model_name, opts.model_name + str(epoch + 1))
            # evaluate ranking and store MAP value
            map_value = utils.evaluate_rankings(rankings_folder_name + '/' + opts.model_name, opts.model_name + str(epoch + 1), qrels_folder_name, 'val_qrels')
            # append MAP value to list
            map_per_epoch.append(map_value)

    # get best model in terms of MAP
    best_epoch = np.argsort(map_per_epoch)[-1] + 1
    # start test graph
    with tf.Session() as sess:
        # restore model and get required tensors
        saver = tf.train.import_meta_graph(os.path.join(model_folder_name, opts.model_name) + '/' + opts.model_name + str(best_epoch) + '.ckpt.meta')
        saver.restore(sess, os.path.join(model_folder_name, opts.model_name) + '/' + opts.model_name + str(best_epoch) + '.ckpt')
        word_embs = sess.run(tf.get_default_graph().get_tensor_by_name('word_embs:0'))
        proj_weights = sess.run(tf.get_default_graph().get_tensor_by_name('proj_weights:0'))
        doc_embs = sess.run(tf.get_default_graph().get_tensor_by_name('doc_embs:0'))
        print('testing best model found at epoch {}'.format(best_epoch))
        queries = list()
        query_ids = list()
        # loop over queries and generate rankings
        for qid, qtext in test_queries.items():  # TODO: make field choice as a flag param
            # prepare query for document matching
            proj_query = utils.prepare_query(qid, qtext[opts.field], word_embs, proj_weights, opts.oov)
            if proj_query is not None:  # query contains known tokens
                queries.append(proj_query)
                query_ids.append(qid)
        queries = np.array(queries)
        utils.perform_search(doc_labels, doc_embs, query_ids, queries, rankings_folder_name + '/' + opts.model_name, opts.model_name + 'test_epoch' + str(best_epoch))
        utils.evaluate_rankings(rankings_folder_name + '/' + opts.model_name, opts.model_name + 'test_epoch' + str(best_epoch), qrels_folder_name, 'test_qrels')

if __name__ == "__main__":
    tf.app.run()
