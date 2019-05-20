import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import argparse

import numpy as np
import tensorflow as tf
from whoosh import index

from utils import Utils
from model import NVSM

flags = tf.flags

flags.DEFINE_integer("word_embs_size", 300, "The word embedding dimension size.")
flags.DEFINE_integer("doc_embs_size", 256, "The document embedding dimension size.")
flags.DEFINE_integer("epochs", 10, "Number of epochs to train. Each epoch processes the training data once completely.")
flags.DEFINE_integer("negative_samples", 10,
                     "Negative samples per training example.")
flags.DEFINE_integer("num_true", 1,
                     "Number of true labels per training example.")
flags.DEFINE_float("regularization_term", 1e-2, "Regularization parameter.")
flags.DEFINE_integer("batch_size", 51200,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("ngram_size", 16,
                     "The number of words to predict to the left and right of the target word.")
flags.DEFINE_bool("l2_norm_ngrams", True,
                  "Whether to l2 normalize ngram representations.")
flags.DEFINE_bool("include_oov", False,
                  "Whether to include out of vocabulary token into term dictionary.")
flags.DEFINE_integer("seed", 0,
                     "Answer to ultimate question of life, the universe and everything.")
flags.DEFINE_string("stopwords", "indri_stopwords.txt", "path to stopwords file")
flags.DEFINE_string("data_folder", "corpus/wsj/wsj", "path to collection folder")
flags.DEFINE_string("queries_file", "corpus/wsj/queries/topics.51-200.txt", "queries to use.")
flags.DEFINE_string("query_field", "title", "query field to use.")
flags.DEFINE_string("model_name", "nvsm_robust", "Model name.")
flags.DEFINE_string("qrels_file", "corpus/wsj/qrels/wsj_51-200.qrel", "qrels file to use.")
flags.DEFINE_string("splits_test", "corpus/wsj/splits/test.txt", "path to splits test file")
flags.DEFINE_string("splits_val", "corpus/wsj/splits/validation.txt", "path to splits val file")
flags.DEFINE_string("output_folder", "outputs", "output folder.") # data
flags.DEFINE_string("model_folder", "outputs", "output model folder.") # output

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
        # query field
        self.field = FLAGS.query_field
        # model name
        self.model_name = FLAGS.model_name
        # data folder
        self.data_folder = FLAGS.data_folder
        # test split path
        self.splits_test_path = FLAGS.splits_test
        # validation split path
        self.splits_val_path = FLAGS.splits_val
        # stopwords path
        self.stopwords_path = FLAGS.stopwords
        # qrels path
        self.qrels_file_path = FLAGS.qrels_file
        # topics path
        self.queries_file_path = FLAGS.queries_file
        # outputs path
        self.output_folder_path = FLAGS.output_folder

        self.model_folder_path = FLAGS.model_folder


def main(_):
    os.chdir(os.path.dirname(os.path.realpath('__file__')))
    # load options
    opts = Options()
    # set paths and create paths accordingly
    index_folder_name = os.path.join(opts.output_folder_path, 'index')
    processed_folder_name = os.path.join(opts.output_folder_path, 'processed_data')
    models_folder_name = os.path.join(opts.model_folder_path, 'models')
    if not os.path.exists(os.path.join(models_folder_name, opts.model_name)):
        os.makedirs(os.path.join(models_folder_name, opts.model_name))

    rankings_folder_name = os.path.join(opts.model_folder_path, 'rankings')
    if not os.path.exists(os.path.join(rankings_folder_name, opts.model_name)):
        os.makedirs(os.path.join(rankings_folder_name, opts.model_name))

    qrels_folder_name = os.path.join(opts.model_folder_path, 'qrels')
    if not os.path.exists(qrels_folder_name):
        os.makedirs(qrels_folder_name)
    # check if index exists
    if not os.path.exists(index_folder_name):
        print('index does not exists - please run indexing.py first.')
        return False
    elif not index.exists_in(index_folder_name):
        print('index does not exists - please run indexing.py first.')
        return False
    else:
        # load index
        print('loading index')
        ix = index.open_dir(index_folder_name)
    # load utils functions - set seed value
    utils = Utils(opts.seed)
    # load processed data
    if not os.path.exists(processed_folder_name + '/term_dictionary'):  # term dictionary needs to be created
        print('term dictionary does not exist - please run indexing.py first.')
        return False
    else:  # term dictionary exists
        print('term dictionary exists: load term dictionary')
        # load term dictionary
        with open(processed_folder_name + '/term_dictionary', 'rb') as td:
            utils.term_dict = pickle.load(td)
    # print term dictionary size
    print('term dictionary size: {}'.format(len(utils.term_dict)))
    # load encoded corpus
    if not os.path.exists(processed_folder_name + '/corpus'):  # corpus needs to be processed
        print('corpus not encoded - please run indexing.py first.')
        return False
    else:  # corpus exists
        print('loading corpus')
        with open(processed_folder_name + '/corpus', 'rb') as cf:
            # load corpus
            corpus = pickle.load(cf)
    # load queries
    print('loading queries')
    qdict = utils.read_trec_queries(opts.queries_file_path)
    # generate query subsets for validation and test
    print('split queries into validation and test subsets')
    # load validation ids
    with open(opts.splits_val_path, 'r') as val_file:
        val_ids = val_file.read()
    val_ids = val_ids.split('\n')
    # load test ids
    with open(opts.splits_test_path, 'r') as test_file:
        test_ids = test_file.read()
    test_ids = test_ids.split('\n')
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
    # generate qrels subsets for validation and test
    if not os.path.exists(os.path.join(qrels_folder_name, 'val_qrels.qrel')) or not os.path.exists(
            os.path.join(qrels_folder_name, 'test_qrels.qrel')):
        print('split qrels into validation and test subsets')
        # read qrels
        with open(opts.qrels_file_path, 'r') as qrels_f:
            qrels = qrels_f.readlines()
        # generate qrels subsets for validation
        val_qrels = [qrel for qrel in qrels if qrel.split('\t')[0] in val_ids]
        # generate qrels subsets for test 
        test_qrels = [qrel for qrel in qrels if qrel.split('\t')[0] in test_ids]
        # write qrels 
        with open(os.path.join(qrels_folder_name, 'val_qrels.qrel'), 'w') as out_val, open(
                os.path.join(qrels_folder_name, 'test_qrels.qrel'), 'w') as out_test:
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
        # start training NVSM
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
                batch_data = utils.generate_batch_data(corpus, allowed_docs, opts.batch_size, opts.ngram_size,
                                                       opts.neg_samples)
                # feed feed_dict
                feed_dict = {model.ngram_words: batch_data[0],
                             model.labels: batch_data[1],
                             model.negative_labels: batch_data[2]}
                # run train_op
                sess.run(model.train_op, feed_dict=feed_dict)
                if (i + 1) % print_loss_every == 0:
                    loss_value, text_loss, reg_loss = sess.run([model.loss, model.text_loss, model.reg_loss],
                                                               feed_dict=feed_dict)
                    print(loss_value, text_loss, reg_loss)
                    loss.append(loss_value)
                    loss_at_step.append(i + 1)
                    print('loss at step {}: {}'.format(i + 1, loss_value))
                # save embeddings and extract them for validation
                if (i + 1) % save_embeddings_every == 0:
                    model_checkpoint_path = os.path.join(os.getcwd(), models_folder_name + '/' + opts.model_name,
                                                         opts.model_name + str(epoch + 1) + '.ckpt')
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
            ranking_path = os.path.join(rankings_folder_name + '/' + opts.model_name, opts.model_name + str(epoch + 1))
            print('creating ranking: ' + ranking_path)
            utils.perform_search(doc_labels, doc_embs, query_ids, queries, ranking_path)
            # evaluate ranking and store MAP value
            print('evaluate ranking at: ' + ranking_path)
            map_value = utils.evaluate_rankings(ranking_path, qrels_folder_name, 'val_qrels')
            # append MAP value to list
            map_per_epoch.append(map_value)

    # get best model in terms of MAP
    best_epoch = np.argsort(map_per_epoch)[-1] + 1
    print('best model found at epoch {}'.format(best_epoch))
    # store best epoch information
    with open(os.path.join(models_folder_name, opts.model_name) + '/best_epoch.txt', 'w') as bef:
        bef.write('{}'.format(best_epoch))


if __name__ == "__main__":
    tf.app.run()
