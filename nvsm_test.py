import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

import numpy as np
import tensorflow as tf
from whoosh import index

from utils import Utils

flags = tf.app.flags

flags.DEFINE_bool("include_oov", False,
                  "Whether to include out of vocabulary token into term dictionary.")
flags.DEFINE_integer("seed", 0,
                     "Answer to ultimate question of life, the universe and everything.")
flags.DEFINE_string("query_field", "title", "query field to use.")
flags.DEFINE_string("model_name", "nvsm_robust", "Model name.")
flags.DEFINE_string("queries_file", "corpus/wsj/queries/topics.51-200.txt", "queries to use.")
flags.DEFINE_string("qrels_file", "corpus/wsj/qrels/wsj_51-200.qrel", "qrels file to use.")
flags.DEFINE_string("splits_test", "corpus/wsj/splits/test.txt", "path to splits test file")
flags.DEFINE_string("output_folder", "outputs", "output folder.")
flags.DEFINE_string("output_ranking_folder", "test_run.txt", "output path for rankings.")
flags.DEFINE_string("index_folder", "./data", "index parent folder")

FLAGS = flags.FLAGS


class Options(object):
    """options used by the Neural Vector Space Model (NVSM)"""

    def __init__(self):
        # out of vocabulary token
        self.oov = FLAGS.include_oov
        # seed
        self.seed = FLAGS.seed
        # query field
        self.field = FLAGS.query_field
        # model name
        self.model_name = FLAGS.model_name
        # topics path
        self.queries_file_path = FLAGS.queries_file
        # qrels path
        self.qrels_file_path = FLAGS.qrels_file
        # test split path
        self.splits_test_path = FLAGS.splits_test
        # outputs path
        self.output_folder_path = FLAGS.output_folder
        # ranking ouput path
        self.output_ranking_folder = FLAGS.output_ranking_folder
        self.output_ranking_name = 'robust04_test_topics_run.txt'
        self.output_ranking_path = os.path.join(FLAGS.output_ranking_folder, self.output_ranking_name)
        self.index_folder = FLAGS.index_folder
        # os.path.join(FLAGS.output_ranking_folder, 'robust04_test_topics_run.txt')
        print('output ranking file: ' + str(self.output_ranking_path))


def main(_):
    os.chdir(os.path.dirname(os.path.realpath('__file__')))
    # load options
    opts = Options()
    # set paths and create paths accordingly
    index_folder_name = os.path.join(opts.index_folder, 'index')
    processed_folder_name = os.path.join(opts.index_folder, 'processed_data')

    print('index folder name: %s' % index_folder_name)
    print('processed folder name: %s' % processed_folder_name)

    models_folder_name = os.path.join(opts.output_folder_path, 'models')
    if not os.path.exists(os.path.join(models_folder_name, opts.model_name)):
        print('models does not exists - please run nvsm_train.py first.')
        return False
    """
    rankings_folder_name = os.path.join(opts.output_folder_path, 'rankings') 
    if not os.path.exists(os.path.join(rankings_folder_name, opts.model_name)):
        print('rankings does not exists - please run nvsm_train.py first.')
        return False
    """
    qrels_folder_name = os.path.join(opts.output_folder_path, 'qrels')
    if not os.path.exists(qrels_folder_name):
        print('qrels does not exists - please run nvsm_train.py first.')
        return False
    # check if index exists
    if not os.path.exists(index_folder_name):
        print('index path does not exists - please run indexing.py first.')
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
    # load queries
    print('loading queries')
    qdict = utils.read_trec_queries(opts.queries_file_path)
    # generate query subsets for validation and test
    print('keep test queries')
    # load test ids
    with open(opts.splits_test_path, 'r') as test_file:
        test_ids = test_file.read()
    test_ids = test_ids.split('\n')
    # generate query subsets for test
    test_queries = {}
    for qid, qtext in qdict.items():
        if str(int(qid)) in test_ids:
            test_queries[qid] = qtext
    """
    # generate qrels subsets for validation and test
    if not os.path.exists(os.path.join(qrels_folder_name, 'test_qrels.qrel')): 
        print('keep qrels test subset')
        # read qrels
        with open(opts.qrels_file_path, 'r') as qrels_f:
            qrels = qrels_f.readlines()
        # generate qrels subsets for test 
        test_qrels = [qrel for qrel in qrels if qrel.split('\t')[0] in test_ids]
        # write qrels 
        with open(os.path.join(qrels_folder_name, 'test_qrels.qrel'), 'w') as out_test:
            # test qrels 
            for qrel in test_qrels:
                out_test.write(qrel)
    """
    # get doc labels
    doc_labels = utils.get_doc_labels(ix)
    # load best epoch
    with open(os.path.join(models_folder_name, opts.model_name) + '/best_epoch.txt', 'r') as bef:
        best_epoch = bef.read()

    # start test graph
    with tf.Session() as sess:
        # restore model and get required tensors
        saver = tf.train.import_meta_graph(
            os.path.join(models_folder_name, opts.model_name) + '/' + opts.model_name + best_epoch + '.ckpt.meta')
        saver.restore(sess,
                      os.path.join(models_folder_name, opts.model_name) + '/' + opts.model_name + best_epoch + '.ckpt')
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
        utils.perform_search(doc_labels, doc_embs, query_ids, queries, opts.output_ranking_path)
        # utils.evaluate_rankings(self.output_ranking_path, qrels_folder_name, 'test_qrels')


if __name__ == "__main__":
    tf.app.run()
