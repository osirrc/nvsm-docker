import os
import math
import string
import subprocess
import itertools
import numpy as np
import xml.etree.ElementTree as ET

from collections import Counter
from functools import reduce
from textwrap import wrap
from whoosh.analysis import SimpleAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class Utils(object):
    """utils functions for neural vector space models"""

    def __init__(self, seed):
        """set random seed, initialize index variables"""
        np.random.seed(seed)
        self.term_dict = {}

    def build_term_dictionary(self, index, dict_size=65536, oov=False, remove_digits=True, min_doc_freq=2,
                              max_doc_freq=0.5):
        """create term dictionary"""
        reader = index.reader()
        # get corpus size
        corpus_size = reader.doc_count()
        # get unique terms statistics: (term, doc_freq, term_freq)
        terms = self.terms_statistics(index)
        # initialize count list
        count = []
        # add terms to count
        for term, doc_freq, term_freq in terms:
            # check if term does not exceed max_doc_freq (in %)
            if doc_freq / corpus_size <= max_doc_freq:
                # check if term is not inferior to min_doc_freq (not in %)
                if doc_freq >= min_doc_freq:
                    # check if term does not contain digits
                    if remove_digits:
                        if self.has_digit(term):  # skip term
                            continue
                        else:  # keep term
                            count.extend([(term, term_freq)])
                    else:  # keep terms containing digits
                        count.extend([(term, term_freq)])
                else:  # minimum doc freq not reached
                    # skip term
                    continue
            else:  # maximum doc freq exceeded
                # skip term
                continue
        # convert count into Counter object and keep dict_size most frequent terms
        count = Counter(dict(count)).most_common(dict_size)
        if oov:
            # include out of vocabulary token
            count.extend([("__UNK__", 1)])  # last index: dict_size
        # for each term - that we want in the dictionary - add it and make it the value of the prior dictionary length
        for term, term_freq in count:
            self.term_dict[term] = len(self.term_dict)
        return True

    def has_digit(self, term):
        """check whether input term contains digits"""
        return any(char.isdigit() for char in term)

    def only_digits(self, term):
        """check whether input term contains only digits and/or punctuation"""
        return all(char.isdigit() or char in string.punctuation for char in term)

    def get_term_dictionary(self):
        """get term dictionary"""
        return self.term_dict

    def update_term_dictionary(self, term):
        """update term dictionary"""
        if term in self.term_dict:  # term already in term_dict
            return True
        else:  # update term_dict
            self.term_dict[term] = len(self.term_dict)
            return True

    def find_pos(self, line):
        """split text into terms and return dict {pos: [term, ["__NULL__"]]}"""
        pos_terms = {}
        terms = line.split()
        # define sentence index
        index = line.index
        running_offset = 0
        # loop over terms
        for term in terms:
            # get term offset
            term_offset = index(term, running_offset)
            term_len = len(term)
            # update running offset
            running_offset = term_offset + term_len
            # append to term_offset each term + ["__NULL__"] for later use
            pos_terms[term_offset] = [term, ["__NULL__"]]
        return pos_terms

    def terms_statistics(self, index):
        """get unique terms statistics"""
        reader = index.reader()
        # unique terms
        terms = list(reader.field_terms('text'))
        # terms statistics
        terms_stats = list()
        # loop over unique terms
        for term in terms:
            # term info
            term_info = reader.term_info('text', term)
            # doc frequency
            doc_freq = term_info.doc_frequency()
            # term frequency
            term_freq = term_info.weight()
            # append info to terms statistics
            terms_stats.append((term, doc_freq, term_freq))
        return terms_stats

    def index_statistics(self, index):
        """compute and print index statistics"""
        reader = index.reader()
        # doc indexes in whoosh
        doc_ids = list(reader.all_doc_ids())
        # corpus size
        corpus_size = reader.doc_count()
        # maximum length of given field across all documents
        max_length = reader.max_field_length('text')
        # minimum length of given field across all documents
        min_length = reader.min_field_length('text')
        # total number of terms in given field
        corpus_length = reader.field_length('text')
        # total number of unique terms
        terms = list(reader.field_terms('text'))
        # number of terms in given field in given document
        docs_length = list()
        for doc_id in doc_ids:
            doc_length = reader.doc_field_length(doc_id, 'text')
            if doc_length:
                docs_length.append(doc_length)
            else:
                docs_length.append(0)
        # average length of given field across all documents in corpus
        avg_length = reduce((lambda x, y: x + y), docs_length) / corpus_size
        # print statistics
        print('corpus size: {}'.format(corpus_size))
        print('maximum length: {}'.format(max_length))
        print('minimum length: {}'.format(min_length))
        print('average length: {}'.format(avg_length))
        print('all terms: {}'.format(corpus_length))
        print('unique terms: {}'.format(len(terms)))
        return True

    def corpus_statistics(self, corpus):
        """compute and print corpus statistics"""
        corpus_size = len(corpus)
        # compute documents lengths
        docs_length = np.array([len(doc) for doc in corpus])
        # compute corpus length
        corpus_length = [term for doc in corpus for term in doc]
        # print statistics
        print('corpus size: {}'.format(corpus_size))
        print('maximum length: {}'.format(np.max(docs_length)))
        print('minimum length: {}'.format(np.min(docs_length)))
        print('average length: {}'.format(np.mean(docs_length)))
        print('median length: {}'.format(np.median(docs_length)))
        print('std length: {}'.format(np.std(docs_length)))
        print('all terms: {}'.format(len(corpus_length)))
        return True

    def compute_num_batches(self, corpus, batch_size, ngram_size):
        """compute number of batch iterations per epoch"""
        docs_length = [len(doc) for doc in corpus]
        # compute number of batches
        num_batches = math.ceil(sum([max(doc_length - ngram_size + 1, 0) for doc_length in docs_length]) / batch_size)
        return num_batches

    def get_doc_labels(self, index):
        """return list of document labels (e.g. TREC <DOCNO> values)"""
        reader = index.reader()
        doc_ids = list(reader.all_doc_ids())
        # define doc labels list
        doc_labels = list()
        for doc_id in doc_ids:
            label = reader.stored_fields(doc_id)['docno']
            doc_labels.append(label)
        return doc_labels

    def corpus2idx(self, index, oov=False):
        """convert documents into list of indices"""
        reader = index.reader()
        # define corpus as a list of lists
        corpus = []
        # get doc ids (whoosh' index ids)
        doc_ids = list(reader.all_doc_ids())
        # encode corpus
        for doc_id in doc_ids:
            # read doc and return its contents as an ordered seq of terms
            terms = self.pos2terms(reader, doc_id)
            # store doc as ordered list of index terms
            doc = list()
            for term in terms:
                if term in self.term_dict:
                    doc.append(self.term_dict[term])
                else:
                    if oov:  # store oov index
                        doc.append(self.term_dict['__UNK__'])
                    else:  # skip term
                        continue
            # store processed doc in corpus
            corpus.append(doc)
        return corpus

    def pos2terms(self, reader, doc_id):
        """return list of ordered doc terms given doc id"""
        if reader.has_vector(doc_id, 'text'):
            doc_data = reader.vector(doc_id, 'text').items_as('positions')
            # get term-positions dict: {term: [pos1, pos2, ...], ...}
            term_pos = dict(doc_data)
            # create position-term dict: {pos1: term, pos2: term, ...}
            pos_term = dict()
            for term, positions in term_pos.items():
                for pos in positions:
                    pos_term[pos] = term
            # return ordered list of doc terms
            return [pos_term.get(i) for i in range(min(pos_term), max(pos_term) + 1)]
        else:  # target doc does not contain terms
            return []

    def generate_batch_data(self, corpus, allowed_docs, batch_size, ngram_size, neg_samples):
        """generate a batch of data for given corpus (optimized)"""
        corpus_size = len(corpus)
        # select random documents from allowed documents (i.e. documents with len(doc) >= ngram_size)
        rand_docs_idx = np.random.choice(allowed_docs, size=batch_size)
        # compute documents length
        docs_length = [len(corpus[rand_doc_idx]) for rand_doc_idx in rand_docs_idx]
        # store position of last prefixes + 1 (one above the highest prefix available)
        last_prefixes = [doc_length - ngram_size + 1 for doc_length in docs_length]
        # sample random prefixes lower than or equal to last_prefixes
        prefixes = [np.random.randint(last_prefix) for last_prefix in last_prefixes]
        # slices = prefixes + ngram_size
        ngrams = [corpus[rand_doc_idx][prefix:prefix + ngram_size] for rand_doc_idx, prefix in
                  zip(rand_docs_idx, prefixes)]
        # generate negative labels - discrete uniform distribution
        negative_labels = np.random.randint(corpus_size, size=[batch_size, neg_samples])
        # convert batch data to numpy array
        ngrams = np.array(ngrams)
        # return batch data in the form: (ngrams, true labels, negative labels)
        return ngrams, rand_docs_idx, negative_labels

    def get_allowed_docs(self, corpus, ngram_size):
        """return list of allowed documents (as whoosh's indexes) for the given ngram size"""
        allowed_docs = list()
        del_docs = list()
        # loop over documents and store doc indexes when len(doc) >= ngram_size
        for idx, doc in enumerate(corpus):
            if len(doc) >= ngram_size:
                allowed_docs.append(idx)
            else:
                del_docs.append(idx)
        print('deleted {} docs'.format(len(del_docs)))
        return np.array(allowed_docs)

    def read_ohsu_queries(self, query_path):
        """read query file and return a dict[id] = {title: <string>, desc: <string>}"""
        with open(query_path, 'r') as qf:
            q = qf.read()
        q = [query.split('\n') for query in q.split('\n\n') if query]
        # loop through each query and fill dict
        qdict = dict()
        for query in q:
            qid = query[1].split()[-1]
            qdict[qid] = dict()
            qdict[qid]['title'] = query[2].split('<title>')[1].strip()
            qdict[qid]['desc'] = query[4]
        return qdict

    def read_trec_queries(self, query_path):
        """read query file and return a dict[id] = query"""
        with open(query_path, 'r') as qf:
            xml = qf.readlines()
        # convert into true xml
        true_xml = []
        # properly close tags
        for line in xml:
            if '<title>' in line:
                line = '</num>\n' + line
            if '<desc>' in line:
                line = '</title>\n' + line
            if '<narr>' in line:
                line = '</desc>\n' + line
            if '</top>' in line:
                line = '</narr>\n' + line
            # remove noisy information
            line = line.replace('Number:', '')
            line = line.replace('Topic:', '')
            line = line.replace('Description:', '')
            # convert non-valid xml chars
            line = line.replace('&', '&amp;')
            # strip string
            line = line.strip()
            true_xml.append(line)
        # reconvert list to single string
        true_xml = ''.join(true_xml)
        # add root
        true_xml = '<ROOT>' + true_xml + '</ROOT>'
        root = ET.fromstring(true_xml)
        # define query dict: {qid: {title:, desc:}, ...}
        qdict = dict()
        # loop through each query
        for q in root:
            qid = q.find('num').text.strip()
            qdict[qid] = {}
            qdict[qid]['title'] = q.find('title').text.strip()
            qdict[qid]['desc'] = q.find('desc').text.strip()
        return qdict

    def read_clef_queries(self, query_path):  # TODO: add description field 
        """read query file and return a dict[id] = query"""
        qdict = dict()
        with open(query_path, 'r') as qf:
            xml = qf.read()
        root = ET.fromstring(xml)
        # loop through each query
        for q in root:
            qid = q.find('identifier').text.strip()
            qdict[qid] = {}
            qdict[qid]['title'] = q.find('title').text.strip()
            qdict[qid]['desc'] = q.find('description').text.strip()
        return qdict

    def tokenize_query(self, q):
        """lowerize and tokenize query"""
        analyzer = SimpleAnalyzer()
        return [token.text for token in analyzer(q)]

    def query2idx(self, q, qid, oov=False):
        """convert query terms to indices"""
        query_idx = list()
        for term in q:
            if term in self.term_dict:
                query_idx.append(self.term_dict[term])
            else:
                if oov:  # keep term as __UNK__ token
                    query_idx.append(self.term_dict['__UNK__'])
                else:  # skip term
                    continue
        if not query_idx:
            print('query {} does not contain terms'.format(qid))
            return None
        else:
            return np.array(query_idx)

    def query_projection(self, query_idx, word_embs, proj_weights):
        """convert list of indices into dense vector of size [1, doc_embs]"""
        if query_idx is None:
            return None
        else:
            return np.matmul(proj_weights, np.mean(word_embs[query_idx], axis=0))

    def prepare_query(self, qid, qtext, word_embs, proj_weights, oov=False):
        """transform query into dense vector of size [1, doc_embs]"""
        query_tokens = self.tokenize_query(qtext)
        query_idx = self.query2idx(query_tokens, qid, oov)
        query_proj = self.query_projection(query_idx, word_embs, proj_weights)
        return query_proj

    def perform_search(self, doc_labels, docs, query_ids, queries, ranking_path):
        """perform search over docs given queries"""
        doc_labels = np.array(doc_labels)
        # compute similarities
        print('compute similarities between docs and queries')
        similarities = cosine_similarity(docs, queries)
        # open file to write results
        ranking_name = 'nvsm'  # os.path.basename(ranking_path)
        # rf = open(ranking_folder + '/' + ranking_name + '.run', 'w')
        rf = open(ranking_path, 'w')
        # write results in ranking file
        for i in tqdm(range(similarities.shape[1])):
            rank = np.argsort(-similarities[:, i])[:1000]
            docs_rank = doc_labels[rank]
            qid = query_ids[i]
            # verify whether qid is an integer
            if qid.isdigit():  # cast to integer - this operation avoids storing topic ids as '059' instead of '59'
                qid = str(int(qid))  # convert to int and then back to str
            for j in range(len(docs_rank)):
                # write into .run file
                rf.write('%s\t%d\t%s\t%d\t%f\t%s\n' % (qid, 0, docs_rank[j], j, similarities[rank[j]][i], ranking_name))
        rf.close()
        return True

    def get_averaged_measure_score(self, run_path, qrel_path, measure):
        """return averaged measure score over topics"""
        if "P_" in measure:
            cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + " " + qrel_path + " " + run_path
        elif "ndcg_cut" in measure:
            cmd = "./trec_eval/trec_eval -m " + measure.split('_')[0] + '_' + measure.split('_')[
                1] + " " + qrel_path + " " + run_path
        else:
            cmd = "./trec_eval/trec_eval -m " + measure + " " + qrel_path + " " + run_path
        process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
        result = process.stdout.decode('utf-8').split('\n')
        qscore = np.array([score.split('\t')[-1] for score in result
                           if score.split('\t')[0].strip() == measure])
        qscore = qscore.astype(np.float)[0]
        return qscore

    def evaluate_rankings(self, ranking_path, qrels_folder, qrels_name):
        """evaluate rankings performed by neural models"""
        qrels_file_path = qrels_folder + '/' + qrels_name + '.qrel'
        print('qrels file: ' + qrels_file_path)
        if not os.path.isfile(qrels_file_path):
            print('QRELS file NOT FOUND!')
        print('evaluate model ranking')
        MAP = self.get_averaged_measure_score(ranking_path, qrels_file_path, 'map')
        NDCG = self.get_averaged_measure_score(ranking_path, qrels_file_path, 'ndcg_cut_100')
        P_10 = self.get_averaged_measure_score(ranking_path, qrels_file_path, 'P_10')
        print('MAP: ' + str(MAP), 'NDCG: ' + str(NDCG), 'P@10: ' + str(P_10))
        return MAP
