import glob
import os
import argparse
import re
import pickle
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from whoosh import index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import *
from whoosh.formats import Positions

from utils import Utils

flags = argparse.ArgumentParser()

flags.add_argument("--data_folder", default="corpus/wsj/wsj", type=str, help="path to collection folder.")
flags.add_argument("--output_folder", default="outputs", type=str, help="output folder where to store data.")
flags.add_argument("--stopwords", default="indri_stopwords.txt", type=str, help="path to stopwords file")
flags.add_argument("--seed", default=42, type=int,
                   help="answer to ultimate question of life, the univrese and everything.")
flags.add_argument("--include_oov", default=False, type=bool, required=False,
                   help="whether to include out of vocabulary token into term dictionary.")
FLAGS = flags.parse_args()


class Options(object):
    """options used for indexing"""

    def __init__(self):
        # data folder path
        self.data_folder = FLAGS.data_folder
        # output folder path
        self.output_folder = FLAGS.output_folder
        # stopwords path
        self.stopwords = FLAGS.stopwords
        # seed
        self.seed = FLAGS.seed
        # oov
        self.oov = FLAGS.include_oov


def set_schema(stopwords_path=None):
    """define schema to index collection"""
    if stopwords_path:
        # use custom stop list
        with open(stopwords_path, 'r') as f:
            stop_words = [stop.strip() for stop in f]
        # standard analyzer: regex tokenizer, lowercase filter, stopwords removal
        analyzer = StandardAnalyzer(stoplist=stop_words)  # minsize: minimum token size to be indexed (default: 2)
    else:
        # use whoosh default stop list
        analyzer = StandardAnalyzer()  # minsize: minimum token size to be indexed (default: 2)
    # create schema
    schema = Schema(docno=ID(stored=True), text=TEXT(analyzer, vector=Positions()))
    return schema


def indexing(corpus_path, index_path, schema):
    """create index for given corpus"""
    if index.exists_in(index_path):
        print('index already exists at: %s - delete index first and the re-run' % index_path)
        return False

    if not os.path.exists(index_path):
        print('creating index folder')
        os.makedirs(index_path)

    # create index
    ix = index.create_in(index_path, schema)
    # create index writer
    writer = ix.writer()
    # load corpus
    print('loading corpus')
    corpus = get_trec_corpus(corpus_path)
    # loop over documents in corpus
    print('adding docs to index')
    for docs in tqdm(corpus):
        # parse xml doc structure
        root = ETree.fromstring(docs)
        # loop through each document
        for doc in tqdm(root):
            text = u''
            docno = u''
            # loop through each element (tag, value)
            for elem in doc:
                if elem.tag == 'DOCNO':
                    # store doc id
                    docno += elem.text.strip()
                # check if elem has children
                if len(elem) == 0:  # elem does not contain children
                    if elem.text:
                        text += elem.text.strip() + ' '
                else:  # elem does contain children
                    text += get_text_recursively(elem, text)
            # add doc to index
            writer.add_document(docno=docno, text=text)
    # commit documents
    print('committing docs to index')
    writer.commit()
    print('indexing finished')
    return True


def get_recursively_files_in_folder(root_folder):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(root_folder):
        for file in f:
            if os.path.isfile(os.path.join(r, file)) and not file.startswith(r'.'):
                files.append(os.path.join(r, file))
    return files


def get_trec_corpus(corpus_path):
    # print('corpus path: ' + str(corpus_path))
    """convert trec style corpus into valid xml"""
    corpus = list()
    docs_path = get_recursively_files_in_folder(corpus_path)
    for fpath in tqdm(docs_path):
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:  # read file
            xml = f.read()
        # convert into true xml
        xml = '<ROOT>' + xml + '</ROOT>'
        # fix not well-formed tokens
        xml = xml.replace('&', '&amp;')
        # required for Robust04 - FBIS 
        xml = re.sub(r"<F P=\d+>", "F", xml)
        xml = re.sub(r"</F>", "F", xml)
        xml = re.sub(r"<FIG.*?>", "FIG", xml)
        xml = re.sub(r"</FIG>", "FIG", xml)
        xml = re.sub(r"<3>", "3", xml)
        xml = re.sub(r"</3>", "3", xml)
        corpus.append(xml)
    return corpus


def get_text_recursively(root, text):
    """recursively iterate over xml element tree"""
    for elem in root:
        if len(elem) == 0:  # elem does not contain children
            if elem.text:
                text += elem.text.strip() + ' '
        else:  # element does contain children
            # recursively obtain children contents
            text = get_text_recursively(elem, text)
    return text


def main():
    # os.chdir(os.path.dirname(os.path.realpath('__file__')))
    # load options
    opts = Options()
    # set folders paths
    data_folder_name = opts.data_folder
    index_folder_name = os.path.join(opts.output_folder, 'index')
    processed_folder_name = os.path.join(opts.output_folder, 'processed_data')


    print('data folder: %s' % data_folder_name)
    print('index folder: %s' % index_folder_name)
    print('processed data folder: %s' % processed_folder_name)

    # set stoplist path
    stoplist_path = opts.stopwords
    # set schema
    schema = set_schema(stoplist_path)
    # index collection
    print('start indexing collection')
    indexing(data_folder_name, index_folder_name, schema)
    # process indexed data
    utils = Utils(opts.seed)
    # load index
    print('loading index')
    ix = index.open_dir(index_folder_name)
    # compute index statistics
    print('compute index statistics')
    utils.index_statistics(ix)
    # prepare folder structure
    if not os.path.exists(processed_folder_name):
        os.makedirs(processed_folder_name)
    # create term dictionary if not present
    if not os.path.exists(processed_folder_name + '/term_dictionary'):  # term dictionary needs to be processed
        print('term dictionary does not exist: create term dictionary')
        # create dictionary
        utils.build_term_dictionary(ix, oov=opts.oov)
        # store dictionary
        with open(processed_folder_name + '/term_dictionary', 'wb') as td:
            pickle.dump(utils.term_dict, td)
        print('term dictionary created and stored in ' + processed_folder_name)
        # encode corpus if not already encoded
        print('corpus needs to be encoded')
        # encode corpus
        print('encoding corpus')
        corpus = utils.corpus2idx(ix, opts.oov)
        # store corpus as pickle folder
        with open(processed_folder_name + '/corpus', 'wb') as cf:
            pickle.dump(corpus, cf)
        print('corpus encoded and stored in ' + processed_folder_name)
    else:  # dictionary exists
        print('term dictionary already exists')
        # encode corpus if not already encoded
        if not os.path.exists(processed_folder_name + '/corpus'):  # corpus needs to be encoded
            print('corpus needs to be encoded: loading term dictionary')
            # load term dictionary
            with open(processed_folder_name + '/term_dictionary', 'rb') as td:
                utils.term_dict = pickle.load(td)
            # encode corpus
            print('encoding corpus')
            corpus = utils.corpus2idx(ix, opts.oov)
            # store corpus as pickle folder
            with open(processed_folder_name + '/corpus', 'wb') as cf:
                pickle.dump(corpus, cf)
            print('corpus encoded and stored in ' + processed_folder_name)
        else:  # corpus exists
            print('corpus already encoded')


if __name__ == "__main__":
    main()
