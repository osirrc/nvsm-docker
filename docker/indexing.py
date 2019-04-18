import glob
import os
import re
import xml.etree.ElementTree as ETree

from tqdm import tqdm
from whoosh import index
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import *
from whoosh.formats import Positions


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


def indexing(corpus_path, index_path, schema, corpus_name):
    """create index for given corpus"""
    if not os.path.exists(index_path):
        print('creating index folder')
        os.mkdir(index_path)
    if index.exists_in(index_path):
        print('index exists - clearing previous index')  # @smarchesin: TODO modify to make it an interactive choice
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


def main(data_folder, index_folder, corpus_name, stoplist_path=None):
    os.chdir(os.path.dirname(os.path.realpath('__file__')))
    # set schema
    schema = set_schema(stoplist_path)
    # index collection
    indexing(data_folder, index_folder, schema, corpus_name)
    return True
