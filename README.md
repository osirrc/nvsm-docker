# nvsm-docker
Docker for Neural Vector Space Model Tensorflow implementation (https://arxiv.org/abs/1708.02702?context=cs).

To perform indexing run:
python run_nvsm.py prepare --repo albep/nvsm --collections robust04 --opts coll_folder=<collection_folder>+output_folder=<output_folder>+stopwords=<stopwords.txt>

To perform training run:
python run_nvsm.py train --repo albep/nvsm --opts test_split=<test_split.txt>+val_split=<validation.txt>+qrels=<qrels_file.qrels>+topics=<topics_file>+output_folder_host=<output_folder>

To perform search run:
python run_nvsm.py search --repo albep/nvsm --opts test_split=<test_split.txt>+qrels=<qrels_file.qrel>+topics=<topics_file>+output_folder_host=<output_folder>

  <collection_folder>: root of collection folder in trec format (i.e. robust04)
  <test_split.txt>: a txt file with the ids of the queries (one per line), to be used as a test subset during the evaluation step;
  <validation.txt>: a txt file with the ids of the queries (one per line), to be used as a validation subset during the training step;
  <stopwords.txt>: a txt file with one word per line, containing the stoplist to use during indexing;
  <qrels_file.qrel>: a qrels file in TREC format, used during training;
  <topics_file>: a topics file in TREC format, used during training;
  <output_folder>: folder path used by docker to store auxiliary data (i.e. collection index) and output runs.
  
  
  

