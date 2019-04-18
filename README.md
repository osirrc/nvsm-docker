# nvsm-docker
Docker for Neural Vector Space Model Tensorflow implementation

The docker application at the moment performs indexing, training, test, validation and evaluation steps in sequence.
To launch the pipeline run:

python -u my_runner.py --repo albep/nvsm --tag latest --collection_path <collection path> --test_split <test_split.txt>  --val_split <validation.txt> --stopwords <stopwords.txt> --qrels <qrels_file.qrel> --topics <topics_file> --output_folder <output_folder>
  
  <test_split.txt>: a txt file with the ids of the queries (one per line), to be used as a test subset during the evaluation step;
  <validation.txt>: a txt file with the ids of the queries (one per line), to be used as a validation subset during the training step;
  <stopwords.txt>: a txt file with one word per line, containing the stoplist to use during indexing;
  <qrels_file.qrel>: a qrels file in TREC format, used during training;
  <topics_file>: a topics file in TREC format, used during training;
  <output_folder>: folder path used by docker to store auxiliary data (i.e. collection index) and output runs.
  
  
  

