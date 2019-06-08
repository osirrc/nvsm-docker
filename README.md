# NVSM OSIRRC Docker Image
[**Alberto Purpura**](https://github.com/albpurpura), [**Stefano Marchesin**](https://github.com/stefano-marchesin), [**Gianmaria Silvello**](https://github.com/giansilv) and [**Nicola Ferro**](https://github.com/frrncl)

This is the docker image of our implementation of Neural Vector Space Model [NVSM](https://arxiv.org/abs/1708.02702?context=cs) conforming to the [OSIRRC jig](https://github.com/osirrc/jig/) for the [Open-Source IR Replicability Challenge (OSIRRC) at SIGIR 2019](https://osirrc.github.io/osirrc2019/).
This image is available on [Docker Hub](https://cloud.docker.com/u/albep/repository/docker/albep/nvsm) has been tested with the jig at commit [ca31987](https://github.com/osirrc/jig/commit/ca3198704795f2b6de8b78ed7a66bbdf1dccadb1) (6/5/2019).

+ Supported test collections: `robust04`
+ Supported hooks: `init`, `index`,  `train`,  `search`

## Quick Start

The following `jig` command can be used to index TREC disks 4/5 for `robust04`:

```
python run.py prepare \
  --repo albep/nvsm \
  --collections robust04=/path/to/disk45=trectext
```

The following `jig` command can be used to train the retrieval model on the `robust04` collection:
```
python run.py train \
  --repo albep/nvsm \
  --model_folder path/to/folder/to/save/model \
  --topic topics/topics.robust04.txt \
  --test_split sample_training_validation_query_ids/robust04_test.txt \
  --validation_split sample_training_validation_query_ids/robust04_validation.txt \
  --qrels qrels/qrels.robust04.txt 
  --opts epochs=12 \
  --collection Robust04
```


The following `jig` command can be used to perform a retrieval run on the collection with the `robust04` test collection.

```
python run.py search \
  --repo albep/nvsm \
  --output path/to/folder/of/saved/model \
  --qrels qrels/qrels.robust04.txt \
  --topic topics/topics.robust04.txt \
  --collection robust04
```
