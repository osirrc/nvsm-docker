# OSIRRC Docker Image for NVSM
[![Generic badge](https://img.shields.io/badge/DockerHub-go%21-yellow.svg)](https://hub.docker.com/r/osirrc2019/nvsm)

[**Nicola Ferro**](https://github.com/frrncl), [**Stefano Marchesin**](https://github.com/stefano-marchesin), [**Alberto Purpura**](https://github.com/albpurpura) and [**Gianmaria Silvello**](https://github.com/giansilv)

This is the docker image of our implementation of [Neural Vector Space Model (NVSM)](https://arxiv.org/abs/1708.02702?context=cs) conforming to the [OSIRRC jig](https://github.com/osirrc/jig/) for the [Open-Source IR Replicability Challenge (OSIRRC) at SIGIR 2019](https://osirrc.github.io/osirrc2019/).
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
  --model_folder path/model/directory \
  --topic topics/topics.robust04.txt \
  --test_split sample_training_validation_query_ids/robust04_test.txt \
  --validation_split sample_training_validation_query_ids/robust04_validation.txt \
  --qrels qrels/qrels.robust04.txt \
  --opts epochs=12 \
  --collection Robust04
```


The following `jig` command can be used to perform a retrieval run on the collection with the `robust04` test collection.

```
python run.py search \
  --repo albep/nvsm \
  --output path/model/directory \
  --qrels qrels/qrels.robust04.txt \
  --topic topics/topics.robust04.txt \
  --test_split sample_training_validation_query_ids/robust04_test.txt \
  --collection robust04
```

## Expected Results

### robust04

MAP                                     | NVSM CPU      | NVSM GPU |
:---------------------------------------|-----------|-----------|
[Robust04 test split topics](https://github.com/osirrc/jig/blob/master/sample_training_validation_query_ids/robust04_test.txt)| 0.138    | 0.138*    |

<nowiki>*</nowiki> Results with the NVSM GPU image may slightly vary. In fact, TensorFlow uses the Eigen library, which uses Cuda atomics to implement reduction operations, such as tf.reduce_sum etc. Those operations are non-deterministical and each operation can introduce small variations. See [this](https://github.com/tensorflow/tensorflow/issues/3103) Tensorflow issue for more details.

## Notes
The paths `path/to/model/directory`, passed to the `train` script, and `path/model/directory`, passed to the `search` one, need to point to the same directory.

nvsm_gpu requires nvidia-docker (https://github.com/NVIDIA/nvidia-docker) installed on the host machine.
