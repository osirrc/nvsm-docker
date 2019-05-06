# nvsm-docker
Docker for Neural Vector Space Model Tensorflow implementation (https://arxiv.org/abs/1708.02702?context=cs).

In order to run, download the jig from: https://github.com/osirrc2019/jig.

To perform indexing, run:
prepare --repo albep/nvsm --collections robust04=<robust04_collection_folder>=trectext

To perform training, run:
train --model_folder <path_to_folder_to_save_trained_model> --repo albep/nvsm --topic <path_to_robust04_topics> --test_split <splits/test.txt> --validation_split <splits/validation.txt> --qrels <robust04.qrel>

To perform search, run:
search --repo albep/nvsm --collection robust04 --topic <path_to_robust04_topics> --test_split <splits/test.txt> --output <path_to_ranking_output_folder> --qrels <robust04.qrel> --opts qrels=<robust04.qrel>


<splits/test.txt> and <splits/validation.txt>: the files provided in the sample_data folder of the repository.
<path_to_robust04_topics>: robust04 topics are provided in the jig repository: https://github.com/osirrc2019/jig.
