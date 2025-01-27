# KagNet: Knowledge-Aware Graph Networks
 
_**News:**_
We released a more general-purpose LM-GNN reasoning framework, [MHGRN](https://github.com/INK-USC/MHGRN), which includes more options for text/graph encoders. It also matches the current state-of-the-art performance (76.5% acc) on the offical CommonsenseQA test set. We won't maintain this repo, so please follow the new repo.
 
### Introduction
This codebase is an implementation of the proposed KagNet model for commonsense reasoning (EMNLP-IJCNLP 2019). 


- Overall Workflow
![](figures/intro.jpg)
- GCN + LSTM-based Path Encoder + Hierarchical Path Attention
![](figures/kagnet.png)
 
 
### Install Dependencies 

```
sudo apt-get install graphviz libgraphviz-dev pkg-config
conda create -n kagnet_test python==3.6.3
conda activate kagnet_test
# which python
# which pip
pip install torch torchvision 
pip install tensorflow-gpu==1.10.0
conda install faiss-gpu cudatoolkit=10.0 -c pytorch -n kagnet_test 
pip install nltk
conda install -c conda-forge spacy -n kagnet_test
python -m spacy download en
pip install jsbeautifier
pip install networkx
pip install dgl
pip install pygraphviz
pip install allennlp
```

#### Datasets downloading
```

cd datasets
mkdir csqa_new

wget -P csqa_new https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
wget -P csqa_new https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl 
wget -P csqa_new https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl

在commonsenseQA数据集的每个部分的每一条数据里都生成了statement
这个statement是根据answer和question改来的

python convert_csqa.py csqa_new/train_rand_split.jsonl csqa_new/train_rand_split.jsonl.statements
python convert_csqa.py csqa_new/dev_rand_split.jsonl csqa_new/dev_rand_split.jsonl.statements
python convert_csqa.py csqa_new/test_rand_split_no_answers.jsonl csqa_new/test_rand_split_no_answers.jsonl.statements
```

### Preprocess ConceptNet and embedding files
```
把conceptnet改为<relation><subject><object><weight>的四元组

cd ../conceptnet
wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
gzip -d conceptnet-assertions-5.6.0.csv.gz
python extract_cpnet.py

把conceptnet的每一行的知识，转成一条语料
语料形式：
{"tid": 0, "rel": "antonym", "subj": "ab_extra", "obj": "ab_intra", "temp": "you do not want a #SUBJ#
with #OBJ#", "string": "you do not want a ab extra with ab intra", "subj_start": 5, "subj_end": 7, "obj_start":
8, "obj_end": 10}
temp模板是从template.txt里随机选择的

cd ../triple_string
python triple_string_generation.py

生成concept(subj/obj)和relation的embedding，默认是取concept和relation单词表示的glove embedding(100维)的每一位的max
其中concept(subj/obj)对应每个单词的glove embedding的最大值
relation对应句子里除去subj和obj之外所有单词的glove embedding的最大值，依据检索到不同模板而更新(weighted average)
生成concept_glove.max.npy和relation_glove.max.npy，存储对应的100维glove embedding
生成concept_glove.max.txt和relation_glove.max.txt，存放subj/obj；rel出现过单词的次数字典，每一行是concept/relation    出现次数

# get concept and relation embeddings with frequency and vocab files
cd ../embeddings/
cd glove/
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.*.zip
cd ../
python glove_to_npy.py  
python create_embeddings_glove.py
```

### Concept Grounding
```
# concept grounding: core concept recognition (find mentioned concepts)
 
cd ../grounding/
python batched_grounding.py generate_bash "../datasets/csqa_new/train_rand_split.jsonl.statements"
bash cmd.sh
python batched_grounding.py combine "../datasets/csqa_new/train_rand_split.jsonl.statements"
python prune_qc.py ../datasets/csqa_new/train_rand_split.jsonl.statements.mcp

python batched_grounding.py generate_bash "../datasets/csqa_new/dev_rand_split.jsonl.statements"
bash cmd.sh
python batched_grounding.py combine "../datasets/csqa_new/dev_rand_split.jsonl.statements"
python prune_qc.py ../datasets/csqa_new/dev_rand_split.jsonl.statements.mcp

# python batched_grounding.py generate_bash "../datasets/csqa_new/test_rand_split.jsonl.statements"
# bash cmd.sh
# python batched_grounding.py combine "../datasets/csqa_new/test_rand_split.jsonl.statements"
```

#### Schema Graph Construction
```
cd ../pathfinder/
python graph_construction.py

python batched_pathfinding.py generate_bash "../datasets/csqa_new/train_rand_split.jsonl.statements.mcp"
bash cmd.sh
python batched_pathfinding.py combine "../datasets/csqa_new/train_rand_split.jsonl.statements.mcp"

python batched_pathfinding.py generate_bash "../datasets/csqa_new/dev_rand_split.jsonl.statements.mcp"
bash cmd.sh
python batched_pathfinding.py combine "../datasets/csqa_new/dev_rand_split.jsonl.statements.mcp"

# Pruning 

python path_scoring.py train
python path_scoring.py dev

python path_pruning.py train
python path_pruning.py dev

cd ../graph_generation
python graph_gen.py train
python graph_gen.py test
```

### Train KagNet based on extracted BERT embeddings
```
cd ../baselines/

bash train_csqa_bert.sh
python extract_csqa_bert.py --bert_model bert-large-uncased --do_eval --do_lower_case --data_dir ../datasets/csqa_new --eval_batch_size 60 --learning_rate 1e-4  --max_seq_length 70 --mlp_hidden_dim 16 --output_dir ./models/ --save_model_name bert_large_b60g4lr1e-4wd0.01wp0.1_1337 --epoch_id 1 --data_split_to_extract train_rand_split.jsonl --output_sentvec_file ../datasets/csqa_new/train_rand_split.jsonl.statements.finetuned.large --layer_id -1 
python extract_csqa_bert.py --bert_model bert-large-uncased --do_eval --do_lower_case --data_dir ../datasets/csqa_new --eval_batch_size 60 --learning_rate 1e-4  --max_seq_length 70 --mlp_hidden_dim 16 --output_dir ./models/ --save_model_name bert_large_b60g4lr1e-4wd0.01wp0.1_1337 --epoch_id 1 --data_split_to_extract dev_rand_split.jsonl --output_sentvec_file ../datasets/csqa_new/dev_rand_split.jsonl.statements.finetuned.large --layer_id -1

cd ../models/
python main.py

```

### Citation
```
@inproceedings{kagnet-emnlp19,
  author    = {Bill Yuchen Lin and Xinyue Chen and Jamin Chen and Xiang Ren},
  title     = {KagNet: Knowledge-Aware Graph Networks for Commonsense Reasoning.},
  booktitle = {Proceedings of EMNLP-IJCNLP},
  year      = {2019},
}
``` 
#### Remarks
Feel free to email yuchen[dot]lin[at]usc[dot]edu if you have any questions and need help.

