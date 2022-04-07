import configparser
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import nltk
import json

# print('NLTK Version: %s' % (nltk.__version__))
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may", "wanter"]

config = configparser.ConfigParser()
config.read("paths.cfg")

cpnet = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None
blacklist = {"uk", "us", "take", "make", "object", "person", "people"}


def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    # ../embeddings/concept.txt
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            # concept2id存储{concept: 顺序编码}的键值对
            concept2id[w.strip()] = len(concept2id)
            # id2concept存储{顺序编码: concept}的键值对
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    # ../embeddings/relation.txt
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            # 17种关系
            # id2relation存储{顺序编码: relation}的键值对
            id2relation[len(id2relation)] = w.strip()
            # relation2id存储{relation: 顺序编码}的键值对
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")


def save_cpnet():
    global concept2id, relation2id, id2relation, id2concept, blacklist
    load_resources()
    # load_resource初始化完concept2id, relation2id, id2relation, id2concept

    # 这里的graph是MultiDiGraph，具有自环和平行边的有向图
    graph = nx.MultiDiGraph()
    # ../conceptnet/conceptnet-assertions-5.6.0.csv.en
    # 经处理后的conceptnet文件，每行的格式如下：<relation><head><tail><weight>
    with open(config["paths"]["conceptnet_en"], "r", encoding="utf8") as f:
        lines = f.readlines()

        def not_save(cpt):
            # 如果concept在黑名单里
            if cpt in blacklist:
                return True
            for t in cpt.split("_"):
                # 如果concept包含停用词
                if t in nltk_stopwords:
                    return True
            return False

        for line in tqdm(lines, desc="saving to graph"):
            ls = line.strip().split('\t')

            # 从../conceptnet/conceptnet-assertions-5.6.0.csv.en的每一行取对应信息
            # rel，subj，obj都是对应的顺序编码
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])

            # 舍弃"hascontext"
            if id2relation[rel] == "hascontext":
                continue

            # 对subj和obj的概念进行判断，是否在黑名单或者停用词里
            if not_save(ls[1]) or not_save(ls[2]):
                continue

            # 相关 和 相反 两种关系，设置weight减去0.3
            if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
                weight -= 0.3
                # continue
            if subj == obj:  # delete loops
                continue

            # 转换一下权值
            weight = 1 + float(math.exp(1 - weight))

            # 添加边subj -> obj(rel=rel, weight=weight); obj -> subj(rel=rel+17, weight=weight)
            graph.add_edge(subj, obj, rel=rel, weight=weight)
            graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)

    # ../conceptnet/cpnet.graph
    nx.write_gpickle(graph, config["paths"]["conceptnet_en_graph"])
    # with open(config["paths"]["conceptnet_en_graph"], 'w') as f:
    #     f.write(json.dumps(nx.node_link_data(graph)))


save_cpnet()
