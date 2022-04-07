import configparser
import itertools
import json
import pickle
import sys
import timeit

import networkx as nx
from tqdm import tqdm

split = sys.argv[1]

config = configparser.ConfigParser()
config.read("paths.cfg")

# 生成schema graph的文件存储地址
GRAPH_PATH = "../datasets/csqa_new/%s_rand_split.jsonl.statements.pruned.0.15.pnxg" % split

# 之前生成的文件地址
# PF_PATH存储修剪后的路径
# pf文件格式如下：
# [
#   [ # 一个statement对应的C_q和C_a信息
#     {
#        ac: C_a^i,
#        qc: C_q^j,
#        pf_res: [
#                  {
#                     path: [C_q^j, t_1, t_2, ..., C_a^i], # C_q^j到C_a^i的路径1，均以编码表示
#                     rel: [
#                           [1, 3, 4], # C_q^i到t_1涉及的关系
#                           [2, 5, 6, 9], # t_1到t_2涉及的关系
#                           ...,
#                         ]
#                  },
#                  {
#                     path: [C_q^j, t_3, t_4, ..., C_a^i], # C_q^j到C_a^i的路径2
#                     rel: [
#                           [3, 7, 8], # C_q^i到t_3涉及的关系
#                           [4, 8, 9, 11], # t_3到t_4涉及的关系
#                           ...,
#                         ]
#                  },
#                  ...,
#                ]
#      },
#    ...,
#   ],
#   ..., # 存放若干个statement的信息
# ]
PF_PATH = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.pruned.0.15.pickle" % split
# MCP_PATH存储C_q和C_a
# {"sent": s, "ans": a, "qc": list(question_concepts), "ac": list(answer_concepts)}
MCP_PATH = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp" % split

NUM_CHOICES = 5

cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None
mcp_data = None
pf_data = None


def load_resources():
    global concept2id, relation2id, id2relation, id2concept, mcp_data, pf_data, PF_PATH, MCP_PATH
    concept2id = {}
    id2concept = {}
    # 加载concept2id和id2concept
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    # 加载id2relation和relation2id
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")

    print("loading pf_data from %s" % PF_PATH)
    start_time = timeit.default_timer()
    # 加载pf路径数据
    with open(PF_PATH, "rb") as fi:
        pf_data = pickle.load(fi)
    print('\t Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))

    # 加载mcp概念对数据
    with open(MCP_PATH, "r") as f:
        mcp_data = json.load(f)


def load_cpnet():
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    # ../conceptnet/extract_cpnet.py生成的conceptnet提取文件
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def get_edge(src_concept, tgt_concept):
    """
    取出源节点和目标节点之间的relation，经过去重（这里的relation是数值化的结果）
    """
    global cpnet, concept2id, relation2id, id2relation, id2concept
    rel_list = cpnet[src_concept][tgt_concept]
    # tmp = [rel_list[item]["weight"] for item in rel_list]
    # s = tmp.index(min(tmp))
    # rel = rel_list[s]["rel"]
    return list(set([rel_list[item]["rel"] for item in rel_list]))


# plain graph generation
def plain_graph_generation(qcs, acs, paths, rels):
    """
    传入的是一个statement的信息
    """
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    # print("qcs", qcs)
    # print("acs", acs)
    # print("paths", paths)
    # print("rels", rels)

    graph = nx.Graph()

    # 遍历一个statement里所有的路径
    for index, p in enumerate(paths):
        # 遍历路径中的每一段
        for c_index in range(len(p) - 1):
            h = p[c_index]
            t = p[c_index + 1]
            # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(h, t, weight=1.0)

    # 遍历qcs中长度为2的子序列
    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        # 如果cpnet_simple里有这一对问题概念之间的联系
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        # 如果cpnet_simple里有这一对答案概念之间的联系
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        # 如果没有路径，就建立C_q到C_a的一一映射，rel全为-1
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    # 转换为json格式的字符串
    g_str = json.dumps(nx.node_link_data(g))
    return g_str


# relational graph generation
def relational_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    # print("qcs", qcs)
    # print("acs", acs)
    # print("paths", paths)
    # print("rels", rels)

    graph = nx.MultiDiGraph()
    for index, p in enumerate(paths):
        rel_list = rels[index]
        for c_index in range(len(p) - 1):
            h = p[c_index]
            t = p[c_index + 1]
            if graph.has_edge(h, t):
                existing_r_set = set([graph[h][t][r]["rel"] for r in graph[h][t]])
            else:
                existing_r_set = set()
            for r in rel_list[c_index]:
                # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
                # TODO: do we need to add both directions?
                if r in existing_r_set:
                    continue
                graph.add_edge(h, t, rel=r, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            rs = get_edge(qc1, qc2)
            for r in rs:
                graph.add_edge(qc1, qc2, rel=r, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            rs = get_edge(ac1, ac2)
            for r in rs:
                graph.add_edge(ac1, ac2, rel=r, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    g_str = json.dumps(nx.node_link_data(g))
    return g_str


def main():
    global pf_data, mcp_data
    global cpnet, concept2id, relation2id, id2relation, id2concept

    # 加载以上全局变量
    load_cpnet()
    load_resources()

    final_text = ""

    # 遍历所有statement
    for index, qa_pairs in tqdm(enumerate(pf_data), desc="Building Graphs", total=len(pf_data)):
        # print(mcp_data[index])
        # print(pf_data[index])
        # print(qa_pairs)

        statement_paths = []
        statement_rel_list = []

        # 遍历一个statement里的所有qa对
        for qa_idx, qas in enumerate(qa_pairs):
            if qas["pf_res"] is None:
                cur_paths = []
                cur_rels = []
            else:
                # qas["pf_res"]：取出一个qa对对应的所有路径
                # path: [C_q^j, t_1, t_2, ..., C_a^i], # C_q^j到C_a^i的路径1，均以编码表示
                cur_paths = [item["path"] for item in qas["pf_res"]]
                # rel: [
                #       [1, 3, 4], # C_q^i到t_1涉及的关系
                #       [2, 5, 6, 9], # t_1到t_2涉及的关系
                #       ...,
                #      ]
                cur_rels = [item["rel"] for item in qas["pf_res"]]

            # 注意这里是extend，没有statement个数这个维度了
            statement_paths.extend(cur_paths)
            statement_rel_list.extend(cur_rels)

        # 把每一个statement的C_q和C_a转换为顺序编码
        qcs = [concept2id[c] for c in mcp_data[index]["qc"]]
        acs = [concept2id[c] for c in mcp_data[index]["ac"]]

        gstr = plain_graph_generation(qcs=qcs, acs=acs,
                                      paths=statement_paths,
                                      rels=statement_rel_list)
        final_text += gstr + "\n"

    # 存储所有statement的图数据
    with open(GRAPH_PATH, 'w') as fw:
        fw.write(final_text)
    print("Write Graph Done: %s" % GRAPH_PATH)


main()
