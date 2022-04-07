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
import numpy as np

config = configparser.ConfigParser()
config.read("paths.cfg")

cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None


def load_resources():
    # concept2id存储{concept: 顺序编码}的键值对
    # id2concept存储{顺序编码: concept}的键值对
    # 17种关系
    # id2relation存储{顺序编码: relation}的键值对
    # relation2id存储{relation: 顺序编码}的键值对
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")


def load_cpnet():
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    # ../conceptnet/cpnet.graph
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        # 如果有weight属性，取出对应weight，否则置weight为1.0
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            # 如果已经有了这条边，把取出的weight加上
            cpnet_simple[u][v]['weight'] += w
        else:
            # 否则添加这条边，置权重weight为对应的值
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


# source and target is text
# noinspection PyTypeChecker
def find_paths(source, target, ifprint=False):
    """
    根据C_q和C_a查找路径，从cpnet_simple里
    """
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    # 找到C_q和C_a对应的顺序编码
    s = concept2id[source]
    t = concept2id[target]

    # try:
    #     lenth, path = nx.bidirectional_dijkstra(cpnet, source=s, target=t, weight="weight")
    #     # print(lenth)
    #     print(path)
    # except nx.NetworkXNoPath:
    #     print("no path")
    # paths = [path]

    # 保证C_q和C_a在节点里
    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        return
    # paths =

    # 记录所有路径
    all_path = []
    all_path_set = set()

    for max_len in range(1, 5):
        # 最大路径长度由1到4
        # 从cpnet_simple这个简化的无向图中选取源点为C_q，目标点为C_a，最大路径长度为max_len的路径
        for p in nx.all_simple_paths(cpnet_simple, source=s, target=t, cutoff=max_len):
            # 生成路径字符串
            path_str = "-".join([str(c) for c in p])
            if path_str not in all_path_set:
                # 如果还未记录该路径
                # 就将其添加到记录所有路径字符串的集合里
                all_path_set.add(path_str)
                # 添加路径到记录所有路径的列表里
                all_path.append(p)

            # 取前100个最短的路径
            if len(all_path) >= 100:  # top shortest 300 paths
                break
        if len(all_path) >= 100:  # top shortest 300 paths
            break

    # all_path = [[int(c) for c in p.split("-")] for p in list(set(["-".join([str(c) for c in p]) for p in all_path]))]
    # print(len(all_path))

    # 依据路径长度排序
    all_path.sort(key=len, reverse=False)

    pf_res = []
    for p in all_path:
        # 遍历每一条选出来的路径
        # print([id2concept[i] for i in p])
        rl = []
        for src in range(len(p) - 1):
            # 遍历每一条路径里的节点

            # 取出路径的每一段的源节点和目标节点
            src_concept = p[src]
            tgt_concept = p[src + 1]

            # 取出源节点和目标节点之间的relation，可能有多个，经过去重（这里的relation是数值化的结果，存在列表中）
            rel_list = get_edge(src_concept, tgt_concept)
            rl.append(rel_list)

            if ifprint:
                rel_list_str = []

                # 将rel_list里的关系编码通过id2relation转换为原本的关系描述
                for rel in rel_list:
                    if rel < len(id2relation):
                        rel_list_str.append(id2relation[rel])
                    else:
                        rel_list_str.append(id2relation[rel - len(id2relation)] + "*")

                # 打印出来
                print(id2concept[src_concept], "----[%s]---> " % ("/".join(rel_list_str)), end="")
                if src + 1 == len(p) - 1:
                    print(id2concept[tgt_concept], end="")
        if ifprint:
            print()

        # 将每一条路径及其包含的每一段中涉及的关系保存下来并返回
        # path: [c_i(q), t_1, t_2, ..., c_j(a)]
        # rel: [
        #       [1, 3, 4], 对应c_i(q)到t_1涉及的关系
        #       [2, 5, 6, 9], 对应t_1到t_2涉及的关系
        #       ...,
        #     ]
        pf_res.append({"path": p, "rel": rl})
    return pf_res


def process(filename, batch_id=-1):
    pf = []
    output_path = filename + ".%d" % batch_id + ".pf"
    import os
    if os.path.exists(output_path):
        print(output_path + " exists. Skip!")
        return

    load_resources()
    # concept2id存储{concept: 顺序编码}的键值对
    # id2concept存储{顺序编码: concept}的键值对
    # 17种关系
    # id2relation存储{顺序编码: relation}的键值对
    # relation2id存储{relation: 顺序编码}的键值对

    load_cpnet()
    # cpnet存储graph_construction.py生成的conceptnet的图
    # cpnet_simple存储cpnet处理后的不含平行边的无向图（合并了平行边）
    with open(filename, 'r') as fp:
        # 读取之前生成的mcp总数据文件
        # 每一个问题对应五个statement，每一个statement提取C_q，C_a如下：
        # {"sent": s, "ans": a, "qc": list(question_concepts), "ac": list(answer_concepts)}
        mcp_data = json.load(fp)

        # 分成100组，取第batch_id组
        mcp_data = list(np.array_split(mcp_data, 100)[batch_id])

        # 遍历每一个statement
        # 每一个statement对应一个pfr_qa列表
        # pfr_qa存储statement里由每一组C_q到C_a的路径查找的结果
        for item in tqdm(mcp_data, desc="batch_id: %d " % batch_id):
            # 取出C_q和C_a
            acs = item["ac"]
            qcs = item["qc"]

            # 存储路径查找的结果
            pfr_qa = []  # path finding results
            for ac in acs:
                for qc in qcs:
                    # 获取由C_q到C_a的路径结果
                    # pf_res也是一个列表，每一个元素是字典，存储路径及其每一段涉及的关系
                    pf_res = find_paths(qc, ac)
                    pfr_qa.append({"ac": ac, "qc": qc, "pf_res": pf_res})
            pf.append(pfr_qa)

    # 写入该分组的pf文件
    # 最后每一个分组的pf文件应该为
    # [
    #   [ # 一个statement对应的C_q和C_a信息
    #     {
    #        ac: C_a^i,
    #        qc: C_q^j,
    #        pf_res: [
    #                  {
    #                     path: [C_q^j, t_1, t_2, ..., C_a^i], # C_q^j到C_a^i的路径1
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
    with open(output_path, 'w') as fi:
        json.dump(pf, fi)


process(sys.argv[1], int(sys.argv[2]))
#

# load_resources()
# load_cpnet()
# # find_paths("fill", "fountain_pen", ifprint=True)
# # print("--------")
# # find_paths("write", "fountain_pen", ifprint=True)
# # print("--------")
# # find_paths("write", "pen", ifprint=True)
# find_paths("bottle", "liquor", ifprint=True)
#
# print()
# print()
# print()
# print()
# print()
#
# find_paths("cashier", "store", ifprint=True)
