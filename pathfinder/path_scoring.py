import pickle
import json

from tqdm import tqdm

import configparser

from scipy import spatial
import numpy as np
import os
from os import path
import sys
import random

# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# from embeddings.TransE import *

config = configparser.ConfigParser()
config.read("paths.cfg")

cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None

concept_embs = None
relation_embs = None
mcp_py_filename = None


# def test():
#     global id2concept, id2relation
#     init_predict(2,5,2)
#     print(id2concept[2])
#     print(id2concept[5])
#     print(id2rel[2])


def load_resources(method):
    global concept2id, id2concept, concept_embs, relation2id, id2relation, relation_embs
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")

    # 加载concept embedding
    # concept_embs的shape: (799273, 100)
    concept_embs = np.load("../embeddings/openke_data/embs/glove_initialized/glove.transe.sgd.ent.npy")

    print("concept_embs done")

    if method == "triple_cls":
        # 运行这个if
        relation2id = {}
        id2relation = {}

        with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
            for w in f.readlines():
                relation2id[w.strip()] = len(relation2id)
                id2relation[len(id2relation)] = w.strip()

        print("relation2id done")

        # 加载relation embedding
        relation_embs = np.load("../embeddings/openke_data/embs/glove_initialized/glove.transe.sgd.rel.npy")

        print("relation_embs done")

    return


def vanila_score_triple(h, t, r):
    # return np.linalg.norm(t-h-r)
    return (1 + 1 - spatial.distance.cosine(r, t - h)) / 2


def vanila_score_triples(concept_id, relation_id):
    global relation_embs, concept_embs, id2relation, id2concept

    concept = concept_embs[concept_id]
    relation = []

    flag = []
    for i in range(len(relation_id)):

        embs = []
        l_flag = []

        if 0 in relation_id[i] and 17 not in relation_id[i]:
            relation_id[i].append(17)
        elif 17 in relation_id[i] and 0 not in relation_id[i]:
            relation_id[i].append(0)

        if 15 in relation_id[i] and 32 not in relation_id[i]:
            relation_id[i].append(32)
        elif 32 in relation_id[i] and 15 not in relation_id[i]:
            relation_id[i].append(15)

        for j in range(len(relation_id[i])):

            if relation_id[i][j] >= 17:

                score = vanila_score_triple(concept[i + 1], concept[i], relation_embs[relation_id[i][j] - 17])

                print("%s\tr-%s\t%s" % (
                    id2concept[concept_id[i]], id2relation[relation_id[i][j] - 17], id2concept[concept_id[i + 1]]))
                print("Likelihood: " + str(score) + "\n")



            else:

                score = vanila_score_triple(concept[i], concept[i + 1], relation_embs[relation_id[i][j]])

                print("%s\t%s\t%s" % (
                    id2concept[concept_id[i]], id2relation[relation_id[i][j]], id2concept[concept_id[i + 1]]))
                print("Likelihood: " + str(score) + "\n")


def score_triple(h, t, r, flag):
    res = -10

    for i in range(len(r)):
        if flag[i]:
            temp_h, temp_t = t, h
        else:
            temp_h, temp_t = h, t

        # result  = (cosine_sim(路径第i段的第j个关系的embedding, head概念embedding减去tail概念embedding的结果) + 1) / 2
        res = max(res, (1 + 1 - spatial.distance.cosine(r[i], temp_t - temp_h)) / 2)

    # 返回最大的关系计算结果
    return res


def score_triples(concept_id, relation_id, debug=False):
    # item为：
    #   {
    #      path: [C_q^j, t_1, t_2, ..., C_a^i], # C_q^j到C_a^i的路径1
    #      rel: [
    #            [1, 3, 4], # C_q^i到t_1涉及的关系
    #            [2, 5, 6, 9], # t_1到t_2涉及的关系
    #            ...,
    #          ]
    #   }

    # concept_id = item["path"], relation_id = item["rel"]

    global relation_embs, concept_embs, id2relation, id2concept

    # 依据路径中concept的顺序编码，取到对应的embedding
    concept = concept_embs[concept_id]

    relation = []
    flag = []

    for i in range(len(relation_id)):
        # relation_id是个二维列表
        # 每一个元素是路径中第i段涉及的若干关系
        embs = []
        l_flag = []

        # 对antonym（反义）和relatedto（相关）保留两份关系
        if 0 in relation_id[i] and 17 not in relation_id[i]:
            relation_id[i].append(17)
        elif 17 in relation_id[i] and 0 not in relation_id[i]:
            relation_id[i].append(0)

        if 15 in relation_id[i] and 32 not in relation_id[i]:
            relation_id[i].append(32)
        elif 32 in relation_id[i] and 15 not in relation_id[i]:
            relation_id[i].append(15)

        # 遍历第i段路径涉及的关系
        for j in range(len(relation_id[i])):
            # 保留两份关系embedding在这里实现
            if relation_id[i][j] >= 17:
                embs.append(relation_embs[relation_id[i][j] - 17])
                # 同时用标志数组记录
                l_flag.append(1)

            else:
                embs.append(relation_embs[relation_id[i][j]])
                l_flag.append(0)

        relation.append(embs)
        flag.append(l_flag)

    # 至此relation是个三维列表，第0维是表示路径有n段，第1维表示路径第i段有m个关系，第2维表示第j个关系的embedding

    res = 1

    for i in range(concept.shape[0] - 1):
        # 取出每一段head概念和tail概念
        h = concept[i]
        t = concept[i + 1]
        score = score_triple(h, t, relation[i], flag[i])

        # 路径每一段的评分相乘作为最终的路径评判分数
        res *= score

    if debug:
        print("Num of concepts:")
        print(len(concept_id))

        to_print = ""

        for i in range(concept.shape[0] - 1):

            h = id2concept[concept_id[i]]

            to_print += h + "\t"
            for rel in relation_id[i]:
                if rel >= 17:

                    # 'r-' means reverse
                    to_print += ("r-" + id2relation[rel - 17] + "/  ")
                else:
                    to_print += id2relation[rel] + "/  "

        to_print += id2concept[concept_id[-1]]
        print(to_print)

        print("Likelihood: " + str(res) + "\n")

    return res


def context_per_qa(acs, qcs, pooling="mean"):
    '''
    calculate the context embedding for each q-a statement in terms of mentioned concepts
    '''

    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple, concept_embs
    for i in range(len(acs)):
        acs[i] = concept2id[acs[i]]

    for i in range(len(qcs)):
        qcs[i] = concept2id[qcs[i]]

    concept_ids = np.asarray(list(set(qcs) | set(acs)), dtype=int)
    concept_context_emb = np.mean(concept_embs[concept_ids], axis=0) if pooling == "mean" else np.maximum(
        concept_embs[concept_ids])

    return concept_context_emb


def path_scoring(path, context):
    global concept_embs

    path_concepts = concept_embs[path]

    # cosine distance, the smaller the more alike

    cosine_dist = np.apply_along_axis(spatial.distance.cosine, 1, path_concepts, context)
    cosine_sim = 1 - cosine_dist
    if len(path) > 2:
        return min(cosine_sim[1:-1])  # the minimum of the cos sim of the middle concepts
    else:
        return 1.0  # the source and target of the paths are qa concepts


def calc_context_emb(pooling="mean", filename=""):
    global mcp_py_filename
    mcp_py_filename = filename + "." + pooling + ".npy"
    if os.path.exists(mcp_py_filename):
        print(mcp_py_filename, "exists!")
        return

    with open(filename, "rb") as f:
        mcp = json.load(f)

    embs = []

    for s in tqdm(mcp, desc="Computing concept-context embedding.."):
        qcs = s["qc"]
        acs = s["ac"]

        embs.append(context_per_qa(acs=acs, qcs=qcs, pooling=pooling))

    embs = np.asarray(embs)
    print("output_path: " + mcp_py_filename)
    np.save(mcp_py_filename, embs)


def score_paths(filename, score_filename, method, debug=False, debug_range=None):
    global id2concept, mcp_py_filename

    print("Loading paths")

    # input里存有pathfinder.py生成的路径信息
    # [
    #   [ # 一个statement对应的C_q和C_a信息
    #     {
    #        ac: C_a^i,
    #        qc: C_q^j,
    #        pf_res: [
    #                  {
    #                     p: [C_q^j, t_1, t_2, ..., C_a^i], # C_q^j到C_a^i的路径1
    #                     rl: [
    #                           [1, 3, 4], # C_q^i到t_1涉及的关系
    #                           [2, 5, 6, 9], # t_1到t_2涉及的关系
    #                           ...,
    #                         ]
    #                  },
    #                  {
    #                     p: [C_q^j, t_3, t_4, ..., C_a^i], # C_q^j到C_a^i的路径2
    #                     rl: [
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
    with open(filename, "rb") as f:
        input = pickle.load(f)

    print("Paths loaded")

    if not method == "triple_cls":
        # 不执行这个if
        print("Loading context embeddings")

        context_embs = np.load(mcp_py_filename)

        print("Loaded")

    all_scores = []

    if debug:
        a, b = debug_range
        input = input[a:b]
    else:
        pass

    for index, qa_pairs in tqdm(enumerate(input), desc="Scoring the paths", total=len(input)):
        # 遍历每一个statement的信息
        statement_scores = []
        for qa_idx, qas in enumerate(qa_pairs):
            # 遍历每一个statement里的C_q和C_a对及其路径信息
            statement_paths = qas["pf_res"]
            # statement_paths为：
            # [
            #   {
            #      path: [C_q^j, t_1, t_2, ..., C_a^i], # C_q^j到C_a^i的路径1
            #      rel: [
            #            [1, 3, 4], # C_q^i到t_1涉及的关系
            #            [2, 5, 6, 9], # t_1到t_2涉及的关系
            #            ...,
            #          ]
            #   },
            #   {
            #      path: [C_q^j, t_3, t_4, ..., C_a^i], # C_q^j到C_a^i的路径2
            #      rel: [
            #            [3, 7, 8], # C_q^i到t_3涉及的关系
            #            [4, 8, 9, 11], # t_3到t_4涉及的关系
            #            ...,
            #          ]
            #   },
            #   ...,
            # ]
            if statement_paths is not None:
                if not method == "triple_cls":
                    context_emb = context_embs[index]

                path_scores = []
                for pf_idx, item in enumerate(statement_paths):
                    # item为：
                    #   {
                    #      path: [C_q^j, t_1, t_2, ..., C_a^i], # C_q^j到C_a^i的路径1
                    #      rel: [
                    #            [1, 3, 4], # C_q^i到t_1涉及的关系
                    #            [2, 5, 6, 9], # t_1到t_2涉及的关系
                    #            ...,
                    #          ]
                    #   }
                    assert len(item["path"]) > 1
                    # vanila_score_triples(concept_id=item["path"], relation_id=item["rel"])

                    if not method == "triple_cls":
                        score = path_scoring(path=item["path"], context=context_emb)

                    else:
                        score = score_triples(concept_id=item["path"], relation_id=item["rel"], debug=debug)

                    # path_scores存储每一条路径的score
                    path_scores.append(score)

                # statement_scores存储每一个statement中多条C_q到C_a的路径分数
                statement_scores.append(path_scores)
            else:
                statement_scores.append(None)

        all_scores.append(statement_scores)

    if not debug:
        print("saving the path scores")

        # 将all_scores写入文件
        # 文件格式如下：
        #  [
        #     [ # 一个statement
        #       [ # 一个qa对
        #         score_1, # 第一条路径：C_q^1-C_a^1
        #         score_2, # 第二条路径：C_q^1-C_a^1
        #         ...,
        #       ],
        #       [
        #         score_3, # 第一条路经：C_q^1-C_a^2
        #         score_4, # 第二条路经：C_q^1-C_a^2
        #         ...,
        #       ],
        #       ...,
        #     ]
        #  ]
        with open(score_filename, 'wb') as fp:
            pickle.dump(all_scores, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("done!")


if __name__ == "__main__":
    import sys

    flag = sys.argv[1]
    method = "triple_cls"  #
    mcp_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp" % flag
    ori_pickle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.pickle" % flag
    scores_pickle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.scores.pickle" % flag

    '''to calculate the context embedding for qas'''

    load_resources(method=method)

    if not method == "triple_cls":
        calc_context_emb(filename=mcp_file)
    # score_paths(filename=ori_pckle_file, score_filename=scores_pckle_file, method=method, debug=True, debug_range=(
    # 10, 11))

    score_paths(filename=ori_pickle_file, score_filename=scores_pickle_file, method=method, debug=False)

    # score_paths(filename=ori_pckle_file, score_filename=scores_pckle_file, method=method, debug=True,
    #                 debug_range=(11, 12))
