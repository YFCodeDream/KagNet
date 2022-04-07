import pickle
import sys
import json
import random

path_csqa_train = "../datasets/csqa_new/train_rand_split.jsonl.statements"
path_csqa_dev = "../datasets/csqa_new/dev_rand_split.jsonl.statements"
path_csqa_test = "../datasets/csqa_new/test_rand_split.jsonl.statements"  # rename test file first

PATH = path_csqa_train  # switch mannually
NUM_BATCHES = 100


def generate_bash():
    PATH = sys.argv[2]
    with open("cmd_lucy.sh", 'w') as f:
        for i in range(0, 50):
            f.write("CUDA_VISIBLE_DEVICES=NONE python pathfinder.py %s %d &\n" % (PATH, i))
        f.write('wait')

    with open("cmd_ron.sh", 'w') as f:
        for i in range(50, 80):
            f.write("CUDA_VISIBLE_DEVICES=NONE python pathfinder.py %s %d &\n" % (PATH, i))
        f.write('wait')

    with open("cmd_molly.sh", 'w') as f:
        for i in range(80, 100):
            f.write("CUDA_VISIBLE_DEVICES=NONE python pathfinder.py %s %d &\n" % (PATH, i))
        f.write('wait')


def combine():
    final_json = []
    PATH = sys.argv[2]
    for i in range(NUM_BATCHES):
        with open(PATH + ".%d.pf" % i) as fp:
            tmp_list = json.load(fp)
        final_json += tmp_list

    # 汇总每一组pf文件，生成包含所有数据的pf文件
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
    with open(PATH + ".pf", 'w') as fp:
        json.dump(final_json, fp)  # js beautify is too time-consuming now

    # with open(PATH + ".pf.pickle", 'wb') as fp:
    #     pickle.dump(final_json, fp)


if __name__ == '__main__':
    import sys

    globals()[sys.argv[1]]()
