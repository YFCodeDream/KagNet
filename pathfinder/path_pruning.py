import pickle
from tqdm import tqdm

import sys

flag = sys.argv[1]
threshold = 0.15
ori_pickle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.pickle" % flag
scores_pickle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.scores.pickle" % flag
pruned_pickle_file = "../datasets/csqa_new/%s_rand_split.jsonl.statements.mcp.pf.cls.pruned.%s.pickle" % (
    flag, str(threshold))

# threshold = 0.75
# threshold = 0.15

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
ori_paths = []
with open(ori_pickle_file, "rb") as fi:
    ori_paths = pickle.load(fi)

# scores文件格式如下：
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
all_scores = []
with open(scores_pickle_file, "rb") as fi:
    all_scores = pickle.load(fi)

assert len(ori_paths) == len(all_scores)

ori_len = 0
pruned_len = 0

# 遍历所有statement
for index, qa_pairs in tqdm(enumerate(ori_paths[:]), desc="Scoring the paths", total=len(ori_paths)):
    # 遍历一个statement里的所有qa对
    for qa_idx, qas in enumerate(qa_pairs):
        # 取出一个qa对对应的所有路径
        statement_paths = qas["pf_res"]
        if statement_paths is not None:
            pruned_statement_paths = []
            # 遍历该qa对的所有路径
            for pf_idx, item in enumerate(statement_paths):
                # 取出每一条路径的评分
                score = all_scores[index][qa_idx][pf_idx]

                # 如果评分大于阈值，就保存
                if score >= threshold:
                    pruned_statement_paths.append(item)

            # ori_len存储原来的路径条数
            ori_len += len(ori_paths[index][qa_idx]["pf_res"])
            # pruned_len存储修剪后的路径条数
            pruned_len += len(pruned_statement_paths)

            assert len(ori_paths[index][qa_idx]["pf_res"]) >= len(pruned_statement_paths)

            # 修改保存的路径
            ori_paths[index][qa_idx]["pf_res"] = pruned_statement_paths

print("ori_len:", ori_len, "\t\tafter_pruned_len:", pruned_len, "keep rate: %.4f" % (pruned_len / ori_len))
print("saving the pruned paths")

# 保存修剪后的路径
with open(pruned_pickle_file, 'wb') as fp:
    pickle.dump(ori_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)
print("done!")
