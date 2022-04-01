import configparser
import json
import random

random.seed(42)
template = {}
config = configparser.ConfigParser()
config.read("paths.cfg")


def load_templates():
    """
    这个函数就是用来生成template字典的
    """
    # 打开templates.txt
    with open(config["paths"]["tp_str_template"], encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            # 带有[的都是之前定义的17种关系
            if line.startswith("["):
                # 与之前extract_cpnet.py里的relation_mapping对应，取第一个关系
                rel = line.split('/')[0][1:]
                if rel.endswith(']'):
                    rel = rel[:-1]
                # 每一个关系都对应若干语句模板，存放于列表
                template[rel] = []
            # 含有#SUBJ#和#OBJ#就是模板语句
            # #SUBJ#是主语，#OBJ#是宾语
            elif "#SUBJ#" in line and "#OBJ#" in line:
                template[rel].append(line)


def generate_triple_string(tid, subj, rel, obj):
    # 从template的指定relation的模板列表中随便选一个语句模板
    temp = random.choice(template[rel])
    tp_str = {"tid": tid, "rel": rel, "subj": subj, "obj": obj, "temp": temp}

    # 将subj和obj的下划线换成空格
    subj = subj.replace('_', ' ')
    obj = obj.replace('_', ' ')
    # 将模板中#SUBJ#和#OBJ#占位符换成实际的实体
    tp_str["string"] = temp.replace("#SUBJ#", subj).replace("#OBJ#", obj)

    subj_lst = subj.split()
    obj_lst = obj.split()
    tp_str_lst = tp_str["string"].split()
    tp_str["subj_start"] = 0

    # print(subj, rel, obj, tp_str["string"])

    while tp_str_lst[tp_str["subj_start"]: tp_str["subj_start"] + len(subj_lst)] != subj_lst:
        tp_str["subj_start"] += 1
    tp_str["subj_end"] = tp_str["subj_start"] + len(subj_lst)

    tp_str["obj_start"] = 0
    while tp_str_lst[tp_str["obj_start"]: tp_str["obj_start"] + len(obj_lst)] != obj_lst:
        tp_str["obj_start"] += 1
    tp_str["obj_end"] = tp_str["obj_start"] + len(obj_lst)
    return tp_str


def create_corpus():
    """
    创建语料库 e.g. {"tid": 0, "rel": "antonym", "subj": "ab_extra", "obj": "ab_intra", "temp": "you do not want a #SUBJ#
    with #OBJ#", "string": "you do not want a ab extra with ab intra", "subj_start": 5, "subj_end": 7, "obj_start":
    8, "obj_end": 10}
    tid: conceptnet英文实体三元组编号
    rel: 定义的17种关系
    subj: head实体
    obj: tail实体
    temp: 模板
    string: 将模板中#SUBJ#和#OBJ#占位符换成实际的实体之后的句子
    subj_start: head实体的起始下标（在句子中第几个单词）
    subj_end: head实体的结束下标
    obj_start和obj_end同理
    """
    corpus = []
    with open(config["paths"]["conceptnet_en"], "r", encoding="utf8") as f:
        for line in f.readlines():
            ls = line.strip().split('\t')
            rel = ls[0]
            head = ls[1]
            tail = ls[2]
            tp_str = generate_triple_string(len(corpus), head, rel, tail)
            corpus.append(tp_str)
    # tp_str_corpus.json
    with open(config["paths"]["tp_str_corpus"], "w", encoding="utf8") as f:
        print("Writing Json File....")
        json.dump(corpus, f)


if __name__ == "__main__":
    load_templates()
    # print(template)
    # tp_str = generate_triple_string(1,"love","antonym","hate")
    create_corpus()
