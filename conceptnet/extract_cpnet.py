import configparser
import json

relation_mapping = dict()


def load_merge_relation():
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    # 读取paths.cfg里的merge_relation.txt里存的关系种类
    with open(config["paths"]["merge_relation"], encoding="utf8") as f:
        for line in f.readlines():
            # line存放同一类的关系，ls是存放同一类关系的列表
            ls = line.strip().split('/')
            # 取第一个关系
            rel = ls[0]
            # 对于某一类型的关系列表进行遍历
            for l in ls:
                # relation_mapping存放merge_relation.txt出现的所有关系到17种关系的映射
                # e.g. atlocation：atlocation；locatednear：atlocation；motivatedbygoal：causes
                if l.startswith("*"):
                    relation_mapping[l[1:]] = "*" + rel
                else:
                    relation_mapping[l] = rel


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    从实体字符串（如果存在）中删除词性编码。
    参数s：实体字符串。
    ：return：删除词性编码的实体字符串。
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_english():
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    读取原始conceptnet csv文件，并将所有英文关系（头和尾都是英文实体）提取到一个新文件，
    每行的格式如下：<relation><head><tail><weight>。
    """
    config = configparser.ConfigParser()
    config.read("paths.cfg")

    only_english = []
    with open(config["paths"]["conceptnet"], encoding="utf8") as f:
        for line in f.readlines():
            ls = line.split('\t')
            # 筛选出英文实体（en是English）
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                一些预处理：
                    -删除词性编码。
                    -Split("/")[-1]以修剪“/c/en/”并只获取实体名称，将所有转换为
                    -小写表示一致性。
                """
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()

                # 不是纯字母的实体舍去
                if not head.replace("_", "").replace("-", "").isalpha():
                    continue

                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue

                # 关系必须包含在relation_mapping的键值里
                if rel not in relation_mapping:
                    continue
                rel = relation_mapping[rel]
                # 如果关系前有*，则将head和tail反转
                if rel.startswith("*"):
                    rel = rel[1:]
                    tmp = head
                    head = tail
                    tail = tmp

                data = json.loads(ls[4])

                only_english.append("\t".join([rel, head, tail, str(data["weight"])]))

    with open(config["paths"]["conceptnet_en"], "w", encoding="utf8") as f:
        f.write("\n".join(only_english))


if __name__ == "__main__":
    load_merge_relation()
    print(relation_mapping)
    extract_english()
