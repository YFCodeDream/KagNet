import configparser
import json
import numpy as np
import sys
from tqdm import tqdm


def load_glove_from_npy(glove_vec_path, glove_vocab_path):
    """
    返回{word: glove_embedding}的字典
    """
    vectors = np.load(glove_vec_path)
    with open(glove_vocab_path, "r", encoding="utf8") as f:
        vocab = [l.strip() for l in f.readlines()]

    assert (len(vectors) == len(vocab))

    glove_embeddings = {}
    for i in range(0, len(vectors)):
        glove_embeddings[vocab[i]] = vectors[i]
    print("Read " + str(len(glove_embeddings)) + " glove vectors.")
    return glove_embeddings


def weighted_average(avg, new, n):
    # TODO: maybe a better name for this function?
    return ((n - 1) / n) * avg + (new / n)


def max_pooling(old, new):
    # TODO: maybe a better name for this function?
    return np.maximum(old, new)


def write_embeddings_npy(embeddings, embeddings_cnt, npy_path, vocab_path):
    words = []
    vectors = []
    for key, vec in embeddings.items():
        words.append(key)
        vectors.append(vec)

    matrix = np.array(vectors, dtype="float32")
    print(matrix.shape)

    # 生成concept_glove.max.npy和relation_glove.max.npy，存储对应的100维glove embedding
    print("Writing embeddings matrix to " + npy_path, flush=True)
    np.save(npy_path, matrix)
    print("Finished writing embeddings matrix to " + npy_path, flush=True)

    print("Writing vocab file to " + vocab_path, flush=True)
    # 生成concept_glove.max.txt和relation_glove.max.txt，存放subj/obj；rel出现过单词的次数字典
    to_write = ["\t".join([w, str(embeddings_cnt[w])]) for w in words]
    with open(vocab_path, "w", encoding="utf8") as f:
        f.write("\n".join(to_write))
    print("Finished writing vocab file to " + vocab_path, flush=True)


def create_embeddings_glove(pooling="max", dim=100):
    print("Pooling: " + pooling)

    config = configparser.ConfigParser()
    config.read("paths.cfg")

    # ./triple_string/tp_str_corpus.json，之前生成的conceptnet的json文件
    with open(config["paths"]["triple_string_cpnet_json"], "r", encoding="utf8") as f:
        triple_str_json = json.load(f)
    print("Loaded " + str(len(triple_str_json)) + " triple strings.")

    glove_embeddings = load_glove_from_npy(config["paths"]["glove_vec_npy"], config["paths"]["glove_vocab"])
    print("Loaded glove.", flush=True)

    # 存储概念和关系的embedding
    concept_embeddings = {}
    concept_embeddings_cnt = {}
    rel_embeddings = {}
    rel_embeddings_cnt = {}

    for i in tqdm(range(len(triple_str_json))):
        """
        语料库 e.g. {"tid": 0, "rel": "antonym", "subj": "ab_extra", "obj": "ab_intra", "temp": "you do not want a #SUBJ#
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
        data = triple_str_json[i]

        words = data["string"].strip().split(" ")

        rel = data["rel"]
        subj_start = data["subj_start"]
        subj_end = data["subj_end"]
        obj_start = data["obj_start"]
        obj_end = data["obj_end"]

        # 取出subj和obj的单词
        subj_words = words[subj_start:subj_end]
        obj_words = words[obj_start:obj_end]

        subj = " ".join(subj_words)
        obj = " ".join(obj_words)

        # counting the frequency (only used for the avg pooling)
        if subj not in concept_embeddings:
            # 如果subj没有出现过，就新建键值
            concept_embeddings[subj] = np.zeros((dim,))
            concept_embeddings_cnt[subj] = 0
        # 计数字典对应位置加一
        concept_embeddings_cnt[subj] += 1

        # 对obj同理
        # subj和obj都加到concept_embeddings和concept_embeddings_cnt里
        if obj not in concept_embeddings:
            concept_embeddings[obj] = np.zeros((dim,))
            concept_embeddings_cnt[obj] = 0
        concept_embeddings_cnt[obj] += 1

        # 对rel同理
        # 加到rel_embeddings和rel_embeddings_cnt里
        if rel not in rel_embeddings:
            rel_embeddings[rel] = np.zeros((dim,))
            rel_embeddings_cnt[rel] = 0
        rel_embeddings_cnt[rel] += 1

        if pooling == "avg":
            # 将subj和obj出现的单词的glove embedding相加
            subj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in subj_words])
            obj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in obj_words])

            if rel in ["relatedto", "antonym"]:
                # Symmetric relation. 对称关系
                # 对称关系就用句子里除去subj和obj之外所有单词的glove embedding之和表示
                rel_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in words]) \
                                   - subj_encoding_sum - obj_encoding_sum
            else:
                # Asymmetrical relation. 不对称关系
                # 不对称关系就用obj减去subj表示
                rel_encoding_sum = obj_encoding_sum - subj_encoding_sum

            subj_len = subj_end - subj_start  # subj单词数
            obj_len = obj_end - obj_start  # obj单词数

            # 将subj，obj和rel的和平均一下
            subj_encoding = subj_encoding_sum / subj_len
            obj_encoding = obj_encoding_sum / obj_len
            rel_encoding = rel_encoding_sum / (len(words) - subj_len - obj_len)

            # 记录subj，obj和rel的embedding
            concept_embeddings[subj] = subj_encoding
            concept_embeddings[obj] = obj_encoding
            rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

        elif pooling == "max":
            # 默认走的max

            # 取subj和obj里每个单词的embedding的最大值
            subj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in subj_words], axis=0)
            obj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in obj_words], axis=0)

            mask_rel = []
            for j in range(len(words)):
                # 跳过subj和obj的单词序号
                if subj_start <= j < subj_end or obj_start <= j < obj_end:
                    continue
                mask_rel.append(j)
            rel_vecs = [glove_embeddings.get(words[i], np.zeros((dim,))) for i in mask_rel]
            # 句子里除去subj和obj之外所有单词的glove embedding的最大值
            rel_encoding = np.amax(rel_vecs, axis=0)

            # here it is actually avg over max for relation
            # 以最大值保留对应的embedding，均为100维
            concept_embeddings[subj] = max_pooling(concept_embeddings[subj], subj_encoding)
            concept_embeddings[obj] = max_pooling(concept_embeddings[obj], obj_encoding)
            rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

    print(str(len(concept_embeddings)) + " concept embeddings")
    print(str(len(rel_embeddings)) + " relation embeddings")

    write_embeddings_npy(concept_embeddings, concept_embeddings_cnt,
                         config["paths"]["concept_vec_npy_glove"] + "." + pooling,
                         config["paths"]["concept_vocab_glove"] + "." + pooling + ".txt")
    write_embeddings_npy(rel_embeddings, rel_embeddings_cnt,
                         config["paths"]["relation_vec_npy_glove"] + "." + pooling,
                         config["paths"]["relation_vocab_glove"] + "." + pooling + ".txt")


if __name__ == "__main__":
    create_embeddings_glove()
