import configparser
import numpy as np


def parse_glove(path, dim=300):
    # path是./glove/glove.6B.100d.txt
    print("Parsing glove", flush=True)
    glove_embeddings = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f.readlines():
            # 每一行是一个单词的glove embedding，第一个是单词，之后的是embedding
            elements = line.strip().split(" ")
            word = elements[0]
            vec = np.array(elements[1:]).astype(np.float64)
            glove_embeddings[word] = vec
        print("Read " + str(len(glove_embeddings)) + " glove embeddings.", flush=True)
        # 最终返回{word: glove_embedding}的字典
        return glove_embeddings


def glove_to_npy():
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    glove_embeddings = parse_glove(config["paths"]["glove"])

    words = []
    vectors = []

    for key, vec in glove_embeddings.items():
        words.append(key)
        vectors.append(vec)

    print("Writing glove vectors.", flush=True)
    # ./glove/glove_vec.npy
    # vectors就是每个单词对应的glove embedding向量
    np.save(config["paths"]["glove_vec_npy"], vectors)
    print("Finished writing glove vectors.", flush=True)

    vocab_text = "\n".join(words)

    print("Writing glove vocab.", flush=True)
    # ./glove/glove_vocab.txt
    # 把glove embedding出现的单词都收录到一个词典中
    with open(config["paths"]["glove_vocab"], "wb") as f:
        f.write(vocab_text.encode("utf-8"))
    print("Finished writing glove vocab.", flush=True)


if __name__ == "__main__":
    glove_to_npy()
