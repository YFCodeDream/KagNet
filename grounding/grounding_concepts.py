"""
批量执行这个脚本
python grounding_concepts.py ../datasets/csqa_new/train_rand_split.jsonl.statements （0到50，50到80，80到100）
"""
import configparser
import json
import sys

import numpy as np
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm

blacklist = {"-PRON-", "actually", "likely", "possibly", "want", "make", "my", "someone", "sometimes_people",
             "sometimes", "would", "want_to", "one", "something", "sometimes", "everybody", "somebody", "could",
             "could_be"}

concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")
# ../embeddings/concept.txt
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
# 将conceptnet里每个concept的下划线换成空格
cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]


def lemmatize(nlp, concept):
    """
    把concept中所有的词进行词根化
    同时利用set去重
    """
    doc = nlp(concept.replace("_", " "))
    lcs = set()
    # for i in range(len(doc)):
    #     lemmas = []
    #     for j, token in enumerate(doc):
    #         if j == i:
    #             lemmas.append(token.lemma_)
    #         else:
    #             lemmas.append(token.text)
    #     lc = "_".join(lemmas)
    #     lcs.add(lc)
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    # matcher_patterns.json
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    # 通过LEMMA规则，将所有词全部词根化，e.g. 现在分词，过去分词全部转换为原形
    # 词根化的词均是conceptnet里的词
    matcher = Matcher(nlp.vocab)
    for concept, pattern in tqdm(all_patterns.items(), desc="Adding patterns to Matcher."):
        matcher.add(concept, None, pattern)
    return matcher


def ground_mentioned_concepts(nlp, matcher, s, ans=""):
    # s和a分别为各自的statement和text
    # e.g.
    # s: The sanctions against the school were a punishing blow, and they seemed to ignore the efforts the school had made to change
    # a: ignore
    # 将statement全部小写化
    s = s.lower()
    doc = nlp(s)
    # matcher是通过matcher_patterns.json里定义的LEMMA规则的匹配器
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    # 遍历所有的匹配结果
    for match_id, start, end in matches:
        # 选出statement里出现的conceptnet里的concept，记录在span里
        span = doc[start:end].text  # the matched span
        # 舍弃与答案无相同单词的concept
        if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            continue
        # 取出原有conceptnet里的concept
        original_concept = nlp.vocab.strings[match_id]
        # print("Matched '" + span + "' to the rule '" + string_id)

        if len(original_concept.split("_")) == 1:
            # 对没有下划线的concept进行处理，将其词根化
            original_concept = list(lemmatize(nlp, original_concept))[0]

        # span是与答案有相同单词的concept
        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        # 建立与答案有相同单词的，conceptnet里的concept，与原有conceptnet里的concept（可能有多个）的映射
        span_to_concepts[span].add(original_concept)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])
        # 选取长度最短的三个概念
        shortest = concepts_sorted[0:3]  #
        for c in shortest:
            # 排除黑名单里的概念
            if c in blacklist:
                continue
            # 词根化
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                # 如果原本就是词根化的词，就添加词根化的词
                mentioned_concepts.add(list(intersect)[0])
            else:
                # 否则就直接把c添加进去
                mentioned_concepts.add(c)

    # stop = timeit.default_timer()
    # print('\t Done! Time: ', "{0:.2f} sec".format(float(stop - start_time)))

    # if __name__ == "__main__":
    #     print("Sentence: " + s)
    #     print(mentioned_concepts)
    #     print()

    # 提取出statement里涉及的概念
    return mentioned_concepts


def hard_ground(nlp, sent):
    global cpnet_vocab
    # 小写
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        # 将statement的token词根化，判断是不是在conceptnet里
        if t.lemma_ in cpnet_vocab:
            # 在就加到res集合里
            res.add(t.lemma_)
    sent = "_".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    return res


def match_mentioned_concepts(nlp, sents, answers, batch_id=-1):
    matcher = load_matcher(nlp)

    res = []
    # print("Begin matching concepts.")
    # 依次遍历所有的statement
    for sid, s in tqdm(enumerate(sents), total=len(sents), desc="grounding batch_id:%d" % batch_id):
        a = answers[sid]
        # s和a分别为各自的statement和text
        # e.g.
        # s: The sanctions against the school were a punishing blow, and they seemed to ignore the efforts the school had made to change
        # a: ignore

        # 这里取出C_q和C_a
        all_concepts = ground_mentioned_concepts(nlp, matcher, s, a)
        answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
        question_concepts = all_concepts - answer_concepts

        # 如果没检测到问题概念C_q
        if len(question_concepts) == 0:
            # print(s)
            question_concepts = hard_ground(nlp, s)  # not very possible
        if len(answer_concepts) == 0:
            print(a)
            answer_concepts = hard_ground(nlp, a)  # some case
            print(answer_concepts)

        res.append({"sent": s, "ans": a, "qc": list(question_concepts), "ac": list(answer_concepts)})
    return res


def process(filename, batch_id=-1):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    # sentencizer将文章切分成句子，
    # 原理是Spacy通过将文章中某些单词的is_sent_start属性设置为True，来实现对文章的句子的切分，这些特殊的单词在规则上对应于句子的开头
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    sents = []
    answers = []

    # filename传进来../datasets/csqa_new/train_rand_split.jsonl.statements
    with open(filename, 'r') as f:
        lines = f.read().split("\n")

    # 加载对应的statements文件
    for line in tqdm(lines, desc="loading file"):
        if line == "":
            continue
        # 加载每一个问题及其答案（一行就是一条数据）
        j = json.loads(line)
        # j["statements"]
        # [
        #   {"label": true, "statement": "The sanctions against the school were a punishing blow, and they seemed to ignore the efforts the school had made to change."},
        #   {"label": false, "statement": "The sanctions against the school were a punishing blow, and they seemed to enforce the efforts the school had made to change."},
        #   {"label": false, "statement": "The sanctions against the school were a punishing blow, and they seemed to authoritarian the efforts the school had made to change."},
        #   {"label": false, "statement": "The sanctions against the school were a punishing blow, and they seemed to yell at the efforts the school had made to change."},
        #   {"label": false, "statement": "The sanctions against the school were a punishing blow, and they seemed to avoid the efforts the school had made to change."}
        # ]
        for statement in j["statements"]:
            # sents存放所有的statement
            sents.append(statement["statement"])
        # j["question"]["choices"]
        # [
        #   {"label": "A", "text": "ignore"},
        #   {"label": "B", "text": "enforce"},
        #   {"label": "C", "text": "authoritarian"},
        #   {"label": "D", "text": "yell at"},
        #   {"label": "E", "text": "avoid"}
        # ]
        for answer in j["question"]["choices"]:
            # answers存放所有的text
            answers.append(answer["text"])

    if batch_id >= 0:
        # 分成100组
        output_path = filename + ".%d.mcp" % batch_id
        batch_sents = list(np.array_split(sents, 100)[batch_id])
        batch_answers = list(np.array_split(answers, 100)[batch_id])
    else:
        output_path = filename + ".mcp"
        batch_sents = sents
        batch_answers = answers

    # 一个mcp文件对应100组中的一个组
    # 每个问题5个statement，所以一个mcp文件存9741*5/100 = 488个statement的C_q和C_a
    res = match_mentioned_concepts(nlp, sents=batch_sents, answers=batch_answers, batch_id=batch_id)
    with open(output_path, 'w') as fo:
        json.dump(res, fo)


def test():
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    res = match_mentioned_concepts(nlp, sents=["Sometimes people say that someone stupid has no swimming pool."],
                                   answers=["swimming pool"])
    print(res)


# "sent": "Watch television do children require to grow up healthy.", "ans": "watch television",
if __name__ == "__main__":
    process(sys.argv[1], int(sys.argv[2]))

# test()
