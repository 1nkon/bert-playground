#!/usr/bin/env python
# coding: utf-8
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import time
from pathlib import Path
import fasttext.util
from .utils import ROOT_DIR


def lget(lst, pos):
    return list(map(lambda x: x[pos], lst))


def calc_w(x, y, w):
    return x * w[0] + y * w[1]


def mk_graph(x1):
    x1 = list(filter(lambda x: -2 < x < 0.99, x1))[:40]
    kwargs = dict(alpha=0.3, bins=20)

    plt.hist(x1, **kwargs, color='g', label='FastText score')
    plt.gca().set(title='Top 40 masks histogram of embeddings score', ylabel='Count')

    plt.legend()
    plt.show()


def mk_graph2(x1):
    kwargs = dict(alpha=1, bins=50)

    plt.hist(x1, **kwargs, color='r', label='Weighted score')
    plt.gca().set(
        title='Distribution of weighted score of top 200 unfiltered results (Target excluded)',
        ylabel='Count'
    )

    plt.legend()
    plt.show()


# TODO: make Model configurable
class Pipeline:
    def __init__(self, interactive=False):
        start_time = time.time()
        # ft_size = 100 # ~2.6 GB
        ft_size = 200  # ~4.5 GB
        # ft_size = 300  # ~8 GB

        self.ft_size = ft_size

        def get_ft_path(n):
            return ROOT_DIR + "/data/cc.en." + str(n) + ".bin"

        cur_path = get_ft_path(ft_size)

        print("Initializing fast text")

        if Path(cur_path).exists():
            print("Found existing model, loading.")
            ft = fasttext.load_model(cur_path)
        else:
            print("Configured model is not found. Loading default model.")
            ft = fasttext.load_model(get_ft_path(300))

            print("Compressing model")
            fasttext.util.reduce_model(ft, ft_size)

            ft.save_model(cur_path)

        self.ft = ft

        print("Loading bert")
        # ~3 GB
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.model = AutoModelWithLMHead.from_pretrained("roberta-large")

        self.interactive = interactive

        print("Server started in %s seconds" % ('{0:.4f}'.format(time.time() - start_time)))

    def find_top(self, sentence, k, top_bert, bert_norm, min_ftext, weights):
        tokenizer = self.tokenizer
        model = self.model
        ft = self.ft

        print("Input: ", sentence)
        start_time = time.time()
        sentence_match = re.search("(\w+)#(\w+)?", sentence)
        target = None

        if sentence_match:
            target = re.sub("#", "", sentence_match.group(1))
            target = target.strip()

        sequence = re.sub("(\w+)?#(\w+)?", tokenizer.mask_token, sentence)

        input = tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

        token_logits = model(input)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]

        # Filter top <top_bert> results of bert output
        topk = torch.topk(mask_token_logits, top_bert, dim=1)
        top_tokens = list(zip(topk.indices[0].tolist(), topk.values[0].tolist()))

        unfiltered = list()
        filtered = list()

        norm_d = top_tokens[bert_norm - 1][1]
        norm_k = top_tokens[0][1] - norm_d

        # Filter bert output by <min_ftext>
        for token, value in top_tokens:
            word = tokenizer.decode([token]).strip()
            norm_value = (value - norm_d) / norm_k

            sim = cosine_similarity(ft[target].reshape(1, -1), ft[word].reshape(1, -1))[0][0]

            if not self.interactive and word == target:
                continue

            if sim >= min_ftext:
                filtered.append((word, value, norm_value, sim, calc_w(norm_value, sim, weights)))

            unfiltered.append((word, value, norm_value, sim, calc_w(norm_value, sim, weights)))

        done = (time.time() - start_time)

        kfiltered = filtered[:k]
        kunfiltered = unfiltered[:k]

        kfiltered = sorted(kfiltered, key=lambda x: -x[len(x) - 1])
        kunfiltered = sorted(kunfiltered, key=lambda x: -x[len(x) - 1])

        filtered_top = pd.DataFrame({
            'word': lget(kfiltered, 0),
            'bert': self.dget(kfiltered, 1),
            'normalized': self.dget(kfiltered, 2),
            'ftext': self.dget(kfiltered, 3),
            'score': self.dget(kfiltered, 4)
        })

        if self.interactive:
            if min_ftext > 0:
                print("Unfiltered top:")

                print(pd.DataFrame({
                    'word': lget(kunfiltered, 0),
                    'bert': self.dget(kunfiltered, 1),
                    'normalized': self.dget(kunfiltered, 2),
                    'ftext': self.dget(kunfiltered, 3),
                    'score': self.dget(kunfiltered, 4)
                }))

            print("Filtered top:")

            print(filtered_top)

            mk_graph(lget(unfiltered, 2)[:100])
            mk_graph2(lget(list(filter(lambda x: x[0] != target, unfiltered)), 4))

            if target is not None:
                vec = tokenizer.encode(target, return_tensors="pt")[0]
                if len(vec) == 3:
                    tk = vec[1].item()
                    pos = None
                    score = None

                    for e, (t, v) in enumerate(top_tokens):
                        if t == tk:
                            score = v
                            break
                    print("Original word position: ", pos, "; score: ", score)
                else:
                    if len(vec) > 3:
                        print("Original word is more then 1 token")
                        print(tokenizer.tokenize(target))
                    else:
                        print("Original word wasn't found")

        print("Finished in %s seconds" % '{0:.4f}'.format(done))
        print("===================")

        return filtered_top

    def do_find(self, s):
        return self.find_top(s, 10, 200, 200, 0.25, [1, 1])

    def dget(self, lst, pos):
        return list(map( lambda x: '{0:.2f}'.format(x[pos]), lst)) if self.interactive else lget(lst, pos)
