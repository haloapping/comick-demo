import torch
import pickle
from pathlib import Path
from annotated_text import annotated_text

token_to_idx = pickle.load(open(Path("token/token_to_idx.pkl"), "rb"))
oov_embeddings = pickle.load(open(Path("word_embeddings/oov_embedding_dict.pkl"), "rb"))
zero_word_embeddings = pickle.load(open(Path("word_embeddings/zero_oov_word_embeddings.pkl"), "rb"))
unk_word_embeddings = pickle.load(open(Path("word_embeddings/unk_oov_word_embeddings.pkl"), "rb"))
comick_word_embeddings = pickle.load(open(Path("word_embeddings/comick_word_embeddings.pkl"), "rb"))

def text_preprocessing(tokens: str):
    tokens = [token.lower() if token.lower() in list(oov_embeddings.keys()) else token for token in tokens]
    tokens_to_idxs = [token_to_idx[token] for token in tokens]
    tokens_to_idxs = torch.LongTensor(tokens_to_idxs)

    return tokens, tokens_to_idxs

def word_embedding(idxs, mode="comick"):
    if mode == "zero":
        return zero_word_embeddings(idxs)
    elif mode == "unk":
        return unk_word_embeddings(idxs)
    else:
        return comick_word_embeddings(idxs)

def idxs_to_tags(idxs):
    tags = ["CC", "CD", "DT", "FW", "IN", "JJ", "MD", "NEG", "NN", "NND", "NNP", "OD", "PR", "PRP", "RB", "RP", "SC", "SYM", "UH", "VB", "WH", "X", "Z", "UNK"]
    tags_dict = dict(zip(list(range(25)), tags))

    return [tags_dict[idx.item()] for idx in idxs]

def print_annotated_text(pos_tags, num_token_per_line=8):
    start = 0 
    stop = num_token_per_line
    for _ in range(len(pos_tags) // 4 + 1):
        annotated_text(*pos_tags[start:stop])
        start += 8
        stop += 8