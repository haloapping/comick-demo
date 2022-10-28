import pickle
import streamlit as st
from pathlib import Path
from tagger import POSTagger
from utils import text_preprocessing, word_embedding, idxs_to_tags, print_annotated_text
from oov_sents import sents, tokenize_sents

st.write(
    "<h2 style='text-align: center; margin-bottom: 50px'>Comick Demo Application</h2>",
    unsafe_allow_html=True
)

st.write(
    "<h4 style='text-align: left;'>Choose text</h4>",
    unsafe_allow_html=True
)

selected_text = st.selectbox(
    '',
    options=list(sents.keys()),
    format_func=lambda option: sents[option],
    label_visibility="hidden"
)

oov_tokens = list(pickle.load(open(Path("word_embeddings/oov_embedding_dict.pkl"), "rb")).keys())

# zero
words, idxs_tokenize = text_preprocessing(tokenize_sents[selected_text])
word_embeddings = word_embedding(idxs_tokenize, mode="zero")
zero_pos_tagger = POSTagger(pretrained="zero")
pred_tags = idxs_to_tags(zero_pos_tagger(word_embeddings).argmax(dim=-1))
zero_pos_tags = [(word + " (OOV)", tag) if word.lower() in oov_tokens else (word, tag) for word, tag in zip(tokenize_sents[selected_text], pred_tags)]

# unknown
words, idxs_tokenize = text_preprocessing(tokenize_sents[selected_text])
word_embeddings = word_embedding(idxs_tokenize, mode="unk")
unk_pos_tagger = POSTagger(pretrained="unknown")
pred_tags = idxs_to_tags(unk_pos_tagger(word_embeddings).argmax(dim=-1))
unk_pos_tags = [(word + " (OOV)", tag) if word.lower() in oov_tokens else (word, tag) for word, tag in zip(tokenize_sents[selected_text], pred_tags)]

# comick
words, idxs_tokenize = text_preprocessing(tokenize_sents[selected_text])
word_embeddings = word_embedding(idxs_tokenize, mode="comick")
comick_pos_tagger = POSTagger(pretrained="comick")
pred_tags = idxs_to_tags(comick_pos_tagger(word_embeddings).argmax(dim=-1))
comick_pos_tags = [(word + " (OOV)", tag) if word.lower() in oov_tokens else (word, tag) for word, tag in zip(tokenize_sents[selected_text], pred_tags)]

st.write(
    "<h4 style='text-align: left; margin-top: 20px;'>Part-of-Speech Tag Result</h4>",
    unsafe_allow_html=True
)

st.write(
    "<h6 style='text-align: center'>Initialized OOV Token with Zero Embedding</h6>",
    unsafe_allow_html=True
)

print_annotated_text(zero_pos_tags)

st.write(
    "<h6 style='margin-top: 50px; text-align: center'>Initialized OOV Token with Unknown Polyglot Embedding</h6>",
    unsafe_allow_html=True
)

print_annotated_text(unk_pos_tags)

st.write(
    "<h6 style='margin-top: 50px; text-align: center'>Initialized OOV Token with Comick Embedding</h6>",
    unsafe_allow_html=True
)

print_annotated_text(comick_pos_tags)