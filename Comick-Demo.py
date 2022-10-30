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
words, idxs_tokenize = text_preprocessing(tokenize_sents[selected_text])
word_embeddings = word_embedding(idxs_tokenize, mode="comick")
comick_pos_tagger = POSTagger(pretrained="comick")
pred_tags = idxs_to_tags(comick_pos_tagger(word_embeddings).argmax(dim=-1))
comick_pos_tags = [(word + " (OOV)", tag) if word.lower() in oov_tokens else (word, tag) for word, tag in zip(tokenize_sents[selected_text], pred_tags)]

st.write(
    "<h4 style='text-align: left; margin-top: 20px;'>Part-of-Speech Tag</h4>",
    unsafe_allow_html=True
)

print_annotated_text(comick_pos_tags)

expander = st.expander("See tag information")
expander.write(
    """
    <ol>
        <li>CC  : Coordinating conjunction, also called coordinator. Example: dan, tetapi, atau.</li>
        <li>CD  : Cardinal number. Example: dua, juta, enam, 7916, sepertiga, 0,025, 0,525, banyak, kedua, ribuan, 2007, 25.</li>
        <li>DT  : Determiner / article. Example: para, sang, si.</li>
        <li>FW  : Foreign word.. Example: climate change, terms and conditions.</li>
        <li>IN  : Preposition. Example: dalam, dengan, di, ke, oleh, pada, untuk.</li>
        <li>JJ  : Adjective. Example: bersih, panjang, hitam, lama, jauh, marah, suram, nasional, bulat.</li>
        <li>MD  : Modal and auxiliary verb. Example: boleh, harus, sudah, mesti, perlu.</li>
        <li>NEG : Negation. Example: tidak, belum, jangan.</li>
        <li>NN  : Noun. Example: monyet, bawah, sekarang, rupiah.</li>
        <li>NND : Classifier, partitive, and measurement noun. Example: orang, ton, helai, lembar.</li>
        <li>NNP : Proper noun. Example: Boediono, Laut Jawa, Indonesia, India, Malaysia, Bank Mandiri, BBKP, Januari, Senin, Idul Fitri, Piala Dunia, Liga Primer, Lord of the Rings: The Return of the King.</li>
        <li>OD  : Ordinal number. Example: ketiga, ke-4, pertama.</li>
        <li>PR  : Demonstrative pronoun. Example: ini, itu, sini, situ.</li>
        <li>PRP : Personal pronoun. Example: saya, kami, kita, kamu, kalian, dia, mereka.</li>
        <li>RB  : Adverb. Example: sangat, hanya, justru, niscaya, segera.</li>
        <li>RP  : Particle. Example: pun, -lah, -kah.</li>
        <li>SC  : Subordinating conjunction, also called subordinator. Example: sejak, jika, seandainya, supaya, meski, seolah-olah, sebab, maka, tanpa, dengan, bahwa, yang, lebih ... daripada ..., semoga.</li>
        <li>SYM : Symbol. Example: IDR, +, %, @.</li>
        <li>UH  : Interjection. Example: brengsek, oh, ooh, aduh, ayo, mari, hai.</li>
        <li>VB  : Verbs. Example: merancang, mengatur, pergi, bekerja, tertidur.</li>
        <li>WH  : Question. Example: siapa, apa, mana, kenapa, kapan, di mana, bagaimana, berapa.</li>
        <li>X   : Unknown. Example: statemen.</li>
        <li>Z   : Punctuation. Example: "...", ?, .</li>
        <li>UNK : Unknown token, because word embedding does not exist.</li>
    </ol>
    """,
    unsafe_allow_html=True
)