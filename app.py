"""Demonstração Streamlit: streamlit run app.py"""
import streamlit as st
from PIL import Image

from src.pipeline import EcoAssist
from src.pln.busca_semantica import Buscador
from src.visao_computacional.classificador import ClassificadorVisual

st.set_page_config(page_title="EcoAssist", page_icon="♻️")
st.title("♻️ EcoAssist — descarte correto de resíduos")


@st.cache_resource
def carregar():
    return EcoAssist(ClassificadorVisual(), Buscador())


try:
    app = carregar()
except FileNotFoundError:
    st.error("Pesos não encontrados em resultados/. Rode o treino (veja o README).")
    st.stop()

arq = st.file_uploader("Foto do resíduo", type=["jpg", "jpeg", "png"])
pergunta = st.text_input("Sua pergunta", "Como descarto isso?")

if arq and pergunta:
    try:
        img = Image.open(arq)
        img.load()
    except Exception:
        st.error("Não foi possível ler essa imagem. Envie um JPG ou PNG válido.")
        st.stop()
    st.image(img, width=320)
    r = app.responder(img, pergunta)
    d = r["decisao"]
    st.subheader("Resposta")
    st.write(d.resposta)
    for a in d.avisos:
        st.warning(a)
    with st.expander("Como o sistema decidiu"):
        st.write("Visão (top-3):", [(c, f"{p:.0%}") for c, p in r["top_visual"]])
        st.write("Intenção:", r["intencao"], "| Material no texto:", r["material_texto"])
        st.write("Palavras-chave:", r["palavras_chave"])
        st.write("Regra aplicada:", d.regra)
