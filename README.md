# EcoAssist — assistente de descarte de resíduos (Projeto B)

Disciplina de Inteligência Artificial — AFYA | UNIMA Maceió (Prof. Icaro Ferreira).

**Integrantes e responsabilidades:** _(preencher: nomes completos e papéis)_
**Artigo (Overleaf):** _(colar link)_

## Problema
O usuário envia a foto de um resíduo e faz uma pergunta em português ("Posso reciclar isso?"). O sistema reconhece o material, interpreta a pergunta e responde com a orientação de descarte.

## Arquitetura
```
foto ──► Visão Computacional (MobileNetV2, top-3) ─┐
                                                   ├─► Agente de decisão (regras + confiança) ─► resposta
pergunta ─► PLN (intenção + material citado + busca semântica) ─┘
```
| Módulo | Pasta | Técnica |
|---|---|---|
| Visão Computacional + ML | `src/visao_computacional`, `src/machine_learning` | Transfer learning (MobileNetV2) no TrashNet; baseline HOG+SVM |
| PLN | `src/pln` | TF-IDF + Regressão Logística (intenção), extração de palavras-chave/material, busca semântica (MiniLM multilíngue, com fallback TF-IDF) |
| Decisão | `src/decisao` | Agente baseado em utilidade com regras R1–R4 (baixa confiança, texto resolve incerteza, conflito texto×imagem) |
| Integração | `src/pipeline.py`, `app.py` | Streamlit |

## Instalação e execução
```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm   # opcional
python -m src.pln.gerar_intencoes           # gera dados/intencoes.csv
python -m src.pln.intencao                  # treina e avalia a intenção
```
Dataset de imagens: baixe o TrashNet (6 classes) e coloque em `dados/trashnet/<classe>/*.jpg`.
```bash
python -m src.machine_learning.treinar_imagens --dados dados/trashnet --epocas 8
streamlit run app.py
pytest testes
```
O treino roda em CPU, porém é bem mais rápido no Google Colab.

## Resultados
Preencher **somente com valores gerados em `resultados/`** (`metricas_imagens.json`, gráficos, exemplos).
**Imagens (medido, TrashNet, 2527 imagens, teste = 380, CPU, 6 épocas):** MobileNetV2 acurácia 0,882 / F1 macro 0,865; baseline HOG+SVM acurácia 0,611 / F1 macro 0,582. A classe `trash` é a mais difícil (poucas imagens). Detalhes em `resultados/metricas_imagens.json` e `resultados/graficos/`.

**Intenção (medido):** F1 = 1,0 no teste sintético, mas acurácia 0,65 e F1 macro 0,66 em 48 perguntas livres (`dados/perguntas_livres.csv`, `python -m src.pln.avaliar_intencao_livre`). Esse conjunto é um rascunho gerado por IA: a equipe deve revisar e ampliar antes de citar no artigo.

Atenção: o dataset de intenções é sintético (modelos de frases) e a divisão treino/teste compartilha os mesmos modelos, então o F1 ≈ 1,0 do classificador de intenção é otimista. Para o artigo, avalie também em perguntas escritas livremente pela equipe (conjunto à parte).

## Limitações
- TrashNet tem fundo limpo; fotos reais tendem a piorar o desempenho.
- Base de conhecimento pequena e baseada em regras gerais brasileiras; regras locais de coleta variam.
- Intenções sintéticas (ver acima).

## Referências
Adicionar: TrashNet (Yang & Thung, 2016), MobileNetV2 (Sandler et al., 2018), Sentence-BERT (Reimers & Gurevych, 2019), Resolução CONAMA 275/2001.
