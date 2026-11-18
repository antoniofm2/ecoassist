# Avaliação final (pares novos do grupo)

Preencher **antes** de usar o app com essas fotos.

- Fotos de celular em `fotos/` (JPG/PNG). Não use fotos do TrashNet nem pessoas, rostos ou locais identificáveis.
- `pares.csv`, uma linha por par:
  - `foto`: nome do arquivo dentro de `fotos/`
  - `pergunta`: pergunta escrita pelo grupo (não copiar de `dados/perguntas_livres.csv`)
  - `classe_esperada`: cardboard, glass, metal, paper, plastic ou trash
  - `intencao_esperada`: como_descartar, identificar_material, reciclavel, cor_lixeira, cuidados ou curiosidade
  - `resposta_esperada`: texto curto do que a resposta correta deve conter (ex.: "lixeira amarela")
- Não ajuste o PLN nem o modelo depois de ver esses resultados.

## Como rodar

1. Preencha `pares.csv` e coloque as fotos em `fotos/`.
2. Gere os resultados (usa o mesmo fluxo do app Streamlit):
   ```bash
   python -m src.avaliacao_final
   ```
   Isso cria `resultados_final.csv` com a classe prevista, a intenção prevista, a resposta do app, a fonte e a coluna `resposta_correta` em branco. Acertos de **classe** e **intenção** são calculados automaticamente.
3. Abra `resultados_final.csv` e marque `resposta_correta` com **1** (correta) ou **0** (incorreta). Essa marcação é do grupo, porque frases diferentes podem responder corretamente à mesma pergunta. Perguntas com duas partes só contam como corretas se **ambas** forem respondidas.
4. Recalcule as métricas depois da marcação:
   ```bash
   python -m src.avaliacao_final --resumo
   ```

O script recusa sobrescrever um `resultados_final.csv` que já tenha marcações (use `--forcar` só se quiser perdê-las, ou `--saida outro.csv`).
