# NLP com Tweets de Desastre — Classificação Binária

> Prever se tweets se referem a desastres reais usando NLP clássico e transformers — melhor F1 de 0,8169 com DistilBERT.

*Read in [English](README.md)*

## Resumo

Desafio do Kaggle para classificar tweets como relatos de desastres reais (1) ou uso figurativo/não relacionado da linguagem (0). Construí um pipeline completo de NLP, do TF-IDF com ML clássico ao fine-tuning do DistilBERT. O DistilBERT atingiu 85,2% de acurácia e F1 de 0,82, superando Regressão Logística otimizada (F1 0,76) e Naive Bayes (F1 0,75). Demonstra habilidades end-to-end de NLP, do pré-processamento e feature engineering ao deep learning.

**Stack:** Python · pandas · scikit-learn · NLTK · Hugging Face Transformers · DistilBERT · PyTorch · matplotlib · seaborn · wordcloud  
**Resultado:** DistilBERT F1 = 0,8169 · Acurácia = 85,2% (época 2)

## Problema

Distinguir tweets sobre desastres reais do uso figurativo de vocabulário catastrófico é um desafio clássico de NLP, com aplicação direta em sistemas de monitoramento de emergência. Um tweet dizendo "o show foi um incêndio" e outro reportando um incêndio florestal real se parecem na superfície — os modelos precisam capturar contexto, não apenas palavras-chave.

## Abordagem

- Limpeza de texto (URLs, menções, hashtags, caracteres especiais) e lematização com NLTK WordNetLemmatizer
- Vetorização TF-IDF com unigramas e bigramas (30.000 features)
- Feature engineering: `text_length`, `Composta` (flag de keyword composta), features geográficas (país, continente) extraídas via GeoText e geonamescache
- Otimização de hiperparâmetros de Naive Bayes e Regressão Logística via RandomizedSearchCV (CV com 5 folds)
- Fine-tuning do DistilBERT pré-treinado para classificação de sequências (3 épocas, máx. 128 tokens)

## Resultados e achados

| Modelo | Acurácia | F1 | Precisão | Recall |
|---|---|---|---|---|
| DistilBERT (época 2) | 0,8523 | 0,8169 | 0,8655 | 0,7735 |
| Regressão Logística (otimizada) | 0,8070 | 0,7602 | 0,8076 | 0,7180 |
| Naive Bayes (otimizado) | 0,8050 | 0,7531 | 0,8177 | 0,6980 |

- DistilBERT supera substancialmente o ML clássico em todas as métricas
- Regressão Logística com TF-IDF + features engenheiradas supera o Naive Bayes
- Naive Bayes tem precisão alta mas recall baixo — tende a perder desastres reais
- Features geográficas carregam sinal relevante: tweets africanos têm taxa de desastre de 66,25% vs. 44,56% dos europeus
- DistilBERT apresenta leve queda de desempenho na época 3 em relação à época 2, tornando o checkpoint da época 2 o ótimo

## Dataset

**Kaggle — Natural Language Processing with Disaster Tweets**  
[kaggle.com/c/nlp-getting-started](https://www.kaggle.com/c/nlp-getting-started)  
7.613 tweets rotulados (42,97% desastre, 57,03% não-desastre) com metadados de keyword e localização.

## Notebook

Análise completa com pré-processamento, EDA, feature engineering, treinamento e comparação de modelos em `notebook.ipynb`.
