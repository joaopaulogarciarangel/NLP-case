# NLP with Disaster Tweets — Binary Classification

> Predicting whether tweets refer to real disasters using classical NLP and transformer models — best F1 of 0.8169 with DistilBERT.

*Leia em [Português](README.pt-BR.md)*

## Summary

Kaggle challenge to classify tweets as real disaster reports (1) or figurative/unrelated language (0). Built a full NLP pipeline from TF-IDF + classical ML to a fine-tuned DistilBERT transformer. DistilBERT reached 85.2% accuracy and F1 of 0.82, outperforming tuned Logistic Regression (F1 0.76) and Naive Bayes (F1 0.75). Demonstrates end-to-end NLP skills from text preprocessing and feature engineering through deep learning.

**Stack:** Python · pandas · scikit-learn · NLTK · Hugging Face Transformers · DistilBERT · PyTorch · matplotlib · seaborn · wordcloud  
**Result:** DistilBERT F1 = 0.8169 · Accuracy = 85.2% (epoch 2)

## Problem

Distinguishing tweets about real disasters from figurative or unrelated use of disaster vocabulary is a classic NLP challenge with direct application in emergency monitoring systems. A tweet saying "the concert was on fire" and one reporting an actual wildfire look superficially similar — models must capture context, not just keywords.

## Approach

- Text cleaning (URLs, mentions, hashtags, special chars) and lemmatization with NLTK WordNetLemmatizer
- TF-IDF vectorization with unigrams and bigrams (30,000 features)
- Feature engineering: `text_length`, `Composta` (compound keyword flag), geographic features (country, continent) extracted via GeoText and geonamescache
- Hyperparameter tuning for Naive Bayes and Logistic Regression via RandomizedSearchCV (5-fold CV)
- Fine-tuning pre-trained DistilBERT for sequence classification (3 epochs, max 128 tokens)

## Results & findings

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| DistilBERT (epoch 2) | 0.8523 | 0.8169 | 0.8655 | 0.7735 |
| Logistic Regression (tuned) | 0.8070 | 0.7602 | 0.8076 | 0.7180 |
| Naive Bayes (tuned) | 0.8050 | 0.7531 | 0.8177 | 0.6980 |

- DistilBERT substantially outperforms classical ML on all metrics
- Logistic Regression with TF-IDF + engineered features edges out Naive Bayes
- Naive Bayes shows high precision but lower recall — tends to miss real disasters
- Geographic features carry meaningful signal: tweets from Africa have a 66.25% disaster rate vs. 44.56% from Europe
- DistilBERT performance drops slightly at epoch 3 vs. epoch 2, making epoch 2 the optimal checkpoint

## Dataset

**Kaggle — Natural Language Processing with Disaster Tweets**  
[kaggle.com/c/nlp-getting-started](https://www.kaggle.com/c/nlp-getting-started)  
7,613 labeled tweets (42.97% disaster, 57.03% non-disaster) with keyword and location metadata.

## Notebook

Full analysis with preprocessing, EDA, feature engineering, model training, and comparison in `notebook.ipynb`.
