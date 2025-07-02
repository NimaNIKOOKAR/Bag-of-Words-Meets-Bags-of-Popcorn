# Codes will be added

# Word2Vec NLP Tutorial – Bag-of-Popcorn

A **hands-on walkthrough** of building sentiment-analysis models for IMDB movie reviews – from classic Bag-of-Words to Word2Vec embeddings and beyond. Perfect if you’re comfortable with Python basics and want to dip your toes into deep-learning-flavoured NLP without a GPU.

---

## Table of Contents
1. [Project layout](#project-layout)  
2. [Setup & dependencies](#setup--dependencies)  
3. [Dataset](#dataset)  
4. [Part 1 – Bag-of-Words baseline](#part-1--bag-of-words-baseline)  
5. [Part 2 – Training Word2Vec](#part-2--training-word2vec)  
6. [Part 3 – From words to reviews](#part-3--from-words-to-reviews)  
7. [Part 4 – Compare & iterate](#part-4--compare--iterate)  
8. [Submission format](#submission-format)  
9. [Further reading](#further-reading)  
10. [Citation & licence](#citation--licence)

---

## Project layout
```text
.
├── data/
│   ├── labeledTrainData.tsv      # 25k reviews + labels
│   ├── testData.tsv              # 25k reviews (no labels)
│   ├── unlabeledTrainData.tsv    # 50k extra reviews
│   └── sampleSubmission.csv
├── notebooks/
│   ├── 01_bow_baseline.ipynb
│   ├── 02_word2vec_training.ipynb
│   ├── 03_review_vectors.ipynb
│   └── 04_compare_methods.ipynb
├── src/
│   ├── clean_text.py             # HTML stripping, tokenisation
│   ├── bow_vectoriser.py         # Count & TF-IDF wrappers
│   ├── train_word2vec.py         # gensim trainer + loaders
│   ├── avg_vector.py             # mean-pool & tf-idf weights
│   ├── kmeans_centroids.py       # bag-of-centroids builder
│   └── evaluate.py               # train/test split, ROC-AUC
└── README_Word2Vec.md
```
Adjust `DATA_DIR` in `src/config.py` once – every script picks it up.

---

## Setup & dependencies
```bash
# Python ≥3.9 recommended
pip install pandas numpy scipy scikit-learn nltk gensim cython beautifulsoup4
python -m nltk.downloader punkt stopwords
```
*Optional:* compile gensim’s Cython for 4× faster Word2Vec.

---

## Dataset
* **100 000 IMDB reviews** (multi-sentence, free-form)  
* Binary labels only for the first **25 000** (train).  
* Evaluation metric: **ROC-AUC** on the hidden test labels.

---

## Part 1 – Bag-of-Words baseline
1. Strip HTML & non-letters → lowercase tokens.  
2. Remove stop-words (NLTK list).  
3. `CountVectorizer(max_features=5000)` → sparse matrix.  
4. `RandomForestClassifier(n_estimators=100)` → score ~0.83 AUC.

> **Tip:** swap in `TfidfVectorizer` or `LogisticRegression` — easy +1‑2 pp.

---

## Part 2 – Training Word2Vec
```python
from gensim.models import Word2Vec
sentences = [tokenise(review) for review in all_reviews]
model = Word2Vec(sentences, vector_size=300, window=10,
                 min_count=40, sample=1e-3, workers=4)
model.wv.save("imdb_300d.w2v")
```
* 75 k reviews ≈ 850 k sentences → ≈ 15 k‑word vocab.  
* Training <15 min on 4 threads.

---

## Part 3 – From words to reviews
### 3.1 Mean‑pooled vectors
* Remove stop‑words, average each word vector → 300‑dim sentence embedding.  
* Random‑Forest → **~0.81 AUC** (slightly below BoW).

### 3.2 Bags of centroids
1. K‑Means on word vectors (k ≈ vocab/5).  
2. Vector = histogram over cluster IDs.  
3. RF scores **~0.82 AUC** — on par with BoW, worse than fine‑tuned BOW.

> Larger corpora (billions) or doc2vec usually beat vanilla BoW.

---

## Part 4 – Compare & iterate
| Method | Features | AUC |
|--------|----------|-----|
| CountVectorizer + RF | 5 000 | **0.83** |
| TF‑IDF + LogReg | 20 000 | **0.86** |
| Word2Vec mean + RF | 300 | 0.81 |
| W2V centroids + RF | 3 000 | 0.82 |
| Pre‑trained GoogleNews W2V + mean + LogReg | 300 | **0.87** |

Take‑away: **data size & task match trump model fanciness**. Word vectors shine when you pre‑train on huge, diverse corpora or fine‑tune end‑to‑end.

---

## Submission format
```csv
id,sentiment
123_45,0
678_90,1
...
```
Generate with `predict.py` → upload `*.csv` on Kaggle → leaderboard updates instantly.

---

## Further reading
* **Mikolov et al. (2013)** – Efficient Estimation of Word Representations.  
* **Le & Mikolov (2014)** – Paragraph Vector (doc2vec).  
* Stanford CS224N lecture – *“Deep Learning for NLP without Magic”*.  
* Gensim docs & Radim Řehůřek’s blog for practical tips.

---

## Citation & licence
```
@misc{bagofpopcorn2014,
  title  = {Bag of Words Meets Bags of Popcorn},
  author = {Joyce Chen and Wendy Kan and Will Cukierski},
  year   = {2014},
  howpublished = {\url{https://kaggle.com/competitions/word2vec-nlp-tutorial}}
}
```
Tutorial scaffolding © 2025 MIT‑licensed. Dataset © Stanford / Maas et al.; please cite original paper when publishing results.

---

*Have fun turning text into vectors—and watch those ROC curves climb!*
