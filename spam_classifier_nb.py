#split_data.py

from __future__ import division
from codecs import open
from operator import __pos__
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            truncated_line = line.replace("{} {} {}".format(words[0], words[1], words[2]), '')
            docs.append(truncated_line)
            labels.append(words[1])
    return docs, labels

if __name__ == "__main__":
    all_docs, all_labels = read_documents("sample-text.txt")
    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    
    train_labels = np.array(all_labels[:split_point])
    train_labels[train_labels == 'neg'] = 0
    train_labels[train_labels == 'pos'] = 1
    train_labels = list(train_labels)
    
    eval_docs = all_docs[split_point:]

    eval_labels = np.array(all_labels[split_point:])
    eval_labels[eval_labels == 'neg'] = 0
    eval_labels[eval_labels == 'pos'] = 1
    eval_labels = list(eval_labels)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_docs)

    classif = MultinomialNB()
    classif.fit(X, train_labels)

    X_eval = vectorizer.transform(eval_docs)

    predictions = classif.predict(X_eval)

    prec = precision_score(eval_labels, predictions, pos_label='1')
    rec = recall_score(eval_labels, predictions, pos_label ='1')
    f1 = f1_score(eval_labels, predictions, pos_label='1')
    acc = accuracy_score(eval_labels, predictions)

    print("Precision: {}, Recall: {}, F1 score: {}, Accuracy: {}".format(prec, rec, f1, acc))

