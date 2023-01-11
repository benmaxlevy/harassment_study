import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV

from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from imblearn.under_sampling import RandomUnderSampler

from sklearn import metrics, utils

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


class MHClassifier:
    def __init__(self, df=None, samples=None, ngram_range=(1, 1), text_key="text", label_key="label", word2vec=False,
                 model=None):
        if model is None:
            self.clf = None
            self.samples = samples
            self.ngram_range = ngram_range
            self.text_key = text_key
            # cluster sample
            if samples is not False:
                df = df.sample(n=self.samples)

            data = df[self.text_key]

            df[self.text_key] = data

            if not word2vec:

                # fit vectorizer to data

                self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range).fit(data)

                X = self.vectorizer.transform(data)
                y = df[label_key]

            else:
                w2v = KeyedVectors.load_word2vec_format(
                    './GoogleNews-vectors-negative300.bin', binary=True)

                X = []
                y = []
                for i, row in tqdm(df.iterrows()):
                    # get tokenized
                    vectorized = np.zeros(shape=(300,))
                    length = 0
                    for word in word_tokenize(row[self.text_key]):
                        if word in w2v:
                            vectorized += np.array(w2v[word])
                            length = length + 1
                    if length != 0:
                        # average
                        if not np.isnan(np.sum(vectorized)):
                            average = np.array(vectorized) / length
                            X.append(average)
                            y.append(row[label_key])

            # because of limitations with OpenBLAS' multithreading, ensure that
            # OPENBLAS_NUM_THREADS is a small number (e.g., 16), especially if running on a CPU-heavy system
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)

            self.scaler = StandardScaler(with_mean=False)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25,
                                                                                    random_state=16)

            # Don't cheat - fit only on training data
            self.scaler.fit(self.X_train)
            self.X_train = self.scaler.transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        else:
            self.clf = model

    # word2vec
    def vectorize(self, df, text_key):
        w2v = KeyedVectors.load_word2vec_format(
            './GoogleNews-vectors-negative300.bin', binary=True)

        X = []
        for i, row in tqdm(df.iterrows()):
            # get tokenized
            vectorized = np.zeros(shape=(300,))
            length = 0
            for word in word_tokenize(row[text_key]):
                if word in w2v:
                    vectorized += np.array(w2v[word])
                    length = length + 1
            if length != 0:
                # average
                if not np.isnan(np.sum(vectorized)):
                    average = np.array(vectorized) / length
                    X.append(average)
        return X

    def mlp(self):
        """
        MLP = GridSearchCV(MLPClassifier(), {'solver': ['lbfgs'],
                                             'max_iter': [1000000000000000], 'alpha': 10.0 ** -np.arange(1, 10),
                                             'hidden_layer_sizes': np.arange(10, 15)}, n_jobs=-1)
        """
        # found parameters using GSCV
        MLP = MLPClassifier(solver='sgd', alpha=1e-9, hidden_layer_sizes=12,
                            max_iter=10000000000)
        self.clf = MLP.fit(self.X_train, self.y_train)

        y_score = self.clf.predict_proba(self.X_test)
        y_pred = self.clf.predict(self.X_test)

        # Plot metrics
        plot_roc(self.y_test, y_score)
        plt.show()

        plot_precision_recall(self.y_test, y_score)
        plt.show()

        return metrics.classification_report(self.y_test, y_pred)

    def logistic(self):
        # logistic regression
        LR = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=10000000000, n_jobs=-1)
        self.clf = LR.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        return metrics.classification_report(self.y_test, y_pred)

    def multinomialNB(self):
        # MNB
        mnb = MultinomialNB()
        self.clf = mnb.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        return metrics.classification_report(self.y_test, y_pred)

    def rf(self):
        # rf
        rf = RandomForestClassifier(n_jobs=-1)
        self.clf = rf.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        return metrics.classification_report(self.y_test, y_pred)

    def knn(self, n):
        # knn
        knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
        self.clf = knn.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        return metrics.classification_report(self.y_test, y_pred)

    def svm(self):
        # svm
        svm = SVC()
        self.clf = svm.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        return metrics.classification_report(self.y_test, y_pred)
