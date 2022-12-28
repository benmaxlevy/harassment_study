import pandas as pd
import numpy as np
from gensim.models import Doc2Vec, KeyedVectors

import multiprocessing

from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import metrics, utils

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


class MHClassifier:
    def __init__(self, df, samples, ngram_range=(1, 2), text_key="text", label_key="label", word2vec=False):
        self.clf = None
        self.samples = samples
        self.ngram_range = ngram_range
        self.text_key = text_key
        # cluster sample
        if samples is not False:
            df = df.sample(n=self.samples)

        data = df[self.text_key]
        data = data.str.replace("[^\w\s]", "", regex=True)

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    def mlp(self):
        MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1,
                            max_iter=10000000000)
        self.clf = MLP.fit(self.X_train, self.y_train)

        y_pred = self.clf.predict(self.X_test)

        print(self.clf.predict(self.vectorizer.transform([
            "my mind is already racing from not being able to relax."])))

        return metrics.classification_report(self.y_test, y_pred)

    def logistic(self):
        # logistic regression
        LR = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=10000000000, n_jobs=-1)
        self.clf = LR.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        """
        print(self.clf.predict(self.vectorizer.transform([
            "my mind is already racing from not being able to relax."])))
        """
        return metrics.classification_report(self.y_test, y_pred)

    def multinomialNB(self):
        # MNB
        mnb = MultinomialNB()
        self.clf = mnb.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        print(self.clf.predict(self.vectorizer.transform([
            "my mind is already racing from not being able to relax."])))

        return metrics.classification_report(self.y_test, y_pred)

    def rf(self):
        # rf
        rf = RandomForestClassifier(n_jobs=-1)
        self.clf = rf.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        print(self.clf.predict(self.vectorizer.transform([
            "my mind is already racing from not being able to relax."
        ])))

        return metrics.classification_report(self.y_test, y_pred)

    def knn(self, n):
        # knn
        knn = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
        self.clf = knn.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        print(self.clf.predict(self.vectorizer.transform([
            "my mind is already racing from not being able to relax."
        ])))

        return metrics.classification_report(self.y_test, y_pred)

    def svm(self):
        # svm
        svm = SVC()
        self.clf = svm.fit(self.X_train, self.y_train)

        # get report
        y_pred = self.clf.predict(self.X_test)

        print(self.clf.predict(self.vectorizer.transform([
            "my mind is already racing from not being able to relax."
        ])))

        return metrics.classification_report(self.y_test, y_pred)