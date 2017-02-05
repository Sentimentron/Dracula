import csv
import numpy
import scipy
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def train_test_split(x, y, test_size=0.05):
    l = int(len(x) * test_size)
    print(l)
    return x[:l], x[l:], y[:l], y[l:]

with open("Data/quora_duplicate_questions_stripped.tsv") as fin:
    filereader = csv.reader(fin, delimiter='\t')
    x1, x2, y = [], [], []
    for _, _, _, q1, q2, dup in filereader:
        x1.append(q1)
        x2.append(q2)
        y.append(int(dup))

    x_valid, y_valid = [], []
    with open("Data/quora_duplicate_questions_eval.tsv") as fin:
        filereader = csv.reader(fin, delimiter='\t')
        for _, _, _, q1, q2, dup in filereader:
            x_valid.append((q1, q2))
            y_valid.append(int(dup))

    x_train, x_test, y_train, y_test = train_test_split(
        list(zip(x1, x2)), y, test_size=0.05# random_state=123
    )

    x_eval, x_test, y_eval, y_test = train_test_split(
        x_test, y_test, test_size=0.50# random_state=123
    )

    x1_train = [i[0] for i in x_train]
    x2_train = [i[1] for i in x_train]
    x_train = x1_train + x2_train

    x1_eval = [i[0] for i in x_eval]
    x2_eval = [i[1] for i in x_eval]

    x1_test = [i[0] for i in x_test]
    x2_test = [i[1] for i in x_test]

    x1_valid = [i[0] for i in x_valid]
    x2_valid = [i[1] for i in x_valid]

    count_vect = CountVectorizer().fit(x_train)
    x1_train = count_vect.transform(x1_train)
    x2_train = count_vect.transform(x2_train)
    x_train = count_vect.transform(x_train)
    x1_eval = count_vect.transform(x1_eval)
    x2_eval = count_vect.transform(x2_eval)
    x1_test = count_vect.transform(x1_test)
    x2_test = count_vect.transform(x2_test)
    x1_valid = count_vect.transform(x1_valid)
    x2_valid = count_vect.transform(x2_valid)

    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train)
    x1_train = tf_transformer.transform(x1_train)
    x2_train = tf_transformer.transform(x2_train)
    x1_eval = tf_transformer.transform(x1_eval)
    x2_eval = tf_transformer.transform(x2_eval)
    x1_test = tf_transformer.transform(x1_test)
    x2_test = tf_transformer.transform(x2_test)
    x1_valid = tf_transformer.transform(x1_valid)
    x2_valid = tf_transformer.transform(x2_valid)

    x_train = scipy.sparse.hstack((x1_train, x2_train))
    x_eval = scipy.sparse.hstack((x1_eval, x2_eval))
    x_test = scipy.sparse.hstack((x1_test, x2_test))
    x_valid = scipy.sparse.hstack((x1_valid, x2_valid))

    print(x_train.shape)
    print(x_eval.shape)
    print(x_test.shape)
    print(x_valid.shape)

    clf = MultinomialNB()
    for clf in [MultinomialNB(), SGDClassifier()]:
        clf = clf.fit(x_train, y_train)

        predicted_train = clf.predict(x_train)
        print (numpy.mean(predicted_train == y_train))
        predicted_eval = clf.predict(x_eval)
        print (numpy.mean(predicted_eval == y_eval))
        predicted_test = clf.predict(x_test)
        print (numpy.mean(predicted_test == y_test))
        predicted_valid = clf.predict(x_valid)
        print (numpy.mean(predicted_valid == y_valid))

