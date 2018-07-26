import pandas as pd
import numpy as np
import sys
import pickle
import time
from io import StringIO
import matplotlib.pyplot as plt
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from  sklearn.metrics  import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

def func(file):
    start = time.time()
    ##import dataset
    df = pd.read_csv(file,sep = '\t', encoding = 'utf-8')
    df = df[pd.notnull(df['icerik'])]
    col = ['sinif', 'icerik']

    df = df[col]
    df.columns = ['sinif', 'icerik']
    df['sinif_id'] = df['sinif'].factorize()[0]

    sinif_id_df = df[['sinif', 'sinif_id']].drop_duplicates().sort_values('sinif_id')
    sinif_to_id = dict(sinif_id_df.values)
    id_to_sinif = dict(sinif_id_df[['sinif_id', 'sinif']].values)
    ##shows dataset's class graph
    fig = plt.figure(figsize=(8,6))
    df.groupby('sinif').icerik.count().plot.bar(ylim=0)
    plt.show()
    ##icerigi tfidf vektorlerine cevirir.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.icerik).toarray()
    labels = df.sinif_id

    N = 10

    for sinif, sinif_id in sorted(sinif_to_id.items()):
      features_chi2 = chi2(features, labels == sinif_id)
      indices = np.argsort(features_chi2[0])
      ##kelime listesi olusturur.
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

      print("# '{}':".format(sinif))
      print("  . Most correlated words:\n       . {}".format('\n       . '.join(unigrams[-N:])))

    X_train, X_test, y_train, y_test = train_test_split(df['icerik'], df['sinif'], random_state = 0)

    count_vect = CountVectorizer(vocabulary = unigrams)
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    pikl = './MultiNB.pikl'
    pickle.dump(clf,open(pikl,'wb'))
    pikl_v = './unigram_vocab.pikl'
    pickle.dump(unigrams,open(pikl_v,'wb'))



    def f_predict():
            print(clf.predict(count_vect.transform([input("Enter text:  ")])))

    c_weight = {0 : 0.77, 1 : 0.58, 2 : 0.72, 3 : 0.72}    

    models = [
        RandomForestClassifier(n_estimators=10,criterion = 'gini',warm_start = 'True', min_samples_split = 5, max_depth=20, random_state=0),
        LinearSVC(class_weight = c_weight),
        MultinomialNB(alpha = 10),
        LogisticRegression(penalty = 'l2',C = 10.0, class_weight=c_weight, random_state=0),
        SVC(C = 10.0,kernel='linear',class_weight = c_weight, max_iter = -1 , probability=True),

    ]

    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    linear_class = LinearSVC().fit(X_train_tfidf, y_train)

    SVC_class = SVC(kernel='linear',probability=True).fit(X_train_tfidf, y_train)


    pikl_linear_SVC = './Linear_SVC.pikl'
    pickle.dump(linear_class,open(pikl_linear_SVC,'wb'))

    pikl_SVC = './SVC.pikl'
    pickle.dump(SVC_class,open(pikl_SVC,'wb'))

    print(cv_df.groupby('model_name').accuracy.mean())

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=0)

    linear_class.fit(X_train, y_train)

    y_pred = linear_class.predict(X_test)

    ## confusion_matrix graph.
    conf_mat = confusion_matrix(y_test, y_pred)

    end = time.time() - start
    print(" # Total time = ",end," seconds")

if __name__ == '__main__':
    func()
    joblib.dump(func,'func.pkl')
