import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from evolve_main import prep_data, process_text

'''
Apply KMeans to tf-idf features to produce topic centroids, which can then be used to perform topic modeling of the listings' descriptions for five neighborhoods - Jamaica Plain, South End, Back Bay, Fenway, and Dorchester.
'''


def tf_idf(listings):
    '''
    Perform tf-idf vectorization on the description column from the listings dataset. Return the neighborhood labels, the tf-idf's vocabulary, and the transformed tf-idf matrix.
    '''
    hoods = ['Jamaica Plain', 'South End', 'Back Bay', 'Fenway', 'Dorchester']

    df_hoods = listings[listings['neighbourhood_cleansed'].isin(hoods)]
    df_hoods = df_hoods.reset_index(drop=True)
    df_hoods = df_hoods[df_hoods['description'].notnull()]

    nhoods = df_hoods['neighbourhood_cleansed'].values

    X = df_hoods['description'].values
    new_X = []
    for description in X:
        new_X.append(process_text(description))

    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')

    tfidf.fit(new_X)
    vocab = tfidf.vocabulary_
    mtrx = tfidf.transform(new_X).todense()
    mtrx = np.array(mtrx)

    return nhoods, vocab, mtrx


def fit_kmeans(X, n_clust):
    '''
    Fit the scikit-learn KMeans clustering algorithm to X for n clusters. Return the cluster lables and centroids.
    '''
    km = KMeans(n_clusters=n_clust, init='random', n_init=10, max_iter=100, n_jobs=-1, random_state=31337)
    km.fit_predict(X)
    return km.labels_, km.cluster_centers_


def kmean_score(X, n_clust):
    '''
    Fit the scikit-learn KMeans clustering algorithm to X for n clusters. Return the RSS for the fitted model.
    '''
    km = KMeans(n_clusters=n_clust, init='random', n_init=10, max_iter=50, n_jobs=-1, random_state=31337)
    km.fit(X)
    rss = -km.score(X)
    return rss


def plot_scores(scores):
    '''
    Plot the RSS scores for KMeans models with the number of clusters ranging from 1 to 10.
    '''
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(range(2,11), scores, 'o--')
    ax.set_xlabel('K')
    ax.set_ylabel('RSS')
    ax.set_title('RSS versus K')

    plt.savefig('img/kmeans_scores.png', dpi=400)
    plt.close()


def print_top_10(centroids, vocab):
    '''
    Print the top ten most words with the highest tf-idf score for each cluster.
    '''
    vocab_dct = dict((v,k) for k,v in vocab.items())
    for idx, row in enumerate(centroids):
        top10 = row.argsort()[::-1][:10]
        print('Cluster #{}'.format(idx+1))
        for i in top10:
            print(vocab_dct[i])
        print('---------------')


def nhoods_by_cluster(labels, nhoods):
    '''
    Print a random sample of 10 neighborhood names for each cluster.
    '''
    for i in range(10):
        c_inds = np.where(labels == i)
        inds = c_inds[0]
        samp_inds = np.random.choice(inds, size=10, replace=False)
        samp_hoods = nhoods[samp_inds]

        print('Cluster #{}'.format(i+1))
        for samp in samp_hoods:
            print(samp)
        print('-----------')


if __name__ == '__main__':
    listings, calendar = prep_data()

    nhoods, vocab, mtrx = tf_idf(listings)

    # Plot the RSS score different values of 'k'
    # scores = [kmean_score(mtrx, i) for i in range(2,11)]
    # plot_scores(scores)

    labels, centroids = fit_kmeans(mtrx, 10)
    print_top_10(centroids, vocab)
    # nhoods_by_cluster(labels, nhoods)
