import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from evolve_main import prep_data, process_text

'''
Analyze neighborhood overview text by neighborhood by creating a figure that displays the top 20 words by average tf-idf score to understand the presence of common terms/features and derive key characteristics of each neighborhood. The below code was developed with reference to Thomas Buhrmann's blog post, Analyzing tf-idf results in scikit-learn (https://buhrmann.github.io/tfidf-analysis.html)
'''

def tf_idf(listings):
    hoods = ['Jamaica Plain', 'South End', 'Back Bay', 'Fenway', 'Dorchester']

    df_hoods = listings[listings['neighbourhood_cleansed'].isin(hoods)]
    df_hoods = df_hoods.reset_index(drop=True)
    df_hoods = df_hoods[df_hoods['neighborhood_overview'].notnull()]

    nhoods = df_hoods['neighbourhood_cleansed'].values

    X = df_hoods['neighborhood_overview'].values
    new_X = []
    for overview in X:
        new_X.append(process_text(overview))

    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')

    vec_pipe = Pipeline([
    ('vec', tfidf)])

    X_trans = vec_pipe.fit_transform(new_X)
    vec = vec_pipe.named_steps['vec']
    features = vec.get_feature_names()
    return X_trans, nhoods, features


def top_tfidf_feats(row, features, top_n=20):
    '''
    Get the top n tfidf values in a row and return them with their corresponding feature names
    '''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(X, features, row_id, top_n=20):
    '''
    Return the top n tfidf features in a specific document
    '''
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=20):
    '''
    Return the top n features that on average are most important among documents in rows indentified by indices in grp_ids
    '''
    if grp_ids:
        X_new = X[grp_ids].toarray()
    else:
        X_new = X.toarray()

    X_new[X_new < min_tfidf] = 0

    tfidf_means = np.mean(X_new, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_nhood(X, nhoods, features, min_tfidf=0.1, top_n=20):
    '''
    Return a list of dataframes (dfs), where each df holds the top_n features and their mean tfidf value calculated across documents with the same class label (neighborhood name)
    '''
    dfs = []
    labels = np.unique(nhoods)
    for label in labels:
        ids = np.where(nhoods==label)
        feats_df = top_mean_feats(X, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_dfs(dfs):
    '''
    Plot the dataframes returned by the function top_feats_by_nhood()
    '''
    fig = plt.figure(figsize=(12, 8), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title(str(df.label), fontsize=14)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='blue', alpha=0.5)
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)

    plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.9, wspace=0.52)
    plt.suptitle('Mean Tf-Idf Score for Top 20 Words by Neighborhood Overview', fontsize=16)
    # plt.show()
    plt.savefig('img/tfidf_means.png', dpi=400)
    plt.close()


if __name__ == '__main__':
    listings, calendar = prep_data()

    X, nhoods, features = tf_idf(listings)
    dfs = top_feats_by_nhood(X, nhoods, features)
    plot_tfidf_dfs(dfs)
