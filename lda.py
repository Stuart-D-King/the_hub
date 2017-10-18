import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
from evolve_main import prep_data, process_text


def LDA(listings):
    '''
    Perform and visualize Latent Dirichlet Allocation (LDA) on the text corpus of listing descriptions.

    LDA represents documents as mixtures of topics, it also assumes a topic can be understood as a collection of words that have different probabilities of appearing in text discussing that topic. LDA is a probabilistic technique for topic modeling.
    '''
    hoods = ['Jamaica Plain', 'South End', 'Back Bay', 'Fenway', 'Dorchester']

    df_hoods = listings[listings['neighbourhood_cleansed'].isin(hoods)]
    df_hoods = df_hoods.reset_index(drop=True)
    df_hoods = df_hoods[df_hoods['description'].notnull()]

    X = df_hoods['description'].values
    new_X = []
    for description in X:
        new_X.append(process_text(description))

    max_features = 1000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=max_features, stop_words='english')

    tf = tf_vectorizer.fit_transform(new_X)

    n_topics = 10
    lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=10, learning_method='online', learning_offset=10., n_jobs=-1, random_state=42)

    lda_model.fit(tf)
    vis_data = pyLDAvis.sklearn.prepare(lda_model, tf, tf_vectorizer, R=n_topics, n_jobs=-1)
    # pyLDAvis.show(vis_data)
    pyLDAvis.save_html(vis_data, 'web_app/templates/pylda.html')

if __name__ == '__main__':
    listings, calendar = prep_data()

    LDA(listings)
