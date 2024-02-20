from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

from bertopic_custom_util import utils


def bertopic_batch(docs,
                   ### hyperparams
                   min_df=0.001, # CountVectorizer
                   max_df=1.0,
                   n_components=15,
                   n_neighbors=10,
                   min_dist=0.1, # UMAP
                   min_cluster_size=10,
                   min_samples=None, # HDBSCAN
                   ####
                   embedding_model=None,
                   embeddings=None,
                   ngram_range=(1, 3),
                   prediction_data=True,
                   gen_min_span_tree=True,
                   top_n_words=5,
                   calculate_probabilities=False,
                   random_state=42,
                   verbose=False,
                   hdbscan_model=None,
                   custom_label='keybert'
                   ):

    #-- sub-models
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=ngram_range,
                                       min_df=min_df, max_df=max_df)

    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine', random_state=random_state)

    if hdbscan_model is None:
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=prediction_data,
            gen_min_span_tree=gen_min_span_tree
            )

    keybert = KeyBERTInspired()
    representation_model = {
        "KeyBERT": keybert
    }

    #-- train bertopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,

        top_n_words=top_n_words,
        calculate_probabilities=calculate_probabilities,
        verbose=verbose
    )

    # Train model
    try:
        topics, probs = topic_model.fit_transform(docs, embeddings)

        # set custom label
        if custom_label == 'keybert':
            topic_model_post = utils(topic_model)
            topic_model_post.set_custom_labels(name='KeyBERT')
            
    except Exception as e:
        print('ERROR', e)
        topic_model_post = None

    return topic_model_post



