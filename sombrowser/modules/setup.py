import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer


def get_meta(meta_path, rp):

    meta_path_ = [rp] + meta_path.split("/")

    meta_df = pd.read_csv(os.path.join(*meta_path_))

    print('load from csv:', meta_path)

    return meta_df


def get_pickled_som(pickle_path, rp):
    print('getting pickled som from ', pickle_path)

    pickle_path_ = [rp] + pickle_path.split("/")
    with open(os.path.join(*pickle_path_), 'rb') as infile:
        som = pickle.load(infile)
        print('loading pickled som')
    return som


def get_transformer(rp, dc, modelname='all-MiniLM-L6-v2'):
    """
    Use pre-downloaded model if it exists, 
    otherwise download this model and save it locally.
    """

    models_path_ = [rp] + dc['data']['vectorizer'].split("/")
    models_path = os.path.join(*models_path_)

    if os.path.exists(models_path):
        print('Using pre-loaded model:', modelname)
        return SentenceTransformer(models_path)
    else:
        print('Downloading transformer model', modelname)
        transformer = SentenceTransformer(modelname)
        transformer.save(models_path)
        return transformer
