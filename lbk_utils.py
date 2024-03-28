import os, re
import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as stutil

import torch


def read_csv(file, path='.', cols_eval=None, **kwargs):
    """
    file: str or re pattern. ex) r'wine_\d+\.csv' for wine_45.csv
    kwargs: keyword args for pd.read_csv
    """
    #files = [x for x in os.listdir(path) if x.startswith(file)]
    files = [s for s in os.listdir(path) if re.match(file, s)]

    if len(files) == 0:
        print('ERROR!: No csv to read')
        return None

    if cols_eval is None:
        converters=None
    else:
        if not isinstance(cols_eval, list):
            cols_eval = [cols_eval]
        converters= {c: lambda x: eval(x) for c in cols_eval}

    df_reviews = pd.DataFrame()
    for f in files:
        try:
            df = pd.read_csv(f'{path}/{f}', converters=converters, **kwargs)
            df_reviews = pd.concat([df_reviews, df])
        except SyntaxError:
            print('ERROR: check converters arg first.')
            return None

    return df_reviews.reset_index(drop=True)
    
    
def save_csv(df, file, path='.', overwrite=False, index=False, print_out=True):
    f = f'{path}/{file}'
    if os.path.exists(f) and not overwrite:
        print(f'ERROR: {f} already exists')
    else:
        df.to_csv(f, index=index)
        if print_out:
            print(f'{f} saved.')
    return None

    
def split_str(string, length=50, split='\n', indent=''):
    words = string.split()

    words_split = ''
    current_length = 0
    for i, word in enumerate(words):
        if current_length + len(word) <= length:
            words_split += f'{word} '
            current_length += len(word) + 1  # +1 for the space
        else:
            words_split += f'{split}{indent}{word} '
            current_length = len(word) + len(indent) + 1
    return words_split


def println(input_string, line_length=50, split='\n', indent='  '):
    ws = split_str(input_string, length=line_length, split=split, indent=indent)
    print(ws)
    
    
class SemanticSearch():
    def __init__(self, vocabulary=None, embedding_model='all-MiniLM-L6-v2'):
        """
        vocabulary: list of words
        """
        if isinstance(embedding_model, str):
            embedding_model = SentenceTransformer(embedding_model)
        self.encode = lambda x: embedding_model.encode(x, convert_to_tensor=True)
        self.vocabulary = vocabulary
        if vocabulary is not None:
            self.voca_embeddings = self.encode(vocabulary)


    def search_k(self, queries, top_k=3, top_k_max=100):
        """
        search the queries in big self.vocabulary
        queries: a query word or list of queries
        """
        if self.vocabulary is None:
            print('ERROR: Init with vocabulary first')
            return None

        vocabulary = self.vocabulary
        top_k = min(top_k, top_k_max, len(vocabulary))
        queries_embedding = self.encode(queries)

        # use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = stutil.cos_sim(queries_embedding, self.voca_embeddings)
        # top_results[0] top_k scores for each query in queries
        top_results = torch.topk(cos_scores, k=top_k)

        result = dict()
        result['word'] = [[vocabulary[i] for i in x] for x in top_results[1]]
        result['score'] = top_results[0].tolist()

        return result


    def search(self, queries, threshold=0.7, top_k_max=100,
               exclude_no_result=True, print_out=True, sort_result=True):
        
        if isinstance(queries, str):
            queries = [queries]

        res_k = self.search_k(queries, top_k=top_k_max)
        if res_k is None:
            return None

        result = dict()
        for i, q in enumerate(queries):
            result[q] = [w for w,s in zip(res_k['word'][i], res_k['score'][i]) if s >= threshold]

        if exclude_no_result:
            result = {k:v for k,v in result.items() if len(v) > 0}

        if sort_result:
            #result = sorted(result)
            pass

        if print_out:
            _ = [print(f'{k}: {", ".join(v)}') for k,v in result.items()]

        return result


    def check_existence(self, queries, threshold=0.5, top_k_max=10,
                        return_new=True, print_out=True, sort=True):
        """
        check if the items in queries exit in self.vocabulary
        queries: a query word or list of queries
        """
        result = self.search(queries, top_k=top_k_max)
        if result is None:
            return None

        words_existing = {}
        words_new = {}
        for i, q in enumerate(queries):
            score = result['score'][i][0]
            word = result['word'][i][0]
            if score >= threshold:
                words_existing[q] = [word, round(score, 3)]
            else:
                words_new[q] = [word, round(score, 3)]


        df_existing = pd.DataFrame.from_dict(words_existing, orient='index', columns=['result', 'score']).rename_axis('query')
        df_new = pd.DataFrame.from_dict(words_new, orient='index', columns=['result', 'score']).rename_axis('query')
        if sort:
            df_existing = df_existing.sort_values('score', ascending=False)
            df_new = df_new.sort_values('score', ascending=False)
        df_existing = df_existing.reset_index()
        df_new = df_new.reset_index()

        lf = ''
        if print_out:
            if len(df_existing) > 0:
                print('Existing words (query: result)')
                _ = [print(f'{idx}: {q} / {r} / {s}') for idx, (q, r, s) in df_existing.iterrows()]
                lf = '\n'

            if len(df_new) > 0:
                print(f'{lf}New words (query: result)')
                _ = [print(f'{idx}: {q} / {r} / {s}') for idx, (q, r, s) in df_new.iterrows()]
                lf = '\n'

        if return_new:
            print(f'{lf}Returning new words')
            return df_new
        else:
            print(f'{lf}Returning existing words')
            return df_existing


    def quick(self, query, voca_small):
        """
        search the query in voca_small which is to be redefined in a loop
        query: str
        """
        encode = self.encode
        if not isinstance(voca_small, list):
            voca_small = [voca_small]
        scores = [stutil.pytorch_cos_sim(encode(query), encode(x)) for x in voca_small]
        maxscore = max(scores).item()
        word = voca_small[scores.index(maxscore)]
        #return {word: maxscore}
        return (word, maxscore)
