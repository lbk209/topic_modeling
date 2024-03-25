# selenium의 webdriver를 사용하기 위한 import
from selenium import webdriver

# selenium으로 키를 조작하기 위한 import
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.common.exceptions import ElementNotVisibleException, StaleElementReferenceException, NoSuchElementException

# 페이지 로딩을 기다리는데에 사용할 time 모듈 import
import time

import pandas as pd 
import numpy as np
from tqdm import tqdm
from datetime import datetime

import os

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as stutil
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WINE_STYLE = {
    'red': ['cabernet sauvignon', 'shiraz', 'syrah', 'pinot noir', 'merlot', 'carmenere', 'malbec'],
    'white': ['chardonnay', 'sauvignon blanc', 'riesling'],
    'sparkling': ['champagne', "moscato d'asti", 'cava'],
    #'desert': []
}


def collect_reviews(class_name, driver, by=By.CLASS_NAME):
    l = driver.find_elements(by, class_name)
    return [x.text for x in l]
    

def check_list(list_d, null='', cnt_max=3, consecutive=True):
    """
    check null reviews such as 
    ['', '', '', '', '', '', '', '', '', '', 'blackberry plum cherry oak']
    """
    if len(list_d) == 0:
        return False
    
    result = True
    cnt = 0
    for i, x in enumerate(list_d, start=1):
        if x == null:
           cnt += 1 
        if cnt >= cnt_max:
            result = False
            break
    
    if not (consecutive and i == cnt):
        result = True

    return result
    

def vivino_reviews(wine_url, wine_name, 
                   end_date = '20230101',
                   max_rev = 1e4, 
                   max_scr = 10, # max num of scroll for new review list 
                   time_scr = 2, # time to wait after scrolling. StaleElementReferenceException if not enough
                   loc1=None, 
                   loc2=None, 
                   loc3=None, 
                   loc4=None, # class name
                   loc5=None, # class name
                   # take the final review list as the review page has same old reiviews at the end every update by scroll-down 
                   final_only = True,
                   check_idx = 4, # community review list has 3 old reviews at the end of list every loading of new reviews
                   rev_date_format = '%b %d, %Y',
                   headless = False,
                   source='vivino'
                  ):

    #if headless: # not working
    if False:
        options = webdriver.ChromeOptions()
        options.add_argument(' --headless=new ')
    else:
        options = None

    driver = webdriver.Chrome(options=options) 
    driver.get(wine_url)
    time.sleep(3)

    loc4 = loc4.replace(' ', '.')
    loc5 = loc5.replace(' ', '.')

    try:
        # click all_reviews
        l = loc1
        driver.find_element(By.XPATH, loc1).click()
        time.sleep(2)
        
        # enter community reviews
        l = loc2
        driver.find_element(By.XPATH, loc2).click()
        time.sleep(1)
        
        # Show reviews by recent
        l = loc3
        driver.find_element(By.XPATH, loc3).click()
        time.sleep(1)

        l = loc4
        list_r = collect_reviews(loc4, driver)
        l = loc5
        list_d = collect_reviews(loc5, driver)
    except:
        print(f'ERROR) Check locator: {l}')
        driver.quit()
        return None

    # testing
    #return (list_r, list_d)
     
    reviews = list()
    dates = list()
    n_scr = 0

    #pbar = tqdm(total=max_rev, position=0)
    pbar = tqdm(position=0)
    
    while True:

        if check_list(list_d):
            n = len(list_r) - len(reviews)
        else:
            n = 0
            
        if n > 0: # get new n reviews
            if final_only: # replace review list with the latest scan
                reviews = list_r
                dates = list_d
            else: # update review list with new reviews from the latest scan
                reviews.extend(list_r[-n:])
                dates.extend(list_d[-n:])
            n_scr = 0
            pbar.update(n)
        else:
            n_scr += 1

        # testing
        #return (list_d, check_idx, rev_date_format)
        #return (reviews, dates)
        #return (list_r, list_d)
        #print(list_r)

        d1 = datetime.strptime(list_d[-check_idx], rev_date_format)
        d2 = datetime.strptime(end_date, '%Y%m%d')
        #print('testing:', d1)

        if ((len(reviews) > max_rev) or (d1 < d2)):
            #print(f'{len(reviews)} reviews collected.')
            break
        elif (n_scr > max_scr):
            # redundunt as n_try checking max_scr as well?
            print(f'WARNING: No additional reviews after {max_scr} reloading.')
            break
        else:
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
            time.sleep(time_scr)

            failed = True
            n_try = 0
            while failed:
                try:
                    list_r = collect_reviews(loc4, driver)
                    list_d = collect_reviews(loc5, driver)
                    failed = False
                except StaleElementReferenceException:
                    time.sleep(time_scr)
                    n_try += 1
                    
                if n_try > max_scr:
                    failed = False
                    print('WARNING: fail to collect all reviews')
        

    pbar.close()

    # close browser
    driver.quit()

    print(f'{wine_name}: {len(reviews)} reviews collected.')

    #return (dates, reviews) # testing

    # save result
    df_reviews = pd.DataFrame.from_dict({'date':dates, 'review':reviews})
    df_reviews['date'] = pd.to_datetime(df_reviews['date'], format=rev_date_format)
    df_reviews['source'] = source
    
    if False: #deprecated
        df_reviews.to_csv(filename, index=False)  

    return df_reviews


def generate_wine_id(df_reviews, id_start=0):
    if df_reviews.empty:
        wid = id_start
    else:
        wid = df_reviews['wid'].max()
        wid = id_start if wid is np.nan else wid+1 # check if no data row
    return wid


def concat_reviews(df_reviews, df, wine_name, col_rev, path=None, id_start=0):
    """
    save reviews of a wine and concat to existing reviews df_reviews 
    """
    wine_id = generate_wine_id(df_reviews, id_start=id_start)
    df[['wid', 'wine']] = [wine_id, wine_name]
    df = df.reindex(columns=col_rev)
    if path is not None:
        f = f'{path}/wine_{wine_id}.csv'
        df.to_csv(f, index=False)
        print(f'{f} saved.')
    df_reviews = pd.concat([df_reviews if not df_reviews.empty else None, df])
    return df_reviews


def save_reviews(df_review, file, path='data', overwrite=False):
    f = f'{path}/{file}'
    if os.path.exists(f) and not overwrite:
        print(f'ERROR: {f} already exists')
    else:
        df_review.to_csv(f, index=False)
        print(f'{f} saved.')
    return None


def load_reviews(file, path='data'):
    f = f'{path}/{file}'
    df = pd.read_csv(f, parse_dates=['date'])
    print(f'{f} loaded.')
    return df


def check_url(wines, print_parts=True, split='/', st_id='all-MiniLM-L6-v2', min_score=0):
    list_n = list()
    list_u = list()
    list_s = list()

    ss = SemanticSearch(embedding_model=st_id)
    
    for name, url in wines.items():
        parts = url.split(split)
        name_url, score = ss.quick(name, parts)
        
        list_n.append(name)
        list_u.append(name_url)
        list_s.append(score)

    df = (pd.DataFrame()
            .from_dict({'wine':list_n, 'url':list_u, 'similarity': list_s})
           .sort_values('similarity'))

    if print_parts:
        n = 5
        print(f'The top {n} pairs of least similarity:')
        _ = [print(f'{x[3]:.2f}) {x[1]}: {x[2]}') for x in df.head(n).to_records()]

    return df


def check_duplicated(df, cols=['wine','date','review'], drop=False):
    """
    check or drop duplicated reviews
    """
    if drop:
        df = df.drop_duplicates(cols)
    else:
        df = df.loc[df.duplicated(cols, keep=False)].sort_values(cols)
    return df


def find_style(name, style_dict=WINE_STYLE, semantic_search=None, threshold=0.5):
    """
    find the style of name from style_dict
    semantic_search: set to a instance of SemanticSearch for loop job
    """
    if semantic_search is None:
        semantic_search = SemanticSearch()
        
    res = dict()
    for s, words in style_dict.items():
        # 19 s ± 2.88 s
        #scores = [semantic_search.quick(x, name)[1] for x in words]
        #res[s] = max(scores)
        
        # same result with same speed (19.7 s ± 2.56)
        res[s] = semantic_search.quick(name, words)[1]
    
    #return [k for k,v in res.items() if v==max(res.values())][0]
    
    res = sorted(res.items(), key=lambda x:x[1], reverse=True)
    st, ms = res[0]
    if ms < threshold:
        return None
    else:
        return st



class SemanticSearch():
    def __init__(self, vocabulary=None, embedding_model='all-MiniLM-L6-v2'):
        if isinstance(embedding_model, str):
            embedding_model = SentenceTransformer(embedding_model)
        self.encode = lambda x: embedding_model.encode(x, convert_to_tensor=True)
        self.vocabulary = vocabulary
        if vocabulary is not None:
            self.voca_embeddings = self.encode(vocabulary)

    def search(self, query, top_k=3, min_score=0.3):
        """
        search the query in big self.vocabulary
        """
        if self.vocabulary is None:
            print('ERROR: Init with vocabulary first')
        else:
            vocabulary = self.vocabulary
        query_embedding = self.encode(query)

        # use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = stutil.cos_sim(query_embedding, self.voca_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        words = [vocabulary[i] for i in top_results[1]]
        scores = [x.item() for x in top_results[0]]

        res = {w:s for w,s in zip(words, scores) if s >= min_score}
        if len(res) == 0:
            print('No search result')
            res = None
        return res

    def quick(self, query, voca_small):
        """
        search the query in voca_small which is to be redefined in a loop
        """
        encode = self.encode
        if not isinstance(voca_small, list):
            voca_small = [voca_small]
        scores = [stutil.pytorch_cos_sim(encode(query), encode(x)) for x in voca_small]
        maxscore = max(scores).item()
        word = voca_small[scores.index(maxscore)]
        #return {word: maxscore}
        return (word, maxscore)
