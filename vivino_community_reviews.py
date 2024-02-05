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


def collect_reviews(class_name, driver, by=By.CLASS_NAME):
    l = driver.find_elements(by, class_name)
    return [x.text for x in l]
    

def vivino_reviews(wine_url, wine_name, 
                   end_date = '20230101',
                   max_rev = 1e4, 
                   max_scr = 10, # max num of scroll for new review list 
                   time_scr = 2, # time to wait after scrolling. StaleElementReferenceException if not enough
                   loc1=None, 
                   loc2=None, 
                   loc3=None, 
                   loc4=None,
                   # take the final review list as the review page has same old reiviews at the end every update by scroll-down 
                   final_only = True,
                   check_idx = 4, # community review list has 3 old reviews at the end of list every loading of new reviews
                   rev_date_format = '%b %d, %Y',
                   #headless = False # fail to locate loc1 if set to True
                  ):

    #if headless:
    if False:
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
    else:
        options = None

    driver = webdriver.Chrome(options=options) 
    driver.get(wine_url)
    time.sleep(3)

    loc3 = loc3.replace(' ', '.')
    loc4 = loc4.replace(' ', '.')

    try:
        # enter community reviews
        l = loc1
        driver.find_element(By.XPATH, loc1).click()
        time.sleep(1)
        
        # Show reviews by recent
        l = loc2
        driver.find_element(By.XPATH, loc2).click()
        time.sleep(1)

        l = loc3
        list_r = collect_reviews(loc3, driver)
        l = loc4
        list_d = collect_reviews(loc4, driver)
    except:
        print(f'ERROR) Check locator: {l}')
        driver.quit()
        return None
     
    reviews = list()
    dates = list()
    n_scr = 0

    #pbar = tqdm(total=max_rev, position=0)
    pbar = tqdm(position=0)
    
    while True:
        n = len(list_r) - len(reviews)
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
        #return list_d

        d1 = datetime.strptime(list_d[-check_idx], rev_date_format)
        d2 = datetime.strptime(end_date, '%Y%m%d')
        #print('testing:', d1)

        if ((len(reviews) > max_rev) or (d1 < d2)):
            print(f'{len(reviews)} reviews collected.')
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
                    list_r = collect_reviews(loc3, driver)
                    list_d = collect_reviews(loc4, driver)
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

    # save result
    df_reviews = pd.DataFrame.from_dict({'date':dates, 'review':reviews})
    df_reviews['date'] = pd.to_datetime(df_reviews['date'])
    
    if False: #deprecated
        df_reviews.to_csv(filename, index=False)  

    return df_reviews


def generate_wine_id(df_reviews, id_start=0):
    if df_reviews.empty:
        wid = id_start
    else:
        wid = df_reviews.id.max()
        wid = id_start if wid is np.nan else wid+1 # check if no data row
    return wid


def concat_reviews(df_reviews, df, wine_name, col_rev, save=True, id_start=0):
    wine_id = generate_wine_id(df_reviews, id_start=id_start)
    df[['id', 'wine']] = [wine_id, wine_name]
    df = df.reindex(columns=col_rev)
    if save:
    	f = f'wine_{wine_id}.csv'
        df.to_csv(f, index=False)
    	print(f'{f} saved.')
    df_reviews = pd.concat([df_reviews if not df_reviews.empty else None, df])
    return df_reviews
