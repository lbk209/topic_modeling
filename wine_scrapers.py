# selenium의 webdriver를 사용하기 위한 import
from selenium import webdriver

# selenium으로 키를 조작하기 위한 import
#from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

#from selenium.common.exceptions import ElementNotVisibleException, StaleElementReferenceException, NoSuchElementException

# 페이지 로딩을 기다리는데에 사용할 time 모듈 import
import time

#import pandas as pd 
#import numpy as np
#from tqdm import tqdm
#from datetime import datetime

#import os

def simple_scraper(url, *elements, by=By.CLASS_NAME, driver=None, quit_driver=True):
    if driver is None:
        driver = webdriver.Chrome() 
        driver.get(url)
        time.sleep(2)

    result = dict()
    for e in elements:
        l = driver.find_elements(by, e)
        result[e] = [x.text for x in l]

    if quit_driver:
        driver.quit()
    
    return (result, driver)