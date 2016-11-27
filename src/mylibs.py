from flask import Flask, render_template, request, session, make_response

import tempfile
import matplotlib
matplotlib.use('Agg') # this allows PNG plotting
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import json
import requests
from datetime import date
from wordcloud import WordCloud, STOPWORDS
import sys, string
import seaborn as sns
from statsmodels.tsa.arima_model import ARMA
plt.style.use('fivethirtyeight')
#import pymc


#-------- MODEL GOES HERE -----------#

def make_url(query, offset, rows, since):
    """a funtion to make the api url"""
        
    url = 'https://api.crossref.org/works?query.title={}&rows={}'.format(query,rows)
    url +='&offset={}'.format(offset)
    url +='&filter=from-pub-date:{}'.format(since)
    return url


# In[32]:

def update_dicts(api_content, auth_nums, auth_cits, year_nums, year_text):
    """
    a function to extract data from each api query 
       and update them to dictionaries
    """
    
    for item in api_content:
        try:
            item['author']
            for author in item['author']:
                name = author['given'].title() + ' ' + author['family'].title()
                auth_nums[name] = auth_nums.get(name,0) + 1
                auth_cits[name] = auth_cits.get(name,0) + item['reference-count']
        except:
            pass
        
        year = item['issued'].values()[0][0][0]        
        year_text[year] = year_text.get(year, '') + ' ' + item['title'][0]
        year_nums[year] = year_nums.get(year, 0) + 1

    return auth_nums, auth_cits, year_nums, year_text


def add_stopwords(query):
    add_stop = query.lower().split() + ['using', 'based', '-based', 'system', 'systems', 'study', 'studies', 'analysis']
    
    for p in ':,"?':
        for w in query.lower().split():
            add_stop.append(w+p)
    return add_stop


# In[34]:

def make_df(dic):
    dic_df = pd.DataFrame( dic.values(), dic.keys())
    dic_df.columns = ['counts']
    return dic_df


# ### Initial Author Graphs

# In[35]:

def plot_authors(nums_df, cits_df, howmany=-9,w=10.0, h=3.3):
    """ a funtion to plot top authors"""
        
    fig = plt.subplots(figsize=(w,h))
    
    ax1 = plt.subplot(121)
    nums_df = nums_df.sort_values(by='counts')
    nums_df[howmany:].plot(kind='barh', color='darkred', width=0.8, ax=ax1)
 #   plt.xlabel('counts', fontsize=12)
    plt.title('Top authors by # of publications', fontsize=12)

    ax2 = plt.subplot(122)
    cits_df = cits_df.sort_values(by='counts')
    cits_df[howmany:].plot(kind='barh', color='y', width=0.8, ax=ax2)

    plt.title('Top cited authors', fontsize=12)
 #   plt.xlabel('counts', fontsize=12)   
    plt.tight_layout()

    # make the temporary file
    f = tempfile.NamedTemporaryFile(
        dir='static/temp',
        suffix='.png',delete=False)
    # save the figure to the temporary file
    plt.savefig(f, dpi=400)
    f.close() # close the file
    # get the file's name (rather than the whole path)
    # (the template will need that)
    plotPng = f.name.split('/')[-1]

    return plotPng 



# ### Getting Top Words and Wordcloud

# In[37]:

def show_wordcloud(text_lst, add_stops, title, w=6):
        
    fig = plt.subplots(figsize=(w,w))
    ax = plt.subplot(111) 

    txt = string.translate( ' '.join(text_lst), string.punctuation)
    
    word_bag = ' '.join( [word for word in txt.split()
                         if word.lower() not in add_stops] )
    
    cloud = WordCloud(
                         stopwords=STOPWORDS,
                         background_color='black',
                         width=1200,
                         height=1200).generate(word_bag)
    
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()

    # make the temporary file
    f = tempfile.NamedTemporaryFile(
        dir='static/temp',
        suffix='.png',delete=False)
    # save the figure to the temporary file
    plt.savefig(f, dpi=400)
    f.close() # close the file
    # get the file's name (rather than the whole path)
    # (the template will need that)
    plotPng = f.name.split('/')[-1]

    return plotPng


# In[38]:

def word_counter(txt, add_stops):
    d = {}
    for word in txt.split():
        try: 
            word = word.encode('ascii').lower()
            word = word.translate(None, string.punctuation)
            
            if word not in STOPWORDS and word not in add_stops:
                d[word] = d.get(word,0) + 1
        except:
            continue
    lst = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return lst


# In[39]:

def get_top_words(year_text, add_stops, n_words=15):
    
    txt = string.translate( ' '.join( year_text.values()), string.punctuation)
    popular = word_counter(txt, add_stops)[:n_words]
    
    common_words = {}
    for key in year_text.keys():
        for ii in range(n_words):
            word = popular[ii][0]
            c = year_text[key].lower().split().count(word)
            common_words[key] = common_words.get(key, []) + [c]
            
    common_words_df = pd.DataFrame(common_words).T
    common_words_df.columns = [popular[ii][0] for ii in range(n_words)] 
    
    return common_words_df


# In[71]:

def plot_top_words(common_words_df, w=10, h=4.0):
    fig = plt.subplots(figsize=(w,h))
    ax = plt.subplot(111) 

    print '\nOCCURENCE OF TOP FREQUENT WORDS'
    common_words_df.plot(ax=ax, linewidth=2.6)    
    plt.ylabel('Word occurence', fontsize=13)
#    plt.title('All-time top {} most popular words'.format(common_words_df.shape[1]) , fontsize=15)
    plt.tight_layout()

    # make the temporary file
    f = tempfile.NamedTemporaryFile(
        dir='static/temp',
        suffix='.png',delete=False)
    # save the figure to the temporary file
    plt.savefig(f, dpi=400)
    f.close() # close the file
    # get the file's name (rather than the whole path)
    # (the template will need that)
    plotPng = f.name.split('/')[-1]

    return plotPng


# In[41]:

def projected_bag(common_words_df):
    bag = ' '

    common_df = common_words_df[-16:-2]
    
    for col in common_df.columns:
        t = pd.DataFrame(common_df[col])
        try:
            pred_t = arma_preds(t, p_max=5, q_max=5, start_date=0, end_date=5)
            count =  int(sum(list(pred_t.preds)[1:]))
            w = str(col) + ' '
            bag += w*count
        except:
            continue
    return bag


# In[42]:

def show_projected_wordcloud(bag, year_text, add_stops, title, w=6):

    fig = plt.subplots(figsize=(w,w))
    ax = plt.subplot(111) 
    today = date.today().year
    
    word_bag = ' '.join( [word for word in year_text[today].split()
                         if word.lower() not in add_stops] )  # get the words for this year
    word_bag += bag
    
    cloud = WordCloud(
                         stopwords=STOPWORDS,
                         background_color='black',
                         width=1200,
                         height=1200).generate(word_bag)

    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()

    # make the temporary file
    f = tempfile.NamedTemporaryFile(
        dir='static/temp',
        suffix='.png',delete=False)
    # save the figure to the temporary file
    plt.savefig(f, dpi=400)
    f.close() # close the file
    # get the file's name (rather than the whole path)
    # (the template will need that)
    plotPng = f.name.split('/')[-1]

    return plotPng


# ### MCMC simulations (with PYMC) of expected publication numbers per year

def mcmc_runs(df, mc_steps):

    n_year = len(df)

    alpha = 1.0 / df.mean() 
    lambda_1 = pymc.Exponential("lambda_1", alpha)
    lambda_2 = pymc.Exponential("lambda_2", alpha)
    lambda_3 = pymc.Exponential("lambda_3", alpha)

    tau1 = pymc.DiscreteUniform("tau1", lower=0, upper=n_year) 
    tau2 = pymc.DiscreteUniform("tau2", lower=0, upper=n_year)

    @pymc.deterministic
    def lambda_(tau1=tau1, tau2=tau2, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3):
        out = np.zeros(n_year)
        out[:tau1]     = lambda_1  # lambda before tau is lambda1
        out[tau1:tau2] = lambda_2  # lambda after (and including) tau is lambda2
        out[tau2:]     = lambda_3
        return out 
    
    observation = pymc.Poisson("obs", lambda_, value=df, observed=True)
    model = pymc.Model([observation, lambda_1, lambda_2, lambda_3, tau1, tau2]) 
    
    print ('\nRunning MCMC simulations to estimate the expected number of publications per year')
 
    mcmc = pymc.MCMC(model)
    mcmc.sample(mc_steps, 10000, 1)

    lambda_1_samples = mcmc.trace('lambda_1')[:]
    lambda_2_samples = mcmc.trace('lambda_2')[:]
    lambda_3_samples = mcmc.trace('lambda_3')[:]

    tau1_samples = mcmc.trace('tau1')[:]
    tau2_samples = mcmc.trace('tau2')[:]   

    N = tau1_samples.shape[0]
    
    expected_per_year = np.zeros(n_year)

    for year in range(n_year):
        ix1 = year < tau1_samples
        ix3 = year > tau2_samples
        ix2 = ~(ix1 + ix3)

        expected_per_year[year] = (lambda_1_samples[ix1].sum() + lambda_2_samples[ix2].sum()
                                  + lambda_3_samples[ix3].sum()) / N        
    return expected_per_year


# ### ARIMA Models


def index_date_convert(df2):
    def date_convert(aa):
        return str(aa)+'-01-01'

    df2['year'] = df2.index
    df2.year = df2.year.apply(date_convert)
    df2.year = pd.to_datetime(df2.year)
    df2.index = df2.year
    df2.set_index('year', inplace=True)
    return df2

def arma_preds(df2, p_max=5, q_max=5, start_date=-10, end_date=5):    
    
    df2 = index_date_convert(df2)
    
    for col in df2.columns:
        df2[col] = df2[col].astype(float)   
    
    # GridSearch for p, q values of ARIMA model    
    d = {}
    for p in range(1, p_max+1):
        for q in range(1, q_max+1):
            try:
                arma_model = ARMA(df2,(p,q)).fit()
                if arma_model.aic > 0:
                    d[(p,q)] = arma_model.aic
            except:
                continue
                
    lst = sorted(d.items(), key=lambda x: x[1], reverse=False)    
    p, q = lst[0][0]
    
    # fitting to get predictions
    from_date = str(date.today().year + start_date) + '-01-01'
    to_date   = str(date.today().year + end_date) + '-01-01'
    
    arma_model = ARMA(df2,(p,q)).fit()
    preds = arma_model.predict(from_date, to_date)
    preds_df = pd.DataFrame(preds, columns=['preds'])    
    preds_df.index = preds_df.index.year
    
    return preds_df



def plot_year_nums(query, lim, years, df, mc_steps, w=10, h=3.6):
    
    df1 = df[-41:]
#    mc_expected = mcmc_runs(df[-41:-1], mc_steps=mc_steps)
    
    fig = plt.subplots(figsize=(w,h)) 
    
    df2 = df[-26:-1].copy()
    df1.counts.plot(kind='bar', width=0.9, label="observed number of publications per year") 
    
    print ('\n\nSearching for the best Time-Series ARIMA model to predict the future values ...')        
    try:
        preds_df = arma_preds(df2, p_max=5, q_max=5, start_date=-15, end_date=5) 
        df1 = df1.join(preds_df, how='outer')        
        df1.preds.plot(kind='bar', width=0.5, label="ARIMA model", color='r', alpha=0.8)    
    except:
        pass
    
    # plot expected publications per year with MCMC    
#    plt.plot(range(len(mc_expected)), mc_expected, lw=6, label="expected publications per year (simulated with MCMC)", color='y')
   
    plt.title('# of publications for query = \'{}\' (analysis of {} most relevant articles)'.format(query, lim), fontsize=13) 
    
    plt.ylabel('counts', fontsize=12)    
    plt.legend(loc="best", fontsize=11)  

    plt.tight_layout()

    # make the temporary file
    f = tempfile.NamedTemporaryFile(
        dir='static/temp',
        suffix='.png',delete=False)
    # save the figure to the temporary file
    plt.savefig(f, dpi=400)
    f.close() # close the file
    # get the file's name (rather than the whole path)
    # (the template will need that)
    plotPng = f.name.split('/')[-1]

    return plotPng


# ### Getting API Results


def get_api_data(query, lim, since):
    query = '+' + query.replace(' ','+')
    
    rows = 1000
    auth_nums = {}
    auth_cits = {}
    year_nums = {}
    year_text = {}    
    
    print "\nData mining in progress, acquiring:"
    for offset in range(0, lim, rows):
        print '\r', ' '*34, '{} of {} most relevant articles ...'.format(offset+rows,lim)

        url = make_url(query, offset, rows, since)

        try:
            json_dict = json.loads(requests.get(url).text) 
            api_content = json_dict['message']['items']
            auth_nums, auth_cits, year_nums, year_text=                 update_dicts(api_content, auth_nums, auth_cits, year_nums, year_text)

        except:
            print '\r', ' '*34, '... oops! connection timeout :(',
            continue

    try:
        auth_nums_df = make_df(auth_nums)
        auth_cits_df = make_df(auth_cits)
        year_nums_df = make_df(year_nums)
        
    except:
        print ('\nAPI issues, data mining was unsuccesful, please try again later')
        sys.exit(0)
        
    print '\nThe analysis for your searh query was finished, a report will now be prepared'
    
    return auth_nums_df, auth_cits_df, year_nums_df, year_text



