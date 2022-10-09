import numpy as np
import pandas as pd
import mlfuncs_clfn as ml
import nltk
import warnings

import streamlit as st
warnings.filterwarnings("ignore")
"""
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
"""
#nltk.download()
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
# importing data
df = pd.read_csv('Mental_Health_FAQ.csv', index_col=None)
df.columns = [i.lower() for i in df.columns]
new_df=df.copy()
#creating wrong question answer pairs

a=new_df["answers"][::-1]
a.reset_index(drop=True,inplace=True)

new_df["answers"]=a

df=df.append(new_df)

df.reset_index(drop=True,inplace=True)

#adding target column

target_1 =pd.Series(1, index=range(98))
target_0 =pd.Series(0, index=range(98))

target=target_1.append(target_0,ignore_index=True)
target.reset_index(drop=True,inplace=True)

df["Target"]=target

#shuffling the dataset

df.sample(frac=1).reset_index(drop=True)

#preprocessing the text

import nltk
import re

#nltk.download('wordnet')
#nltk.download('omw-1.4')

#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import string

stopword = set(stopwords.words('english'))


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [lemmatizer.lemmatize(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


df["questions"] = df["questions"].apply(clean)
df["answers"] = df["answers"].apply(clean)


import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


df["questions"] = df["questions"].apply(decontracted)
df["answers"] = df["answers"].apply(decontracted)

#dividing the dataset into Train ,eval and test sets

df_train, df_eval, df_test = ml.fn_tr_eval_ts_split_clf(df, eval_size = 0.2, ts_size = 0.2)

#statistical features
def fn_df_X_stat_feats(df):
    from fuzzywuzzy import fuzz

    str_lg_q = [len(i) for i in df.questions.values]
    str_lg_a = [len(i) for i in df.answers.values]
    n_words_a = [len(i.split()) for i in df.answers.values]
    n_words_q = [len(i.split()) for i in df.questions.values]

    common_words = [len(set(q.split()) & set(a.split())) for q, a in zip(df.questions.values, df.answers.values)]
    unique_words = [len(set(q.split()) | set(a.split())) for q, a in zip(df.questions.values, df.answers.values)]

    set_ratio = [fuzz.token_set_ratio(q, a) for q, a in zip(df.questions.values, df.answers.values)]
    sort_ratio = [fuzz.token_sort_ratio(q, a) for q, a in zip(df.questions.values, df.answers.values)]
    fuzz_ratio = [fuzz.QRatio(q, a) for q, a in zip(df.questions.values, df.answers.values)]
    partial_ratio = [fuzz.partial_ratio(q, a) for q, a in zip(df.questions.values, df.answers.values)]

    kw = dict(str_lg_q=str_lg_q, str_lg_a=str_lg_a, n_words_a=n_words_a,
              n_words_q=n_words_q, common_words=common_words, unique_words=unique_words,
              set_ratio=set_ratio, sort_ratio=sort_ratio, fuzz_ratio=fuzz_ratio,
              partial_ratio=partial_ratio, labels=df.Target.values)

    df_X_stat_feats = pd.DataFrame().assign(**kw)
    return df_X_stat_feats

df_train_stats = fn_df_X_stat_feats(df_train)
df_eval_stats = fn_df_X_stat_feats(df_eval)
df_test_stats = fn_df_X_stat_feats(df_test)

#standardizing the data

to_transform = [df_eval_stats, df_test_stats]


X_train= df_train_stats.iloc[:, :-1].values
X_eval= df_eval_stats.iloc[:, :-1].values
X_test= df_test_stats.iloc[:, :-1].values
y_train= df_train_stats.iloc[:, -1].values
y_eval= df_eval_stats.iloc[:, -1].values
y_test= df_test_stats.iloc[:, -1].values
# STANDARDIZE FEATURES:

def fn_tr_eval_ts_split_clf(df_Xy_, eval_size=0.2, ts_size=0.2):
    idxs_tr, idxs_ts_ = fn_tr_ts_split_clf(df_Xy_, ts_size=ts_size + eval_size)

    df_tr = df_Xy_.iloc[idxs_tr]
    df_ts_ = df_Xy_.iloc[idxs_ts_]

    idxs_eval, idxs_ts = fn_tr_ts_split_clf(df_ts_, ts_size=ts_size / (ts_size + eval_size))

    df_eval = df_ts_.iloc[idxs_eval]
    df_ts = df_ts_.iloc[idxs_ts]

    return df_tr, df_eval, df_ts

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_eval = sc.transform(X_eval)
X_test = sc.transform(X_test)

#best parameters for the predictions based on our analysis
param_grid_ = dict(penalty=['l1'],
                   C=[1],
                   solver=['saga'],
                   max_iter=[30_000],
                   random_state=[0],
                   )

param_grid = ml.fn_param_grid(param_grid_)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# train all the models
trained_models = ml.fn_train_models(X_train, y_train, LogisticRegression, param_grid)

model=trained_models[0]

st.title("Purchase Prediction app")
st.write("""Predict if user is buying the product based on age and salary""")








