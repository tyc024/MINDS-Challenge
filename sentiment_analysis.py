import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
from flair.models import TextClassifier
from flair.data import Sentence
import emoji
import plotly.express as px
import math
from collections import defaultdict

# preprocessing
CONVERT_QUOTES = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”–-",  u"'''\"\"--") ] ) 
def give_emoji_free_text(text):
    text = text.encode(encoding='utf-8')
    return emoji.get_emoji_regexp().sub(r'', text.decode('utf8'))
def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def has_shib_doge(s):
    s = s.lower()
    return "shib" in s or "doge" in s
def combine_sentence(message_list):
    msg = ""
    for i in message_list:
        if type(i) == str:
            msg += i
    return msg

def preprocess(messages):
    df = pd.DataFrame()
    messages_text = []
    messages_date = []
    for m in messages:
        messages_text.append(m['text'])
        date = m['date'][:10]
        messages_date.append(date)
    
    clean_messages = []
    clean_dates = []
    for m, d in tqdm(zip(messages_text, messages_date)):
        if type(m) != str:
            m = combine_sentence(m)
        m = give_emoji_free_text(m)
        m = m.translate(CONVERT_QUOTES)
        if has_shib_doge(m) and is_english(m):
            clean_messages.append(m)
            clean_dates.append(d)
 
    df['text'] = clean_messages
    df['date'] = clean_dates
    return df

# sentiment analysis
def flair_prediction(nlp, x):
    sentence = Sentence(x)
    nlp.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "pos"
    elif "NEGATIVE" in str(score):
        return "neg"
    else:
        return "neu"
    
def flair_prediction_value(nlp, x):
    sentence = Sentence(x)
    nlp.predict(sentence)
    score = sentence.labels[0]
    return score.score

def sentiment_analysis(sentiment_nlp, df, out_filename):
    df['sentiment'] = df['text'].apply(lambda x: flair_prediction(sentiment_nlp, x))
    df['score'] = df['text'].apply(lambda x: flair_prediction_value(sentiment_nlp, x))
    df.to_csv(out_filename, header=True, index=True)
    return df

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def plot(date_score, df):
    fig = px.bar(df, x = "date", y = "text")
    fig.show()
    fig = px.line(x=date_score.keys(), y=date_score.values(), text=[truncate(x, 2) for x in date_score.values()])
    fig.show()


if __name__=="__main__":
    filename = "messages.json"
    with open(filename, "r") as f:
        chatroom = json.load(f)
    messages = chatroom['messages']
    
    print("Start preprocessing...")
    df = preprocess(messages)
    print("End+++++++++++++")

    print("Start sentiment analysis...")
    sentiment_nlp = TextClassifier.load('sentiment-fast')
    out_filename = "messages_sentiment.csv"
    df = sentiment_analysis(sentiment_nlp, df, out_filename)
    print("End++++++++++++++")

    # count messages per day
    count_df = df.groupby(['date']).count()
    count_df = count_df.reset_index()
    # count pos, neg messages per day
    count_df_posneg = df.groupby(['date', 'sentiment']).count()

    # compute average sentiment per day, 1 for pos 0 for neg
    date_score = defaultdict(int)
    for idx, row in count_df_posneg.reset_index().iterrows():
        if row[1] == 'neg':
            date_score[row[0]] = row[2]
        else:
            date_score[row[0]] = row[2]/(date_score[row[0]]+row[2]) # num of pos messages / num of all messages
    
    plot(date_score, count_df)
