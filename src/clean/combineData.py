import pandas as pd
import numpy as np
import glob
import re

np.random.seed(42)

from nltk import PorterStemmer
from nltk.corpus import stopwords

def makeStopwords():
    stopwordsNLTK = stopwords.words('english')
    stopwordsNLTK.remove('not')
    stopwordsDomainSpecific = [
        'l',
        'year',
        'speaker',
        'chairman',
        'mr',
        'mrs',
        'also',
    ]
    return stopwordsNLTK + stopwordsDomainSpecific

REGEX_TO_SPACE = re.compile('-+')
REGEX_TO_REMOVE = re.compile('[^a-zA-Z ]+')
REGEX_SPACES = re.compile('\s+')
STOPWORDS = makeStopwords()

PORTER_STEMMER = PorterStemmer()

def cleanSentence(sentence):
    sentence = re.sub(REGEX_TO_SPACE, ' ', sentence)
    sentence = re.sub(REGEX_TO_REMOVE, '', sentence)
    sentence = sentence.lower().strip()
    words = re.split(REGEX_SPACES, sentence)
    words = [PORTER_STEMMER.stem(word) for word in words if word not in STOPWORDS]
    if words:
        sentence = ' '.join(words)
    else:
        sentence = ''
    return sentence

files = glob.glob("../../data/csv/processed/record*.csv")
data = pd.concat([pd.read_csv(f, index_col=0) for f in files], keys=files, ignore_index=True)

remove_speakers = ['The PRESIDING OFFICER.',
                   'The SPEAKER pro tempore.']
data = data[~data.speaker.isin(remove_speakers)]

# data['speech'] = data['speech'].str.replace('Mr\.', 'Mr')
# data['speech'] = data['speech'].str.replace('Mrs\.', 'Mrs')
# data['speech'] = data['speech'].str.replace('Ms\.', 'Ms')
# data['speech'] = data['speech'].str.replace('Dr\.', 'Dr')
# data['speech'] = data['speech'].str.replace('a\.m\. ', 'am ')
# data['speech'] = data['speech'].str.replace('p\.m\. ', 'pm ')
# data['speech'] = data['speech'].str.replace('H\.R\. ', 'HR ')
wordsWithPeriodsRegex = 'Mr\.|Mrs\.|Ms\.|Dr\.|a\.m\.|p\.m\.|H\.R\.'
data['speech'] = data['speech'].str.replace('wordsWithPeriodsRegex', '')

sentences = pd.concat([data['speech'].str.split('\. ', expand=True)])
df = pd.concat([data, sentences], axis=1). \
    drop('speech', axis=1)

df = pd.melt(df, id_vars=['branch', 'congressID', 'ideology_score', 'page', 'part', 'speaker', 'year'],
             value_name='sentence'). \
    sort_values(by=['year', 'part', 'page']). \
    reset_index(). \
    drop(['variable', 'index'], axis=1)

df['sentence_raw'] = df['sentence']

df['sentence'] = df['sentence'].str.replace('VerDate.*', '')

df['sent_length'] = df['sentence'].str.len()
df['caps_length'] = df['sentence'].str.findall('[A-Z]').str.len()
df['letters_length'] = df['sentence'].str.findall('\w').str.len()
df['numbers_length'] = df['sentence'].str.findall('\d').str.len()
df['caps_prop'] = df['caps_length'] / df['letters_length']
df['num_prop'] = df['numbers_length'] / df['sent_length']
df = df.query('sent_length > 8 & caps_prop < .4 & num_prop < .5'). \
    drop(['sent_length', 'caps_length', 'letters_length', 'caps_prop', 'numbers_length', 'num_prop'], axis=1)

# df['period'] = pd.cut(df['year'], [1973, 1988, 2003, np.inf], labels=[1, 2, 3])

df['sentence_unclean'] = df['sentence']
df['sentence'] = df['sentence'].apply(cleanSentence)
df = df[df['sentence'] != '']

train, test = np.split(df.sample(frac=1), [int(.8*len(df))])

train.to_csv('../../data/csv/model/train.csv', index=False)
test.to_csv('../../data/csv/model/test.csv', index=False)
