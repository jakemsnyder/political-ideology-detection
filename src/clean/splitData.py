import pandas as pd
import numpy as np
from nltk import PorterStemmer

np.random.seed(23)

train = pd.read_csv('../../data/csv/model/train.csv', index_col=None)
test = pd.read_csv('../../data/csv/model/test.csv', index_col=None)

def makeTaxWords():
    porterStemmer = PorterStemmer()
    taxWords = [
        'tax',
        'tariff',
        'levy',
        'fine',
        'economy',
        'money',
    ]
    return list(map(porterStemmer.stem, taxWords))


def hasTaxWord(sentence):
    words = sentence.split(' ')
    for word in words:
        if word in taxWords:
            return True
    return False


taxWords = makeTaxWords()
trainTax = train[train.apply(lambda x: hasTaxWord(x['sentence']), axis=1)]
testTax = test[test.apply(lambda x: hasTaxWord(x['sentence']), axis=1)]

trainTax.to_csv('../../data/csv/model/trainTax.csv', index=False)
testTax.to_csv('../../data/csv/model/testTax.csv', index=False)

def makeDrugsWords():
    porterStemmer = PorterStemmer()
    drugsWords = [
        'marijuana',
        'pot',
        'dope',
        'drug',
        'opioid',
        'heroin',
        'cocaine',
        'cocaine',
    ]
    return list(map(porterStemmer.stem, drugsWords))


def hasDrugsWord(sentence):
    words = sentence.split(' ')
    for word in words:
        if word in drugsWords:
            return True
    return False


drugsWords = makeDrugsWords()
trainDrugs = train[train.apply(lambda x: hasDrugsWord(x['sentence']), axis=1)]
testDrugs = test[test.apply(lambda x: hasDrugsWord(x['sentence']), axis=1)]

trainDrugs.to_csv('../../data/csv/model/trainDrugs.csv', index=False)
testDrugs.to_csv('../../data/csv/model/testDrugs.csv', index=False)



trainModern = train[train['year'] > 2010]
testModern = test[test['year'] > 2010]

trainModern.to_csv('../../data/csv/model/trainModern.csv', index=False)
testModern.to_csv('../../data/csv/model/testModern.csv', index=False)



def makePoliticalWords():
    porterStemmer = PorterStemmer()
    politicalWords = [
        'political',
        'tariff',
        'levy',
        'fine',
        'economy',
        'money',
        'bipartisan',
        'big government',
        'bleeding heart',
        'checks and balances',
        'gerrymander',
        'left wing',
        'right wing',
        'liberal',
        'conservative',
        'witch hunt',
        'abortion',
        'gay',
        'homosexual',
        'marijuana',
        'teabagger',
        'ground zero',
        'elite',
        'climate change',
        'global warming',
        'job',
        'recession',
        'gun',
        'right',
        'amendment',
        'security',
        'army',
        'armed forces',
        'capitalism',
        'free market',
        'socialist',
        'education',
        'student',
        'college',
        'teacher',
        'class',
        'responsibility',
        'free',
        'greed',
        'immigration',
        'life',
        'deal',
        'patriot',
        'peace',
        'welfare',
        'hand out',
        'handout',
        'rich',
        'science',
        'poor',
        'sustainable',
        'tolerance',
        'victim',
        'god',
        'christian',
        'faith',
        'pray',
        'religion',
        'prison',
        'criminal',
    ]
    return list(map(porterStemmer.stem, politicalWords))


def hasPoliticalWord(sentence):
    words = sentence.split(' ')
    for word in words:
        if word in politicalWords:
            return True
    return False


politicalWords = makePoliticalWords()
trainPolitical = train[train.apply(lambda x: hasPoliticalWord(x['sentence']), axis=1)]
testPolitical = test[test.apply(lambda x: hasPoliticalWord(x['sentence']), axis=1)]

trainPolitical.to_csv('../../data/csv/model/trainPolitical.csv', index=False)
testPolitical.to_csv('../../data/csv/model/testPolitical.csv', index=False)