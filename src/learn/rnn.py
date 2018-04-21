from gensim.models.word2vec import Word2VecVocab
from keras import Sequential
from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing import sequence
from matplotlib import pyplot

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = '../../data/binary/word2Vec.bin'
TRAINING_DATA_PATH = '../../data/csv/model/train.csv'


def saveWord2VecModel(savePath, model):
    model.save(savePath)


def loadWord2VecModel(loadPath):
    return Word2Vec.load(loadPath)


def visualizeWordEmbeddings(wordEmbeddingMatrix):
    # Taken from https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
    pca = PCA(n_components=2)
    fittedPCA = pca.fit_transform(wordEmbeddingMatrix)
    pyplot.scatter(fittedPCA[:, 0], fittedPCA[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(fittedPCA[i, 0], fittedPCA[i, 1]))
    pyplot.show()


def encodeSentences(labelEncoder, sentences):
    sentencesEncoded = []
    for sentence in sentences:
        sentenceEncoded = labelEncoder.transform(sentence)
        sentencesEncoded.append(sentenceEncoded)
    return sentencesEncoded


def splitSentences(sentencesRaw):
    sentences = []
    for sentence in sentencesRaw:
        sentences.append(sentence.split(" "))
    return sentences


print('')
print('Loading Training Data')
data = pd.read_csv(TRAINING_DATA_PATH)
print('Training Data Loaded')

print('')
print('Processing Data')
X = data['sentence']
y = data['ideology_score'].astype('float')
sentencesTrainRaw, sentencesTestRaw, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=23)
sentencesTrain = splitSentences(sentencesTrainRaw)
sentencesTest = splitSentences(sentencesTestRaw)

model = Word2Vec(
    sentencesTrain,
    size=30,  # Size of representation vector for 1 word
    min_count=1,  # Minimum frequency for word too count
)
# print('')
# print('Saving Model')
# saveWord2VecModel(modelPath, model)
# print("Model Saved")

# print('')
# print('Loading Model')
# model = loadWord2VecModel(MODEL_PATH)
# print('Model Loaded')

words = list(model.wv.vocab)
numWords = len(words)

# wordEmbeddingMatrix = model[model.wv.vocab]
# visualizeWordEmbeddings(wordEmbeddingMatrix)

labelEncoder = LabelEncoder()
integerEncoding = labelEncoder.fit(words)

XTrain = encodeSentences(labelEncoder, sentencesTrain)
XTest = encodeSentences(labelEncoder, sentencesTest)

maxSentenceLength = 10
XTrain = sequence.pad_sequences(XTrain, maxlen=maxSentenceLength)
XTest = sequence.pad_sequences(XTest, maxlen=maxSentenceLength)

print('Data Processing Finished')

# load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# print(X_train)

embeddingVectorLength = 32
model = Sequential()
model.add(Embedding(numWords, embeddingVectorLength, input_length=maxSentenceLength))
model.add(LSTM(100))
model.add(Dense(1))
# model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
print('')
print(model.summary())
model.fit(XTrain, yTrain, validation_data=(XTest, yTest), epochs=3, batch_size=64, verbose=0)

score = model.evaluate(XTest, yTest, verbose=0)
print('')
print('score')
print(score)

predictionsTrain = model.predict(XTrain, verbose=0)
predictionsTest = model.predict(XTest, verbose=0)
meanSquaredErrorTrain = mean_squared_error(predictionsTrain, yTrain)
meanSquaredErrorTest = mean_squared_error(predictionsTest, yTest)

print('')
print('Train MSE: {:f}'.format(meanSquaredErrorTrain))
print('Test MSE: {:f}'.format(meanSquaredErrorTest))

print('')
print('Test Predictions')
print(predictionsTest)

print('')
print('Test Actual')
print(yTest)


