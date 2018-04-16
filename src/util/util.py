from pymongo import MongoClient


def connectToMongo(databaseName, collectionName):
    client = MongoClient()
    db = client[databaseName]
    collection = db[collectionName]
    return collection