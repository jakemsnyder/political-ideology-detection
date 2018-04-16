import pandas as pd
import numpy as np
from pymongo import MongoClient

from src.util.util import connectToMongo

stateAbbreviationsMap = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "le",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NE",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY"
}


def findIdeologyScore(lastName, firstName=None, state=None, congressID=None, year=None, gender=None, chamber=None):
    mongoCollection = connectToMongo('politics', 'voteview_members')

    query = {
        "fname": {
            # '$eq': lastName.upper()
            '$regex': lastName.upper() + '[,A-Za-z ]+'
        }
    }
    if congressID:
        query['congress'] = {'$eq': int(congressID)}
    if chamber:
        query['chamber'] = {'$eq': chamber}
    if state:
        if state.lower() in stateAbbreviationsMap:
            stateAbbreviation = stateAbbreviationsMap[state.lower()]
            query['state_abbrev'] = {'$eq': stateAbbreviation}
            # else:
            #     print('COULD NOT FIND STATE: {:s}'.format(state))
    if year:
        year = int(year)
        query['born'] = {'$lt': year}
        query['$or'] = [
            {'died': {'$eq': None}},
            {'died': {'$gt': year}}
        ]
    if gender:
        if gender == 'M':
            pass
        else:
            pass

    results = mongoCollection.find(query)

    # Ensure results are all the same person
    foundID = None
    matchRepresentative = None
    for result in results:
        if foundID:
            if foundID != result['bioguide_id']:
                # print('multiple people for search')
                return None
        if not foundID and 'bioguide_id' in result:
            foundID = result['bioguide_id']
            matchRepresentative = result

    if matchRepresentative and 'nominate' in matchRepresentative and 'dim1' in matchRepresentative['nominate']:
        return matchRepresentative['nominate']['dim1']

    return None
