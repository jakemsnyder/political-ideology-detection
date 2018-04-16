import csv
import json
import pandas as pd
import numpy as np
import re

from src.clean.politicians import findIdeologyScore


def getDictFromJsonPath(jsonPath):
    with open(jsonPath, 'r') as file:
        jsonString = file.read()
        return json.loads(jsonString)


def saveDataFramesAsCSV(dataframe, savePath):
    dataframe.to_csv(savePath)


def cleanText(text):
    badPhrasesRE = re.compile("\\u00b7|\\u2022|\(cid:173\)")
    text = re.sub(badPhrasesRE, "", text)
    return text


def removeAllButLetters(word):
    regex = re.compile('[^a-zA-Z]')
    return regex.sub('', word)

def removeAllButLettersAndApostrophes(word):
    regex = re.compile("[^a-zA-Z']")
    return regex.sub('', word)


def removeAllButLettersAndSpaces(word):
    regex = re.compile('[^a-zA-Z ]')
    return regex.sub('', word)


def getIdeologyFromSpeaker(speaker, year, chamber, congressID):
    lastName = None
    state = None
    gender = None
    firstName = None

    lastNameRE1 = re.compile("M[rs]{1,2}\.\s+[A-Z ']{2,}\.")
    lastNameRE2 = re.compile("M[rs]{1,2}\.\s+[A-Z]{2,}\s+of\s+[A-Za-z ]+\.")
    lastNameRE3 = re.compile("M[rs]{1,2}\.\s+[A-Z]{2,}\s+[A-Z]{2,}\s+of\s+[A-Za-z ]+\.")
    if lastNameRE1.match(speaker): # Mr. DISNEY. or Mr. WALT DISNEY.
        split = speaker.split(" ")

        lastName = split[len(split) - 1]
        lastName = removeAllButLetters(lastName)

        prefix = split[0]
        gender = getGenderFromPrefix(prefix)

        if len(split) > 2:
            firstName = split[1]
            firstName = removeAllButLetters(firstName)
    elif lastNameRE2.match(speaker): # Mr. DISNEY of Florida. or Mr. DISNEY of New York.
        split = speaker.split(" ")

        lastName = split[1]
        lastName = removeAllButLetters(lastName)

        indexOfOf = speaker.index('of')
        state = speaker[indexOfOf + 3:]
        state = removeAllButLettersAndSpaces(state)

        prefix = split[0]
        gender = getGenderFromPrefix(prefix)
    elif lastNameRE3.match(speaker): # Mr. WALT DISNEY of Florida. or Mr. WALT DISNEY of New York.
        split = speaker.split(" ")

        lastName = split[1]
        lastName = removeAllButLetters(lastName)

        indexOfOf = speaker.index('of')
        state = speaker[indexOfOf + 3:]
        state = removeAllButLettersAndSpaces(state)

        prefix = split[0]
        gender = getGenderFromPrefix(prefix)

    if lastName:
        ideologyScore = findIdeologyScore(
            lastName,
            state=state,
            year=year,
            gender=gender,
            firstName=firstName,
            chamber=chamber,
            congressID=congressID
        )
    else:
        ideologyScore = None
    return ideologyScore


def getGenderFromPrefix(prefix):
    if 's' in prefix:
        return 'F'
    else:
        return 'M'


speakerRE = re.compile(
    "M[rs]{1,2}\.\s+[A-Z]{2,}\s+[A-Z]{1}\.\s+[A-Z']{2,}\."
    "|M[rs]{1,2}\.\s+[A-Z']{2,}\."
    # "|M[rs]{1,2}\.\s+[A-Z]{2,}\s[A-Z']{2,}\."
    "|The\s+VICE\s+PRESIDENT\."
    "|The\s+PRESIDENT\."
    "|M[rs]{1,2}\.\s+SPEAKER\."
    "|The\s+PRESIDING\s+OFFICER\."
    "|The\s+SPEAKER\s+pro\s+tempore\."
    "|The\s+ACTING\s+PRESIDENT\s+pro\s+tempore\."
    "|The\s+ACTING\s+PRESIDENT\s+pro\s+tem\s+pore\."
    "|The\s+Acting\s+CHAIR\."
    "|M[rs]{1,2}\.\s+[A-Z ']{2,}\s+of\s+[A-Z][a-z]+\."
    "|M[rs]{1,2}\.\s+[A-Z ']{2,}\s+of\s+[A-Z][a-z]+\s+[A-Z][a-z]+\."
    # "|STATEMENT BY [ A-Z]{2,}\s"
)

startingYear = 1974
endingYear = 1974  # 2018


def extractChamberFromText(text):
    if 'HOUSE' in text[:50]:
        chamber = 'House'
    elif 'SENATE' in text[:50]:
        chamber = 'Senate'
    else:
        chamber = None
    return chamber


def extractSpeakersSpeechesFromText(text):
    speakers = re.findall(speakerRE, text)
    speeches = speakerRE.split(text)
    return speakers, speeches


def getCongressIDFromYear():
    return int(np.ceil(int(year) / 2 - 894))


for year in range(startingYear, endingYear + 1):

    speakerYearBranchSet = set()

    rawRecordsPath = '../../data/json/raw/raw_json-{:d}.json'.format(year)
    rawRecords = getDictFromJsonPath(rawRecordsPath)
    for year, partDict in rawRecords.items():
        found = 0
        total = 0
        ideologyDoneDict = {}

        yearlyResults = []
        for part, pageDict in partDict.items():
            for page, text in pageDict.items():

                text = cleanText(text)

                chamber = extractChamberFromText(text)
                speakers, speeches = extractSpeakersSpeechesFromText(text)
                congressID = getCongressIDFromYear()

                for i in range(len(speakers)):
                    speaker = speakers[i].strip()
                    speaker = re.sub(' +', ' ', speaker)
                    speech = speeches[i + 1]

                    if (speaker, year, chamber, congressID) in ideologyDoneDict:
                        ideologyScore = ideologyDoneDict[(speaker, year, chamber, congressID)]
                    else:
                        ideologyScore = getIdeologyFromSpeaker(speaker, year, chamber, congressID)
                        ideologyDoneDict[(speaker, year, chamber, congressID)] = ideologyScore
                        if ideologyScore:
                            found += 1
                            yearlyResults.append({
                                'year': year,
                                'part': part,
                                'page': page,
                                'speaker': speaker,
                                'speech': speech,
                                'branch': chamber,
                                'congressID': congressID,
                                'ideology_score': ideologyScore
                            })
                        total += 1
        print('')
        print('year: {:s}'.format(year))
        print('found {:d} out of {:d}'.format(found, total))
        print('{:f}%'.format(found / total))

        savePath = '../../data/csv/processed/record-{:s}.csv'.format(year)
        yearlyResultsDF = pd.DataFrame(yearlyResults)
        saveDataFramesAsCSV(yearlyResultsDF, savePath)
