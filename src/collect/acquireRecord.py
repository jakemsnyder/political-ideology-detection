# import pdfminer3k as pdfminer
import json
import logging
import os
from datetime import datetime, timedelta
from time import time, sleep
from pandas.tseries.offsets import BDay
import random

random.seed(42)

from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfparser import PDFParser, PDFDocument, PDFSyntaxError
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTTextContainer, LTLine, LTItem
from pip._vendor import requests

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


def getCongressionalRecordURL(year, partNumber=None, date=None):
    if year >= 1999:
        month = str(date.month)
        day = str(date.day)
        if len(month) == 1:
            month = '0{:s}'.format(month)
        if len(day) == 1:
            day = '0{:s}'.format(day)

        pathTemplate = "https://www.gpo.gov/fdsys/pkg/CREC-{:d}-{:s}-{:s}/pdf/CREC-{:d}-{:s}-{:s}.pdf"
        return pathTemplate.format(year, month, day, year, month, day)
    else:
        pathTemplate = "https://www.govinfo.gov/content/pkg/GPO-CRECB-{:d}-pt{:d}/pdf/GPO-CRECB-{:d}-pt{:d}.pdf"
        return pathTemplate.format(year, partNumber, year, partNumber)


def fileExists(fileSavePath):
    return os.path.isfile(fileSavePath)


def downloadPDFFile(recordURL, year, partNumber=None, date=None):
    print('')
    print('About to download file from the following URL: {:s}'.format(recordURL))

    if partNumber:
        fileSaveName = 'year{:d}-part{:d}'.format(year, partNumber)
        fileSavePath = '../../data/pdf/{:d}/{:s}.pdf'.format(year, fileSaveName)
    else:
        fileSaveName = 'year{:d}-month{:d}-day{:d}'.format(year, date.month, date.day)
        fileSavePath = '../../data/pdf/{:d}/{:s}.pdf'.format(year, fileSaveName)

    if not fileExists(fileSavePath):
        startTime = time()
        response = requests.get(recordURL)
        endTime = time()
        print('Downloading file took {:f} minutes'.format((endTime - startTime) / 60))

        if response.status_code == 404:
            raise FileNotFoundError('Could not download file due to exception: {:s}'.format(response.reason))

        if response.status_code == 429:
            raise Exception('Could not download file due to exception: {:s}'.format(response.reason))

        with open(fileSavePath, 'wb') as f:
            f.write(response.content)
        print('File saved to path: {:s}'.format(fileSavePath))
    else:
        print('File already exists')
    return fileSavePath


def parsePDFFile(filePath, everyNPages=30):
    print('')
    print('About to parse file at path: {:s}'.format(filePath))

    with open(filePath, 'rb') as pdfFile:
        pdfParser = PDFParser(pdfFile)
        document = PDFDocument()
        pdfParser.set_document(document)
        document.set_parser(pdfParser)

        document.initialize('')
        pdfResourceManager = PDFResourceManager()
        laParams = LAParams()
        laParams.char_margin = 1.0
        laParams.word_margin = 1.0
        pdfPageAggregator = PDFPageAggregator(pdfResourceManager, laparams=laParams)
        pdfPageInterpreter = PDFPageInterpreter(pdfResourceManager, pdfPageAggregator)
        extracted_text = {}

        pageNumber = 1
        chosenStoppingPage = random.randint(1, everyNPages-1)
        try:
            for page in document.get_pages():
                if pageNumber % everyNPages == chosenStoppingPage:
                    extracted_text[pageNumber] = ''
                    pdfPageInterpreter.process_page(page)
                    layout = pdfPageAggregator.get_result()
                    for layoutObject in layout:
                        if isinstance(layoutObject, LTTextContainer):
                            text = layoutObject.get_text()
                            text = text.replace('-\n', '')
                            text = text.replace('\n', ' ')
                            extracted_text[pageNumber] += text
                pageNumber += 1
        except KeyError:
            pass

        print('URL parse complete')

        return extracted_text


def saveDictAsJson(rawData, savePath):
    with open(savePath, 'w') as outfile:
        rawJson = json.dumps(rawData, indent=4)
        outfile.write(rawJson)


def countPagesCollected(rawData):
    count = 0
    for year, partDict in rawData.items():
        for part, pageDict in partDict.items():
            count += len(pageDict)
    return count

def makeDirectory(year):
    path = '../../data/pdf/{:d}/'.format(year)
    if not os.path.exists(path):
        os.makedirs(path)

rawData = {}

# template
# {
#     <year>: {
#         <partORdate>: {
#             <page>:
#                 ..text..
#         }
#     }
# }

startingYear = 2011 # First year of data
endingYear = 2018 # Last year of data
for year in range(startingYear, endingYear + 1):

    startTimeAcquireData = time()

    rawData = {
        year: {}
    }

    if year >= 1999:
        makeDirectory(year)
        firstDateOfYear = datetime.strptime('01/01/{:d}'.format(year), '%m/%d/%Y')

        daysUntilStartingDay = random.randint(0, 5)
        recordDate = firstDateOfYear + BDay(daysUntilStartingDay)
        while recordDate.year == year:
            recordURL = getCongressionalRecordURL(year, date=recordDate)
            try:
                filePath = downloadPDFFile(recordURL, year, date=recordDate)
                textPageDict = parsePDFFile(filePath, everyNPages=7)

                recordDateString = str(recordDate)[:10]
                rawData[year][recordDateString] = textPageDict

                daysUntilNextRecordDate = random.randint(4, 9)
            except (PDFSyntaxError, FileNotFoundError):
                print('{:s} doesnt have data'.format(str(recordDate)))

                daysUntilNextRecordDate = random.randint(1,2)

            recordDate = recordDate + BDay(daysUntilNextRecordDate)
    else:
        startingPartNumber = random.randint(1, 3)
        endingPartNumber = 33

        partNumber = startingPartNumber
        while partNumber <= endingPartNumber:
            recordURL = getCongressionalRecordURL(year, partNumber=partNumber)
            filePath = downloadPDFFile(recordURL, year, partNumber=partNumber)
            textPageDict = parsePDFFile(filePath)

            rawData[year][partNumber] = textPageDict

            partNumber += random.randint(4, 8)

    savePath = '../../data/json/raw/raw_json-{:d}.json'.format(year)
    saveDictAsJson(rawData, savePath)

    endTimeAcquireData = time()
    print('Total time to acquire data for {:d} was {:f} minutes'.format(year, (endTimeAcquireData - startTimeAcquireData) / 60))
    print('Total pages collected: {:d}'.format(countPagesCollected(rawData)))
