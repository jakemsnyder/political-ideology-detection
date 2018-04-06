# import pdfminer3k as pdfminer
import json
import logging
import os
from time import time

from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTTextContainer, LTLine, LTItem
from pip._vendor import requests

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


def getCongressionalRecordPathWithPage(year, partNumber, pageNumber):
    # Missing year, part number, year, part number, page number
    pathTemplate = "https://www.govinfo.gov/content/pkg/CRECB-{:d}-pt{:d}/pdf/CRECB-{:d}-pt{:d}-Pg{:d}.pdf"

    return pathTemplate.format(year, partNumber, year, partNumber, pageNumber)


def getCongressionalRecordURL(year, partNumber):
    # Missing year, part number, year, part number, page number
    pathTemplate = "https://www.govinfo.gov/content/pkg/GPO-CRECB-{:d}-pt{:d}/pdf/GPO-CRECB-{:d}-pt{:d}.pdf"
    # pathTemplate = "https://www.govinfo.gov/content/pkg/CRECB-2001-pt1/pdf/CRECB-2001-pt1.pdf"
    # pathTemplate = "https://www.govinfo.gov/content/pkg/GPO-CRECB-1974-pt1/pdf/GPO-CRECB-1974-pt1-1.pdf"
    # pathTemplate = "https://www.govinfo.gov/content/pkg/GPO-CRECB-1974-pt1/pdf/GPO-CRECB-1974-pt1.pdf"
    # pathTemplate = "https://www.govinfo.gov/content/pkg/GPO-CRECB-1974-pt1/pdf/GPO-CRECB-1974-pt1.pdf"


    return pathTemplate.format(year, partNumber, year, partNumber)


def fileExists(fileSavePath):
    return os.path.isfile(fileSavePath)


def downloadPDFFile(fileURL):
    print('')
    print('About to download file from the following URL: {:s}'.format(fileURL))

    fileSaveName = 'year{:d}-part{:d}'.format(year, partNumber)
    fileSavePath = '../../data/pdf/{:s}.pdf'.format(fileSaveName)

    if not fileExists(fileSavePath):
        startTime = time()
        response = requests.get(fileURL)
        endTime = time()
        print('Downloading file took {:f} minutes'.format((endTime - startTime) / 60))

        if response.status_code == 404:
            raise Exception('Could not download file due to exception: {:s}'.format(response.reason))

        with open(fileSavePath, 'wb') as f:
            f.write(response.content)
        print('File saved to path: {:s}'.format(fileSavePath))
    else:
        print('File already exists')
    return fileSavePath


def readPDFFile(filePath):
    print('')
    print('About to parse URL at path: {:s}'.format(filePath))

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
        for page in document.get_pages():
            if pageNumber % 50 == 4:
                extracted_text[pageNumber] = ''
                pdfPageInterpreter.process_page(page)
                layout = pdfPageAggregator.get_result()
                for layoutObject in layout:
                    if isinstance(layoutObject, LTTextContainer):
                        text = layoutObject.get_text()
                        text = text.replace('\n', '')
                        extracted_text[pageNumber] += text
                    # elif isinstance(layoutObject, LTItem):
                    #     something = layoutObject.__getattribute__()

            pageNumber += 1

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

startTimeAcquireData = time()

rawData = {}

# template
# {
#     <year>: {
#         <part>: {
#             <page>:
#                 ..text..
#         }
#     }
# }

startingYear = 1974
endingYear = 1974  # 2018
for year in range(startingYear, endingYear + 1):

    rawData = {
        year: {}
    }

    startingPartNumber = 1
    # endingPartNumber = 2
    endingPartNumber = 34
    partNumberInterval = 5
    for partNumber in range(startingPartNumber, endingPartNumber, partNumberInterval):
        recordURL = getCongressionalRecordURL(year, partNumber)
        filePath = downloadPDFFile(recordURL)
        textPageDict = readPDFFile(filePath)
        rawData[year][partNumber] = textPageDict

    savePath = '../../data/json/raw/raw_json-{:d}.json'.format(year)
    saveDictAsJson(rawData, savePath)

endTimeAcquireData = time()
print('Total time to acquire data was {:f} minutes'.format((endTimeAcquireData - startTimeAcquireData) / 60))
print('Total pages collected: {:d}'.format(countPagesCollected(rawData)))

#
# path01_21_1974 = "https://www.govinfo.gov/content/pkg/GPO-CRECB-1974-pt1/pdf/GPO-CRECB-1974-pt1-1-2.pdf"
# path01_22_1974 = "https://www.govinfo.gov/content/pkg/GPO-CRECB-1974-pt1/pdf/GPO-CRECB-1974-pt1-2-2.pdf"
# path01_24_1974 = "https://www.govinfo.gov/content/pkg/GPO-CRECB-1974-pt1/pdf/GPO-CRECB-1974-pt1-4-1.pdf"
# path01_28_1974 = "https://www.govinfo.gov/content/pkg/GPO-CRECB-1974-pt1/pdf/GPO-CRECB-1974-pt1-5-1.pdf"
#
# sample = "https://www.govinfo.gov/content/pkg/CRECB-2001-pt1/pdf/CRECB-2001-pt1-Pg7.pdf"





# https://www.govinfo.gov/content/pkg/CRECB-2001-pt1/pdf/CRECB-2001-pt1-Pg7.pdf
# https://www.govinfo.gov/content/pkg/CRECB-1974-pt1/pdf/CRECB-1974-pt1-Pg7.pdf
