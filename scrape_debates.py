"""
author: Adarsh Chavakula
email: ac4244@columbia.edu

Script to scrape presidential debates from http://www.presidency.ucsb.edu
The script finds the URLs relevant to the debates, scrapes the speeches from
all of them and stores them.
"""

import bs4 as bs
import urllib.request
import json

sauce = urllib.request.urlopen("http://www.presidency.ucsb.edu/debates.php")
soup = bs.BeautifulSoup(sauce)

# Get list of debate dates
debate_dates_soup = soup.find_all('td', {"class":"docdate"})
debate_date_list = [d.text for d in debate_dates_soup if len(d.text) > 1]

# Get list of debate URLs and their descriptions
all_urls_soup = soup.find_all('a')
all_urls_list = [url.get("href") for url in all_urls_soup]
all_urls_desc = [url.text for url in all_urls_soup]

debate_urls_desc = all_urls_desc[45:-2] #first 45 links and last 2 not debates
debate_urls_list = all_urls_list[45:-2]

# Scrape the text from each of the above links and create dict
ind = 1
full_debates_dict = {}
for url, desc, date in zip(debate_urls_list,
                           debate_urls_desc,
                           debate_date_list):
    full_debates_dict[ind] = {}
    debate_sauce = urllib.request.urlopen(url)
    debate_soup = bs.BeautifulSoup(debate_sauce, "lxml")
    text_dump = str(debate_soup.find("span", {"class": "displaytext"}))
    full_debates_dict[ind]["date"] = date
    full_debates_dict[ind]["desc"] = desc
    full_debates_dict[ind]["url"] = url
    full_debates_dict[ind]["text"] = text_dump
    ind = ind+1

# Write to JSON file
with open("full_debates.json", "w") as f:
    json.dump(full_debates_dict, f, indent=4)
