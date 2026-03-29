import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from collections import Counter

'''
# NOTE: you may need to run these downloads once to get the necessary NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
'''
# create a dictionary of headers to include in the request
headers = {
    # introduce myself to get access
    'User-Agent': 'FSU_Data_Science_Student_Project/1.0 (ach22h@fsu.edu)'
}

# PART 1: WEB SCRAPING - meep!!!
# fetch a wikipedia page and return the cleaned main content div
def fetch_main_content(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # first find the content text container, then mw-parser-output inside that
    content_text = soup.find('div', id='mw-content-text')
    main_content = content_text.find('div', class_='mw-parser-output')

    # remove the infobox so its links don't get included
    infobox = main_content.find('table', class_='infobox')
    if infobox:
        infobox.decompose()

    # remove the hatnotes so their links don't get included
    for note in main_content.find_all('div', class_='hatnote'):
        note.decompose()

    return main_content


# get up to `limit` valid links that start with /wiki/ and don't contain a colon
def extract_wiki_links(main_content, limit=20):
    links = []
    for p in main_content.find_all('p'):
        for a in p.find_all('a'):
            href = a.get('href')
            # check if the href is valid and meets the criteria
            if href and href.startswith('/wiki/') and ':' not in href:
                full_url = 'https://en.wikipedia.org' + href
                # check if the full URL is not already in the list to avoid duplicates
                if full_url not in links:
                    links.append(full_url)
            if len(links) >= limit:
                return links
    return links


# define the URL of the webpage to analyze
seed_url = "https://en.wikipedia.org/wiki/Chicken"

# fetch and parse the seed page, then collect its valid links
seed_content = fetch_main_content(seed_url)
valid_urls = extract_wiki_links(seed_content)

# set up a list to hold the text of all documents
document = [seed_content.get_text()]

for url in valid_urls:
    content = fetch_main_content(url)
    document.append(content.get_text())

# TEST: print the valid URLs and the number of valid URLs found
print(f"Found {len(valid_urls)} valid links")
print("First 10 links:", valid_urls[:10])

# TEST: how many documents did we collect?
print(f"Total documents: {len(document)}")

# TEST: preview the first 200 characters of document 0 (Chicken page)
print("\nDocument 0 (Chicken) preview:")
print(document[0][:200])

# TEST: preview the first 200 characters of document 1 (first linked page)
print("\nDocument 1 preview:")
print(document[1][:200])


# PART 2: TEXT PREPROCESSING
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
cleaned_documents = []

for text in document:
    tokens = word_tokenize(text.lower())
    # keep only alpha tokens not in stop_words, then lemmatize each
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    cleaned_documents.append(' '.join(cleaned))

# TEST: preview the first 200 characters of the cleaned version of document 0
print(cleaned_documents[0][:200])


# PART 3: DATAFRAME CREATION
rows = []
for i in range(len(document)):
    rows.append(
        {
            'doc_id': i,
            'url': seed_url if i == 0 else valid_urls[i-1],
            'raw_text': document[i],
            'cleaned_text': cleaned_documents[i]
        }
    )

df = pd.DataFrame(rows)
df.to_csv('data.csv', index=False)

# TEST: print the first 5 rows of the DataFrame
print(df.head())
print(df.head(5))

# PART 4: WORD FREQUENCY ANALYSIS
top5_rows = []
for i in range(len(cleaned_documents)):
    tokens = cleaned_documents[i].split()
    top5 = Counter(tokens).most_common(5)
    # build a row starting with doc_id, then flatten the (word, freq) pairs
    row = [i]
    for word, freq in top5:
        row.append(word)
        row.append(freq)
    top5_rows.append(row)
 
pd.DataFrame(top5_rows).to_csv('top5Words.csv', index=False, header=False)