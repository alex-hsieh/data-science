import requests
from bs4 import BeautifulSoup

# define the URL of the webpage to analyze
url = "https://en.wikipedia.org/wiki/Chicken"

# create a dictionary of headers to include in the request
headers = {
    # introduce myself to get access
    'User-Agent': 'FSU_Data_Science_Student_Project/1.0 (ach22h@fsu.edu)'
}

# Set up an empty list to hold our final URLs
valid_urls = []

# make a GET request to the URL and store the response
response = requests.get(url, headers=headers)

# extract the content of the response and parse it with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# first find the content text container
content_text = soup.find('div', id='mw-content-text')

# then find mw-parser-output INSIDE that
main_content = content_text.find('div', class_='mw-parser-output')

# remove the infobox so its links don't get included
infobox = main_content.find('table', class_='infobox')
if infobox:
    infobox.decompose()

# remove the hatnote so its links don't get included
hatnote = main_content.find_all('div', class_='hatnote')
if hatnote:
    for note in hatnote:
        note.decompose()

# Get a list of all paragraph tags
paragraphs = main_content.find_all('p')


# get the 20 valid links that start with /wiki/ and don't contain a colon
for p in paragraphs:
    links = p.find_all('a')
    for link in links:
        href = link.get('href')  
        
        # Check if the href is valid and meets the criteria
        if href and href.startswith('/wiki/') and ':' not in href and len(valid_urls) < 20:
            full_url = 'https://en.wikipedia.org' + href

        # Check if the full URL is not already in the list to avoid duplicates
        if full_url not in valid_urls:
            valid_urls.append(full_url)

document = []

document.append(main_content.get_text())

for url in valid_urls:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    content_text = soup.find('div', id='mw-content-text')
    main_content = content_text.find('div', class_='mw-parser-output')
    # first find the content text container
    content_text = soup.find('div', id='mw-content-text')

    # then find mw-parser-output INSIDE that
    main_content = content_text.find('div', class_='mw-parser-output')

    # remove the infobox so its links don't get included
    infobox = main_content.find('table', class_='infobox')
    if infobox:
        infobox.decompose()

    # remove the hatnote so its links don't get included
    hatnote = main_content.find_all('div', class_='hatnote')
    if hatnote:
        for note in hatnote:
            note.decompose()

    # Get a list of all paragraph tags
    paragraphs = main_content.find_all('p')
    text = main_content.get_text()
    document.append(text)

# print the valid URLs and the number of valid URLs found
print(f"Found {len(valid_urls)} valid links")
print("First 10 links:", list(valid_urls)[:10])
print("main_content found:", main_content is not None)

print("paragraphs found:", len(paragraphs))

# TEST: how many documents did we collect?
print(f"Total documents: {len(document)}")

# TEST: preview the first 200 characters of document 0 (Chicken page)
print("\nDocument 0 (Chicken) preview:")
print(document[0][:200])

# TEST: preview the first 200 characters of document 1 (first linked page)
print("\nDocument 1 preview:")
print(document[1][:200])