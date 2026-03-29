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



for p in paragraphs:
    links = p.find_all('a')
    for link in links:
        href = link.get('href')  # ← rename to href
        
        if href and href.startswith('/wiki/') and ':' not in href and len(valid_urls) < 20:
            full_url = 'https://en.wikipedia.org' + href
            if full_url not in valid_urls:
             valid_urls.append(full_url)
            

# Let's test it! 
print(f"Found {len(valid_urls)} valid links!")
print("First 10 links:", list(valid_urls)[:10])
# after finding main_content
print("main_content found:", main_content is not None)

# after finding paragraphs
print("paragraphs found:", len(paragraphs))