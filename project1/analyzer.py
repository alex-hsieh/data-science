import requests
from bs4 import BeautifulSoup

# define the URL of the webpage to analyze
url = "https://en.wikipedia.org/wiki/Chicken"

# create a dictionary of headers to include in the request
headers = {
    # introduce myself to get access
    'User-Agent': 'FSU_Data_Science_Student_Project/1.0 (ach22h@fsu.edu)'
}


# make a GET request to the URL and store the response
response = requests.get(url, headers=headers)

# extract the content of the response and parse it with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# find the main content of the webpage by looking for the div with id 'bodyContent'
main_content = soup.find('div', id='bodyContent')

# Get a list of all paragraph tags
paragraphs = main_content.find_all('p')

# Set up an empty list to hold our final URLs
valid_urls = []

# Loop through every paragraph we found
for p in paragraphs:
    
    # Find all the anchor (link) tags inside THIS specific paragraph
    links = p.find_all('a')
    
    # Loop through those links to check them
    for link in links:
        url = link.get('href')