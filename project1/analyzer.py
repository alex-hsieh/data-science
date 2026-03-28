import requests

# define the URL of the webpage to analyze
url = "https://en.wikipedia.org/wiki/Chicken"

# create a dictionary of headers to include in the request
headers = {
    # introduce myself to get access
    'User-Agent': 'FSU_Data_Science_Student_Project/1.0 (ach22h@fsu.edu)'
}


# make a GET request to the URL and store the response
response = requests.get(url, headers=headers)

print(response.text)
print("response status code:", response.status_code)
print(response.reason)