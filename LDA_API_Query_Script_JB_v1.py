
# %%[markdown]

# This is a python script used to query the Lobbying Disclosures Act API for filing data that we will use in our project. [See API documentation here](https://lda.senate.gov/api/redoc/v1/)


#%%

# Imported for call handling
import requests

# Imported for dump 
import json

# Imported for sleep function to avoid throttling
import time

# %%

# Set up page rounding logic for paginated calls
def pageRound(pages):
    if pages % 25 == 0:
        return int(pages/25)
    else:
        return int(round(pages/25)+1)

# Set up the structure to call the API for each provided URL
def callURL(url):
    
    # Define authorization header with API key to pass through for calls
    auth = {'Authorization': 'Token bcfdf1aa2c38d3f738a9276fd9c2bd51abfaed3f'}

    # Make the request
    response = requests.get(url, headers = auth)

    # Check to see if the response was successful
    head = requests.head(url)
    if response.status_code == 200:

        # If yes, note the URL and return 'results' (25 filings, excluding metadata)
        print("Obtained results from URL: ", url)
        return response.json()['results']
    
    else:

        # If no, print an error message and return a blank list
        print("Failed to fetch data. Status code: ", response.status_code)
        print("Failed URL: ", url)
        return []
    
# %%

# Define a class for the overall call with certain properties and the main paginated call function

class apiCall:
    '''
    A class representing properties of an API call for a specified year and quarter as well as a paginated call function
    '''
    def __init__(self, year, quarter):
        '''
        Define certain properties based on users' year and quarter inputs
        '''
        self.y = year
        self.q = quarter

        # Create the starting URL for use in calls
        self.url = "https://lda.senate.gov/api/v1/filings/?filing_period="+quarter+"&filing_year="+year

        # Create a filename for later use in saving output
        self.filename = year + "_" + quarter + ".json"
        self.pages = pageRound(requests.get(self.url).json()['count'])

    def pagCall(self):
        raw_data = []
        for page in range(1, self.pages + 1, 1):
            page_url = self.url + "&page=" + str(page)
            data = callURL(page_url)
            raw_data.append(data)
            time.sleep(1)
        return raw_data
    
    
# %%

call = apiCall("2023", "fourth_quarter")

data = call.pagCall()

# %%

with open(call.filename, "a") as source:
    json.dump(data, source, indent = 6)





