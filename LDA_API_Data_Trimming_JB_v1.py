
# %%[markdown]

# This is a python script used to clean data obtained from the Senate Lobbying Disclosures Act API, removing unnecessary fields to save space and reformatting for ease of use.

#%%

# Imported for call handling
import requests

# Imported for dump 
import json


# %%

# Asks user to specify local drive filepath for unzipped json file since it was too large to process through the repository itself
filepath = input("Specify unzipped filepath: ")

# Load json data from filepath
with open(filepath, "r") as source:
    data = json.load(source)

data

# %%

# Specify a list of unnecessary fields to remove later
removeList = [
    'url', 
    'filing_uuid', 
    'filing_type', 
    'filing_period_display', 
    'filing_document_url',
    'filing_document_content_type',
    'expenses_method',
    'posted_by_name',
    'termination_date',
    'registrant_address_1',
    'registrant_address_2',
    'registrant_different_address',
    'registrant_city',
    'registrant_state',
    'registrant_zip',
    'conviction_disclosures',
    'foreign_entities'
]

# Define a check for display text to see if the item is actually a quarterly filings report (or an amendment to one)
def typeCheck(display_text):
    if 'Quarter' in display_text and ('Report' in display_text or 'Amendment' in display_text) and 'No Activity' not in display_text:
        return True
    else:
        return False

# Define certain keys in dictionaries accessed through a filing's "top-level" keys for use in filtering

registrantKeys = [
    'id',
    'name',
    'description',
]

clientKeys = [
    'id',
    'name',
    'general_description',
]

# Define a dictionary connecting the specified nested key listings with the "top-level" key of choice
trimChoices = {
    'registrant' : registrantKeys,
    'client' : clientKeys,
}


# Define a function specifically trimmming lobbying activity lists and making other alterations   
def trimLobbyingList(activityList):

    # Define an empty list that will be filled with new, pared-down lobbying activity dictionaries
    newList = []

    # For each activity in the original lobbying activities list:
    for activity in activityList:

        # Create an empty dictionary to fill out with desired fields
        newDict = {}

        # Add the issue code display field from the old dictionary
        newDict['general_issue_code_display'] = activity['general_issue_code_display']

        # Add a count of all listed lobbyists for the activity
        newDict['lobbyist_count_all'] = len(activity['lobbyists'])

        # Initialize a count for new lobbyists only; then, loop through the old lobbyist list to increment the count each time a new lobbyist is found
        newDict['lobbyist_count_new'] = 0
        for item in activity['lobbyists']:
            if item['new'] == True:
                newDict['lobbyist_count_new'] += 1
            else:
                pass
        # Initialize a list of government entities, and append the 'id' fields for each government entity from the old listed dictionaries
        newDict['government_entities'] = []
        for entity in activity['government_entities']:
            newDict['government_entities'].append(entity['id'])

        # Append the newly constructed dictionary (with only desired fields) to the new activity list
        newList.append(newDict)
    
    # Return the new activity list containing a stripped-down set of fields relevant for analysis
    return newList

# Define a function that "trims" each dictionary (not meant in the technical sense, more trimming down fields)
def trimDict(oldDict):

    # Initialize a new dictionary that will be used to accept the trimmed fields and ultimately replace the old dictionary
    newDict = {}

    # For each "top-level" dictionary key, present as a key in the 'trimChoices' dictionary defined earlier:
    for key1 in trimChoices:

        # Initialize a new, empty dictionary within the new dictionary for that key to point to
        newDict[key1] = {}

        # For each "inner-level" dictionary key, which the 'trimChoices' dictionary points to:
        for key2 in trimChoices[key1]:

            # Add the inner-level key within the newly defined top-level key's dictionary
            newDict[key1][key2] = oldDict[key1][key2]
       
    # Separately from this process, define a new key/value pair for lobbying activities that leverages the 'trimLobbyingList' defined above to strip out unnecessary information from the old dictionary's activity list.
    newDict['lobbying_activities'] = trimLobbyingList(oldDict['lobbying_activities'])

    return newDict

# Define an empty list to contain the cleaned/trimmed data
clean_data = []

# Define a count for pages
count = 0

# For each page in the initial data (25 filing results):
for page in data:

    # Increment the count
    count += 1

    # For each dictionary in the page (an individual filing):
    for dict in page:

        # If this is actually a quarterly filing, proceed with changes
        if typeCheck(dict['filing_type_display']) == True:

            # Delete keys that were specified earlier in the list for removal
            for keyR in removeList:
                del dict[keyR]

            # Apply the 'trimDict' function to the dictionary
            dict = trimDict(dict)

            # Append the altered dictionary to the 'clean_data' list
            clean_data.append(dict)
        
        # If this is NOT a quarterly filing (usually some sort of registration or a "No Activity" filing), don't act on it
        else:
            pass
    print(f"Cleaned page: {count}")

# Specify a file name to save the cleaned data
filename = 'cleaned_LDA_filings.json'

# Save the cleaned data to this file
with open(filename, "a") as source:
    json.dump(clean_data, source, indent = 6)
# %%
