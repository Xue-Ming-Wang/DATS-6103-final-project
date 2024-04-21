# %%[markdown]

# This file uses the cleaned LDA filing data as an input, iterating through records to create a dataframe with dummy variables for each category.


# %%
# Importing libraries

# Importing json to handle raw file in repository + API calls
import json

# Importing requests for API calls
import requests

# Importing itertools to combine nested lists into one list conveniently
import itertools

# Importing pandas for dataframe creation/transformation
import pandas as pd

# Importing matplotlib and seaborn for EDA data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Importing scikitlearn for modeling and feature selection work
import sklearn

# Importing numpy for modeling work
import numpy as np

# %%
# Load data from raw file in repository

with open("cleaned_LDA_filings_v2.json") as file:

    data = json.load(file)

# %%
# Make API calls for static data on issue names and entity codes/names

# Part 1: Issues; create a URL, make the call, and add each full name to issuesList

issuesURL = r'https://lda.senate.gov/api/v1/constants/filing/lobbyingactivityissues/'

issuesRequest = requests.get(issuesURL)

issuesJson = issuesRequest.json()

issuesList = []

for issueDict in issuesJson:
    issuesList.append(issueDict['name'])

# Part 2: Entities; create a URL, make the call, and add each full name + code to their own respective lists

entitiesURL = 'https://lda.senate.gov/api/v1/constants/filing/governmententities/'
entitiesRequest = requests.get(entitiesURL)

entitiesJson = entitiesRequest.json()

entitiesIdList = []
entitiesNameList = []

for entityDict in entitiesJson:
    entitiesIdList.append(entityDict['id'])
    entitiesNameList.append(entityDict['name'])

# %%
# Loop through nested records to generate a flattened list for our dataframe

# Initialize a dataList variable to append record-specific results to 
dataList = []

# For each record:
for record in data:
    # Define an empty list to fill out (will ultimately be appended to datalist)
    recordList = []

    # Append a field for income or expenses, depending on which one is available (the nature of filings means that one or the other will be present but not both)
    if record['income'] != None:
        recordList.append(float(record['income']))
    elif record['expenses'] != None:
        recordList.append(float(record['expenses']))
    else:
        recordList.append(0)

    # If the type was expenses, mark True for internal lobbying
    if record['expenses'] != None:
        recordList.append(1)
    else:
        recordList.append(0)

    # Initialize an empty list for government entities
    recordEntitiesList = []
    
    # Initialize an empty list for lobbyists
    recordLobbyistsList = []

    # For each issue that could show up:
    for issue in issuesList:
        # Create/switch issueCheck to False
        issueCheck = False

        # For each activity in the lobbying activities list:
        for activity in record['lobbying_activities']:
            # If this activity matches the issue in question:
            if activity['general_issue_code_display'] == issue:
                # Change issueCheck to True
                issueCheck = True

            # Add any entitity IDs found to our entities list
            recordEntitiesList.append(activity['government_entities'])

            # Add any lobbyists found to our lobbyists list
            recordLobbyistsList.append(activity['lobbyists'])

        # If the issue was found among the record's activities, append True to the recordList, otherwise append False
        if issueCheck == True:
            recordList.append(1)
        else:
            recordList.append(0)

    # Convert the recordEntitiesList, which is now filled with nested entitity lists from all activity disclosures, to a 1-D list
    recordEntitiesConcat = list(itertools.chain.from_iterable(recordEntitiesList))

    # Conver the recordLobbyistsList, which is now filled with nested lobbyist lists from all activity disclosures, to a 1-D list
    recordLobbyistsConcat = list(itertools.chain.from_iterable(recordLobbyistsList))

    # Initialize a list for unique entities
    recordEntitiesUnique = []

    # For each "raw" entity, check to see if it is already present in the unique list before including it
    for recordEntityRaw in recordEntitiesConcat:
        if recordEntityRaw not in recordEntitiesUnique:
            recordEntitiesUnique.append(recordEntityRaw)
    
    # For each entity ID in the list gathered earlier:
    for entityId in entitiesIdList:
        # Create/switch entityCheck to False
        entityCheck = False

        # For each unique entity ID in this record:
        for recordEntityId in recordEntitiesUnique:
            # If this ID matches the value present in the entity ID list:
            if entityId == recordEntityId:
                # Change entityCheck to True
                entityCheck = True
        
        # If the enitity was found among the record's activities, append True to the recordList, otherwise append False
        if entityCheck == True:
            recordList.append(1)
        else:
            recordList.append(0)
    
    # Initialize a list for unique lobbyists
    recordLobbyistsUnique = []

    # For each "raw" lobbyist, check to see if they are already present in the unique list before including them
    for recordLobbyistRaw in recordLobbyistsConcat:
        if recordLobbyistRaw not in recordLobbyistsUnique:
            recordLobbyistsUnique.append(recordLobbyistRaw)
    
    # Append a count for all unique lobbyists
    recordList.append(len(recordLobbyistsUnique))

    # Initialize a list for new lobbyists only
    recordLobbyistsNew = []
    # For each lobbyist in the unique lobbyist list:
    for recordLobbyistRaw in recordLobbyistsUnique:
        # If they are new, add them to the new lobbyist list
        if recordLobbyistRaw['new'] == True:
            recordLobbyistsNew.append(recordLobbyistRaw['new'])
    
    # Append a count for all new, unique lobbyists
    recordList.append(len(recordLobbyistsNew))


    # Add the full recordList to the dataList
    dataList.append(recordList)
# %%
# Define column names

# Initialize an empty list for column names
colNameList = []

# Append column name values in the order that they were added earlier
colNameList.append(['income or expenses', 'internal lobbying'])
colNameList.append(issuesList)
colNameList.append(entitiesNameList)
colNameList.append(['lobbyist_count_all', 'lobbyist_count_new'])

# Convert to a 1-D list
colNameList = list(itertools.chain.from_iterable(colNameList))

# %%
# Convert to dataframe

df = pd.DataFrame(dataList, columns = colNameList)

print(df)


# %%
# Drop filings that have no income/expenses

df = df[df['income or expenses'] != 0]

# %%
# Drop columns with very low incidence rates

for col in df.columns.tolist():
    zeroRows = len(df[df[col]==0])
    zeroPct = zeroRows/len(df)
    if zeroPct > 0.99:
        df.drop(columns=col, inplace = True)
        print(f"Dropping {col}; occurrence %: {1 - (zeroRows/len(df)):.2f}")

print(df)

# %%
# Drop internal lobbying due to focus on policy areas
df = df.drop(columns = ['internal lobbying'])

# %%

# Initializing list of column sums to identify useful targets for visuals
sumList = []

# Add dictionaries with column names and sums to the sumList
for col in df:
    if len(df[col].unique()) == 2:
        sumList.append({'name': col, 'sum': df[col].sum()})

# Sort in reverse order by the sum key
sumList.sort(reverse = True, key = lambda dict: dict['sum'])


# %%
# Include informative scatterplots

for item in sumList[0:5]:
    sns.scatterplot(data = df, x = 'income or expenses', y = 'lobbyist_count_all', hue = item['name'])
    plt.title(f"Filing amounts and total unique lobbyists for lobbyists that reported efforts relating to: {item['name']}")
    plt.show()
# %%
# Correlation plot

# Create correlation matrix
corrMatrix = df.corr()

# Plot a heatmap of all correlation values

plt.figure(figsize = (40,25))

sns.heatmap(corrMatrix, ax = None)

plt.title("Correlation matrix for all features")
plt.show()

# %%
# Define a list of variables for outcomes and split the dataset
yList = ['income or expenses', 'lobbyist_count_all', 'lobbyist_count_new']

 # %%
# Split the dataset into X and Y

Y = df.loc[:, yList]
X = df.drop(columns = yList)

print(Y)
print(X)

xCols = X.columns

# %%
# Gather feature importances

# Import mutual_info_regression; mutual_info_classif fails for continuous target variables
from sklearn.feature_selection import mutual_info_regression

# Calculate importances for all 3 potential target variables

importances_1 = mutual_info_regression(X, Y['income or expenses'])

importances_2 = mutual_info_regression(X, Y['lobbyist_count_all'])

importances_3 = mutual_info_regression(X, Y['lobbyist_count_new'])

feat_importances_1 = pd.Series(importances_1, xCols)

feat_importances_2 = pd.Series(importances_2, xCols)

feat_importances_3 = pd.Series(importances_3, xCols)

plt.figure(figsize = (40,25))

feat_importances_1.plot(kind = 'barh', color = 'purple')
feat_importances_1.plot(kind = 'barh', color = 'blue')
feat_importances_1.plot(kind = 'barh', color = 'green')

plt.show()

# %%
# Generate a list of average importances

# Initialize an empty list
avgImportances = {}

# For each column in the input variable dataframe:
for col in xCols.tolist():
    # Calculate the average of all importances
    avg = (feat_importances_1[col] + feat_importances_2[col] + feat_importances_2[col])/3
    # Add average importance to the dictionary, using the name as the key
    avgImportances[col] = avg

print(f"Average Importances: {avgImportances}")
     
# %% 
# Remove columns with no feature importance related to any potential output
for col in xCols.tolist():
    if avgImportances[col] == 0:
        X.drop(columns = [col], inplace = True)
        print(f"Dropping {col} due to zero feature importance")

# %%

# Mark highly correlated input variables based on which one has the highest average importance

# Initialize a list of column names for the dataframe to drop after checking correlations
dropList = []

# For each column in the dataframe:
for col1 in X.columns.tolist():
    # For each column in the dataframe (again):
    for col2 in X.columns.tolist():
        # Get the correlation between these from the matrix defined earlier
        corr = corrMatrix[col1][col2]

        # If the correlation is past 0.3, the columns aren't the same, and neither is in the exclusion list:
        if corr > 0.3 and col1 != col2:
            # Print the combination out
            print(f"High correlation: {col1} and {col2} ({corr})")

            # Add the column that has a lower sum in the dataset (less occurrences) to the drop list
            if avgImportances[col1] > avgImportances[col2]:
                print(f"Dropping {col2} due to lower feature importance")
                dropList.append(col2)
            else:
                print(f"Dropping {col1} due to power feature importance")
                dropList.append(col1)

# %%

# Drop columns that present multicollinearity concerns using the list generated in the previous loops
X.drop(columns = dropList, inplace = True)

# %%
# Plot elbow for k-means clustering

# Note: Lecture 10 notes; using 'auto' for n_init since the best parameters for this high-dimensional dataset aren't immediately obvious

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 50):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 'auto', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 50), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# %%
# Print cluster differences and select an outcome
for i in range(1, len(wcss)):
    print(f"{i-1} to {i}: {wcss[i] - wcss[i-1]}")

print('The "elbow" seems to be vaguely present at 19')

nChosen = 19

# %%
# Construct k-means model

kmeans = KMeans(n_clusters = nChosen, init = 'k-means++', max_iter = 300, n_init = 'auto', random_state = 0)

y_kmeans = kmeans.fit_predict(X)


# %%
# Graph clusters relative to outcome variables to see if visible relationships exist

Y_kMeans = Y.copy()

Y_kMeans['cluster'] = y_kmeans.tolist()

pct99 = np.percentile(Y_kMeans['income or expenses'],99)

sns.scatterplot(data = Y_kMeans, x = 'income or expenses', y = 'lobbyist_count_all', hue = 'cluster')
plt.xlim(0, pct99)
plt.title(f"Filing amounts and total unique lobbyists by k-means cluster\n(Up to 99th percentile filing amounts)")
plt.show()

# %%
# Group and visualize outcomes by cluster

clusterGroup = Y_kMeans.groupby('cluster')

clusterAvg = clusterGroup.mean()

clusterAvg = clusterAvg.reset_index()

for y in yList:
    sns.barplot(data = clusterAvg, x = 'cluster', y = y, palette = 'pastel')
    plt.title(f"{y} by k-means cluster")
    plt.show()

# %%
# Visualize cluster 4 categories

X_kMeans = X.copy()

X_kMeans['cluster'] = y_kmeans.tolist()

for cNum in [4,11]:

    X_kMeansFilter = X_kMeans[X_kMeans['cluster'] == cNum].mean().sort_values(ascending = True)

    X_kMeansFilter = X_kMeansFilter[X_kMeansFilter > 0.5]

    X_kMeansFilter.drop('cluster', inplace = True)

    plt.figure(figsize = (10,7))
    X_kMeansFilter.plot(kind = 'barh', title = f'Top entity and topic focuses for cluster {cNum}')
    plt.xlim(0, 1)
    plt.show()
# %%
