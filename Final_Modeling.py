# %%[markdown]
## Final Project: Exploring Lobbying Data

### By: Jehan Bugli, Jupiter Kang, and Vivian Wang

# %%[markdown]

# ***
## Initial Imports

# %%
# Initial Library Imports

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

# Importing xgboost for modeling work
import xgboost as xgb

# Importing statistics for easy mode calc w/ XGB CV grid searches
import statistics

# %%
# Importing "Child" Libraries

# Importing functionality for k-means modeling
from sklearn.cluster import KMeans

# Importing model_selection
from sklearn import model_selection

# Importing CV grid search
from sklearn.model_selection import GridSearchCV

# Importing SVM regressor (unused, SVM skipped)
from sklearn.svm import SVR

# Importing preprocessing (unused, SVM skipped)
from sklearn import preprocessing

# Importing gradient boosting library
from sklearn.ensemble import GradientBoostingRegressor

# Importing libraries for Apriori algorithm and association rules
from mlxtend.frequent_patterns import apriori, association_rules

# Importing libraries for data preprocessing and scaling
from sklearn.preprocessing import StandardScaler

# Importing libraries for K-means clustering
from sklearn.cluster import KMeans

# Importing libraries for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Importing libraries for Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# Importing libraries for calculating accuracy and generating classification reports
from sklearn.metrics import accuracy_score, classification_report

# Importing libraries for cross-validation
from sklearn.model_selection import cross_val_score

# Importing libraries for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing libraries for decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Importing library for creating a pipeline for data preprocessing and model training
from sklearn.pipeline import make_pipeline

# Import mutual_info_regression; mutual_info_classif fails for continuous target variables
from sklearn.feature_selection import mutual_info_regression

# %%[markdown]

# ***
## Loading Repository & Online Data

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


# %%[markdown]

# ***
## Dataframe Generation

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

df.describe()

# %%[markdown]

# ***
## Question 2
### What is the relationship between lobbying focus (in total contributions, total lobbyists hired, etc.) and activity/entity focus?

# %%[markdown]

### Initial Filtering

# %%
# Drop filings that have no income/expenses

df = df[df['income or expenses'] != 0]

# %%
# Define a list of variables for outcomes to use throughout the script

yList = ['income or expenses', 'lobbyist_count_all', 'lobbyist_count_new']


# %%
# Drop columns with very low  or near-ubiquitous incidence rates

for col in df.columns.tolist():
    zeroRows = len(df[df[col]==0])
    zeroPct = zeroRows/len(df)
    if (zeroPct > 0.99 or zeroPct < 0.01) and col not in yList:
        df.drop(columns=col, inplace = True)
        print(f"Dropping {col}; occurrence %: {1 - (zeroRows/len(df)):.2f}")
    else:
        print(f"Keeping {col}; occurrence %: {1 - (zeroRows/len(df)):.2f}")


print(df)

# %%
# Drop internal lobbying due to focus on policy areas
df = df.drop(columns = ['internal lobbying'])


# %%[markdown]

### Exploration and Correlations

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
# Include scatterplots of most common features

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

plt.title("Correlation matrix for all features", fontsize = 40)
plt.show()

 # %%
# Split the dataset into X and Y

Y = df.loc[:, yList]
X = df.drop(columns = yList)

print(Y)
print(X)

xCols = X.columns


# %%
# Calculate importances for all 3 potential target variables

importances_1 = mutual_info_regression(X, Y['income or expenses'])

importances_2 = mutual_info_regression(X, Y['lobbyist_count_all'])

importances_3 = mutual_info_regression(X, Y['lobbyist_count_new'])

# Place importances into respective series for plotting

feat_importances_1 = pd.Series(importances_1, xCols)

feat_importances_2 = pd.Series(importances_2, xCols)

feat_importances_3 = pd.Series(importances_3, xCols)

# Plot importances (not super legible but just for reference)

plt.figure(figsize = (40,25))

feat_importances_1.plot(kind = 'barh', color = 'purple')
feat_importances_2.plot(kind = 'barh', color = 'blue')
feat_importances_3.plot(kind = 'barh', color = 'green')

plt.title("Purple = 'Income or expenses', Blue = all lobbyists, Green - new lobbyists", pad = 50, fontsize = 30)
plt.suptitle("Feature importances for inputs given target variables", fontsize = 40)

plt.show()

# %%
# Generate a list of average importances

# Initialize an empty list
avgImportances = {}

# For each column in the input variable dataframe:
for col in xCols.tolist():
    # Calculate the average of all importances
    avg = (feat_importances_1[col] + feat_importances_2[col] + feat_importances_3[col])/3
    # Add average importance to the dictionary, using the name as the key
    avgImportances[col] = avg

print(f"Average Importances: {avgImportances}")
     
# %% 
# Remove columns with no feature importance related to any potential output

# For each input column:
for col in xCols.tolist():
    # If the average importance is zero:
    if avgImportances[col] == 0:
        # Drop it
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

# %%[markdown]

### K-Means Modeling

# %%
# K-means clustering
#

# Plot elbow for k-means clustering

# Note: Lecture 10 notes; using 'auto' for n_init since the best parameters for this high-dimensional dataset aren't immediately obvious

wcss = []
for i in range(1, 50):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)
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


print("There is no clear drop-off in utility from added clusters")


# %%
# Skipped code for k-means clustering contingent upon a reasonable optimal cluster number

'''

import numpy as np

# Setting chosen cluster number as well as a random seed so that results stay consistent across runs
nChosen = 30

np.random.seed(77)

# Construct k-means model

kmeans = KMeans(n_clusters = nChosen, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(X)


# Graph clusters relative to outcome variables to see if visible relationships exist

Y_kMeans = Y.copy()

Y_kMeans['cluster'] = y_kmeans.tolist()

pct99 = np.percentile(Y_kMeans['income or expenses'],99)

sns.scatterplot(data = Y_kMeans, x = 'income or expenses', y = 'lobbyist_count_all', hue = 'cluster')
plt.xlim(0, pct99)
plt.title(f"Filing amounts and total unique lobbyists by k-means cluster\n(Up to 99th percentile filing amounts)")
plt.show()

# Group and visualize outcomes by cluster

clusterGroup = Y_kMeans.groupby('cluster')

clusterAvg = clusterGroup.mean()

clusterAvg = clusterAvg.reset_index()

for y in yList:
    sns.barplot(data = clusterAvg, x = 'cluster', y = y, palette = 'pastel')
    plt.title(f"{y} by k-means cluster")
    plt.show()

# # %%
# Visualize cluster categories

X_kMeans = X.copy()

# Add a cluster number field to the copied input dataframe
X_kMeans['cluster'] = y_kmeans.tolist()

# For each cluster:
for cNum in range(0, nChosen):

    # Filter to the given cluster alone and calculate means for all input fields; sort in ascending order
    X_kMeansFilter = X_kMeans[X_kMeans['cluster'] == cNum].mean().sort_values(ascending = True)

    # Keep fields where the mean is above 0.5 (therefore the presence rate is above 50%)
    X_kMeansFilter = X_kMeansFilter[X_kMeansFilter > 0.5]

    # X_kMeansFilter.drop('cluster', inplace = True)

    plt.figure(figsize = (10,7))
    X_kMeansFilter.plot(kind = 'barh', title = f'Top entity and topic focuses for cluster {cNum}')
    plt.xlim(0, 1)
    plt.show()

'''

# %%[markdown]

### Support Vector Machine Modeling

# %%
# Create train-test split

X_train, X_test, y_train, y_test = train_test_split(X, Y['income or expenses'], test_size=0.2)


# %%
# SVM cv grid search and modeling (skipped because the cv portion was taking far too long)

'''

grid = {
   'C':[0.01,0.1,1,10],
   'kernel' : ["linear","poly","rbf","sigmoid"],
   'degree' : [1,3,5,7],
   'gamma' : [0.01,1]
}
# # %%
# Run grid searches for the best SVM parameter combo (from Lecture 12 notes), skipped
svr  = SVR ()
svm_cv = model_selection.GridSearchCV(svr, grid, cv = 5)
svm_cv.fit(X_train,y_train)
print("Best Parameters:",svm_cv.best_params_)
print("Train Score:",svm_cv.best_score_)
print("Test Score:",svm_cv.score(X_test,y_test))

'''

# %%[markdown]

### XGB Modeling

# %%
# Generate a list of learning rates

# Initialize an empty list
rateList2 = []

# For numbers in the given range:
for num in range(4,10):
    # Add that * 0.01, creating the desired list
    rateList2.append(num*0.01)


# %%
# Generate a list of max depths

depthList2 = range(2,5)

# %%
# Generate a dictionary of parameters for use in CV grid search

params = {
    'learning_rate': rateList2,
    'max_depth': depthList2,
    # Note: manually tested estimator ranges and this seemed to be centered around the usual optimums
    'n_estimators': range(14,20)
}


# %%
# Generating the test model for CV grid search

testModel = xgb.XGBRegressor(random_state = 1)

# %%
# Looping through fold amounts to perform CV grid search and identify the best modeling parameters

# Initializing empty lists
cvLR = []
cvDepth = []
cvEst = []

# For the given cv fold numbers:
for cvNum in [5, 7, 10]:
    # Perform a grid search
    gridSearchTest = GridSearchCV(
    testModel,
    params,
    cv = cvNum
    )
    # Fit the search
    gridSearchTest.fit(X_train,y_train)
    # Print the best parameters and append them to their respective lists
    print(f"Best parameters for cv {cvNum}: {gridSearchTest.best_params_}")
    cvLR.append(gridSearchTest.best_params_['learning_rate'])
    cvDepth.append(gridSearchTest.best_params_['max_depth'])
    cvEst.append(gridSearchTest.best_params_['n_estimators'])



# %%
# Calculate mode values across all cv fold numbers

modeLR = statistics.mode(cvLR)

modeDepth = statistics.mode(cvDepth)

modeEst = statistics.mode(cvEst)

print(f"Mode learning rate: {modeLR}")
print(f"Mode maximum depth: {modeDepth}")
print(f"Mode minimum child weight: {modeEst}")
# %%
# Test optimized model with mode inputs, different outputs, and test/training splits

# Define optimized model
model=xgb.XGBRegressor(random_state = 1, learning_rate = modeLR, max_depth = modeDepth, n_estimators = modeEst)

# Initialize empty list of test sizes
sizeList = []

# Add test sizes from 0.1 to 0.9
for num in range(1,10,1):
    sizeList.append(num/10)

print(f"Test sizes in use: {sizeList}")

# Initialize empty list of scores
scoreList = []

# For each dependent variable:
for y in yList:
    # For each listed test size:
    for size in sizeList:
        # Redefine the testing/training split
        X_train, X_test, y_train, y_test = train_test_split(X, Y[y], test_size = size)
        # Fit the new model
        model.fit(X_train, y_train)
        # Score the new model
        score = model.score(X_test,y_test)
        # Print the scores and append them to the score list
        print(f"Score for {y} at test size {size}: {score}")
        scoreList.append([y, size, score])


# %%
# Define a dataframe of scores by parameter

yScores = pd.DataFrame(scoreList, columns = ['dependentVar', 'size', 'score'])

print(yScores)

# %%
# Plot scores by test size to determine the optimal setup

plt.figure(figsize = (10,7))
sns.lineplot(yScores, x = 'size', y = 'score', hue = 'dependentVar')
plt.title("Scoring distribution by test split size and dependent variable")
plt.show()
# %%[markdown]

# ***
## Question 3
### For a given policy area, what is the likelihood that lobbying efforts would be conducted jointly with other policy areas? In other words, which policy areas have the closest associations?


# %%
# Resetting dataframe

df = pd.DataFrame(dataList, columns = colNameList)

# %%[markdown]
### Setting up variables
# %%
# Dropping two columns
df = df.drop(columns=['income or expenses', 'internal lobbying'])

# Policy areas
policy_areas= df.drop(columns=['lobbyist_count_new', 'lobbyist_count_all'])

# Lobby efforts
lobbyist= ['lobbyist_count_new', 'lobbyist_count_all']

# %%[markdown]
### EDA for Q3
# %%[markdown]
### correlation matrix
# %%
# Calculate the correlation matrix
correlation_matrix = policy_areas.corr()

# Display the correlation matrix
print(correlation_matrix)


# %%[markdown]
### Top/lowest correlation matrix
# %%
## Sort the correlation matrix
sorted_corr = correlation_matrix.unstack().sort_values(ascending=False)

# Remove correlations of variables with themselves (which will be 1)
sorted_corr = sorted_corr[sorted_corr != 1]

# Print out the 10 highest correlations
print("Top 10 highest correlations:")
for i, (indices, correlation) in enumerate(sorted_corr.head(10).items(), 1):
    index1, index2 = indices
    print(f"{i}. {index1} - {index2}: {correlation}")

# Print out the 10 lowest correlations
filtered_corr = sorted_corr[sorted_corr > 0.001].tail(10)

# Print out the 10 lowest correlations
print("\nTop 10 lowest correlations:")
for i, (indices, correlation) in enumerate(filtered_corr.items(), 1):
    index1, index2 = indices
    print(f"{i}. {index1} - {index2}: {correlation}")

# %%
    
# Identify the closest associations using correlation matrix
correlation_matrix = policy_areas.corr()

# Sort the correlations
sorted_corr = correlation_matrix.unstack().sort_values(ascending=False)

# Remove correlations of variables with themselves (which will be 1)
sorted_corr = sorted_corr[sorted_corr != 1]

# Display the top correlated pairs
print("Top correlated policy areas:")
print(sorted_corr.head(10))


    
# %%[markdown]
### correlation map
# %%
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Policy Areas')
plt.show()


# %%[markdown]
## Modeling
### Apriori algorithm 

# %%
# Convert policy_areas to binary values (1 if lobbying effort in that policy area, 0 otherwise)
policy_areas_binary = policy_areas.applymap(lambda x: 1 if x > 0 else 0)

# Find frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(policy_areas_binary, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filter out rules where the antecedent and consequent are complementary pairs of each other
filtered_rules = []

for index, rule in rules.iterrows():
    antecedents = set(rule['antecedents'])
    consequents = set(rule['consequents'])
    
    if (antecedents, consequents) not in filtered_rules and (consequents, antecedents) not in filtered_rules:
        filtered_rules.append((antecedents, consequents))

# Sort rules based on lift values
sorted_rules = rules.sort_values(by='lift', ascending=False)

# Print the top three rules after filtering
print("Top Three Filtered Association Rules:")
count = 0
for antecedents, consequents in filtered_rules:
    rule = sorted_rules[(sorted_rules['antecedents'] == antecedents) & (sorted_rules['consequents'] == consequents)].iloc[0]
    count += 1
    print(f"Rule {count}:")
    print(f"Antecedents: {antecedents}, Consequents: {consequents}")
    print(f"Support: {rule['support'] * 100:.2f}%, Confidence: {rule['confidence'] * 100:.2f}%, Lift: {rule['lift']}")
    if count == 3:
        break


# %%
# Display all resulting association rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)

# %%
#  K-mean Clustering - elbow method)

'''
# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(policy_areas)

# Determine optimal number of clusters using the elbow method
wcss = []
for i in range(1, 20):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 20), wcss)  
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means clustering with optimal number of clusters
optimal_clusters = 17  
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Analyze results
# Assign cluster labels back to DataFrame
policy_areas['Cluster'] = clusters

# Calculate mean lobbying efforts for each cluster
cluster_means = policy_areas.groupby('Cluster').mean()

# Visualize cluster means
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu")
plt.title('Mean Lobbying Efforts for Each Cluster')
plt.xlabel('Policy Areas')
plt.ylabel('Cluster')
plt.xticks(rotation=45, ha='right')
plt.show()

'''


# %%[markdown]
### Train a classification model -Random Forest 
# %%
X = policy_areas.values  # Input features

# Perform K-means clustering with 17 clusters
# kmeans = KMeans(n_clusters=17, random_state=42)
# cluster_assignments = kmeans.fit_predict(X)

y = df['lobbyist_count_new']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Interpret feature importances
feature_importances = rf_classifier.feature_importances_
feature_importance_dict = dict(zip(policy_areas.columns, feature_importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print only the top 20 feature importances
print("\nTop 20 Feature Importances:")
for feature, importance in sorted_feature_importance[:20]:
    print(f"{feature}: {importance}")

# %%
# Extract feature importances from the trained Random Forest classifier
rf_feature_importances = rf_classifier.feature_importances_
rf_feature_importance_dict = dict(zip(policy_areas.columns, rf_feature_importances))
sorted_rf_feature_importance = sorted(rf_feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Plotting the top 20 features for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(range(10), [val[1] for val in sorted_rf_feature_importance[:10]], align='center')
plt.yticks(range(10), [val[0] for val in sorted_rf_feature_importance[:10]])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.show()

# %%

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5) 

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Plot the cross-validation scores
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores - Random Forest Classifier')
plt.xticks(range(1, len(cv_scores) + 1))
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label='Mean CV Score')
plt.legend()
plt.show()

# %%
#  Train a classification model - logistic regression

'''
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression classifier with a higher max_iter value
logistic_reg = LogisticRegression(max_iter=1000) 

# Train the classifier on the scaled training data
logistic_reg.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = logistic_reg.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Fit the logistic regression model on the scaled training data
logistic_reg.fit(X_train_scaled, y_train)

# Extract coefficients from the logistic regression model
coefficients = logistic_reg.coef_[0]

# Create a dictionary mapping feature names to their coefficients
coefficients_dict = dict(zip(policy_areas.columns, coefficients))

# Sort the coefficients in descending order of absolute magnitude
sorted_coefficients = sorted(coefficients_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the top 20 feature coefficients
print("\nTop 20 Feature Coefficients:")
for feature, coefficient in sorted_coefficients[:20]:
    print(f"{feature}: {coefficient}")

# Extract top 20 feature names and coefficients
top_features = [val[0] for val in sorted_coefficients[:20]]
top_coefficients = [val[1] for val in sorted_coefficients[:20]]

# Plot the coefficients
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_coefficients, align='center')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Coefficient Value')
plt.title('Top 20 Feature Coefficients - Logistic Regression')
plt.gca().invert_yaxis() 
plt.show()

# Create a pipeline with scaling and logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)  

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Plot the cross-validation scores
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='lightgreen')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores - Logistic Regression')
plt.xticks(range(1, len(cv_scores) + 1))
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label='Mean CV Score')
plt.legend()
plt.show()
'''
# %%
# Train a classification model - decision tree

'''

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
decision_tree = DecisionTreeClassifier()

# Train the classifier on the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Extract feature importances from the trained Decision Tree classifier
dt_feature_importances = decision_tree.feature_importances_
dt_feature_importance_dict = dict(zip(policy_areas.columns, dt_feature_importances))
sorted_dt_feature_importance = sorted(dt_feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 6))
plt.barh(range(10), [val[1] for val in sorted_dt_feature_importance[:10]], align='center')
plt.yticks(range(10), [val[0] for val in sorted_dt_feature_importance[:10]])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances - Decision Tree')
plt.gca().invert_yaxis()
plt.show()



# Initialize the Decision Tree classifier
decision_tree = DecisionTreeClassifier()

# Perform cross-validation
cv_scores = cross_val_score(decision_tree, X, y, cv=5) 

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Plot the cross-validation scores
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='lightcoral')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores - Decision Tree Classifier')
plt.xticks(range(1, len(cv_scores) + 1))
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label='Mean CV Score')
plt.legend()
plt.show()

'''

# %%