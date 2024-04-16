
# %%[markdown]

# This is a python script used to clean data obtained from the Senate Lobbying Disclosures Act API, removing unnecessary fields to save space and reformatting for ease of use.

#%%

'''
Imports
'''

# Imported for call handling
import requests

# Imported for dump 
import json


# %%

'''
Open and load raw json file with user-input filepath
'''

# Asks user to specify local drive filepath for unzipped json file since it was too large to process through the repository itself
filepath = input("Specify unzipped filepath: ")

# Load json data from filepath
with open(filepath, "r") as source:
    data = json.load(source)

data

# %%
data
# %%


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(data)

# Basic Information 
print(df.info())
# %%
# Display summary statistics for numerical variables
print(df.describe())

# %%
# Check for missing values
print(df.isnull().sum())

# %%
# Summary Statistics
summary_stats = df[['income', 'expenses']].describe()
print("Summary Statistics:")
print(summary_stats)

# %%
plt.figure(figsize=(12, 6))

# Distribution of Income and Expenses
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['income'], bins=20, kde=True)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['expenses'], bins=20, kde=True)
plt.title('Distribution of Expenses')
plt.xlabel('Expenses')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Create boxplot for expenses
sns.boxplot(data=df, y='expenses', color='tab:blue')
plt.title('Boxplot of Expenses')
plt.ylabel('Expenses')
plt.show()

# Create boxplot for income
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='income', color='tab:orange')
plt.title('Boxplot of Income')
plt.ylabel('Income')
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
# Group by index and count occurrences of income and expenses
income_count = df['income'].value_counts().sort_index()
expenses_count = df['expenses'].value_counts().sort_index()

plt.figure(figsize=(10, 6))

# Bar plot for income
plt.bar(income_count.index, income_count.values, color='tab:red', alpha=0.7, label='Income')

# Bar plot for expenses
plt.bar(expenses_count.index, expenses_count.values, color='tab:blue', alpha=0.7, label='Expenses')

plt.title('Frequency of Income and Expenses')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.legend()

plt.xticks(ticks=plt.xticks()[0][::len(income_count) // 10])

plt.tight_layout()
plt.show()



# %%


# %%

# Extract registrant name
df['registrant_name'] = df['registrant'].apply(lambda x: x['name'])
registrant_unique_names = df['registrant_name'].value_counts()
print("Unique registrant names:\n", registrant_unique_names)

# Extract registrant description 
df['registrant_description'] = df['registrant'].apply(lambda x: x['description'])
registrant_unique_descriptions = df['registrant_description'].value_counts()
print("\nUnique registrant descriptions:\n", registrant_unique_descriptions)

# %%
# Extracting client name
df['client_name'] = df['client'].apply(lambda x: x['name'])
unique_client_names = df['client_name'].value_counts()
print("Unique client names and their frequencies:\n", unique_client_names)

# Extracting client general_description
df['general_description'] = df['client'].apply(lambda x: x.get('general_description', None))
unique_descriptions = df['general_description'].value_counts()
print("\nUnique general descriptions:\n", unique_descriptions)

# %%
# Extracting general_issue_code_display from 'lobbying_activities' variable
all_general_issues = [issue for sublist in df['lobbying_activities'].apply(lambda x: [lobby['general_issue_code_display'] for lobby in x]) for issue in sublist]
unique_general_issues = pd.Series(all_general_issues).value_counts()
print("\nUnique general issue codes:\n", unique_general_issues)

# Extracting lobbyist_count_all from 'lobbying_activities' variable
df['lobbyist_count_all'] = df['lobbying_activities'].apply(lambda x: sum(lobby['lobbyist_count_all'] for lobby in x))
total_lobbyist_count = df['lobbyist_count_all'].sum()
print("\nTotal lobbyist count:", total_lobbyist_count)

# Extracting lobbyist_count_new from 'lobbying_activities' variable
df['lobbyist_count_new'] = df['lobbying_activities'].apply(lambda x: sum(lobby['lobbyist_count_new'] for lobby in x))
total_new_lobbyist_count = df['lobbyist_count_new'].sum()
print("Total new lobbyist count:", total_new_lobbyist_count)

# Extracting government_entities from 'lobbying_activities' variable
all_government_entities = [entity for sublist in df['lobbying_activities'].apply(lambda x: [e for lobby in x for e in lobby['government_entities']]) for entity in sublist]
unique_government_entities = pd.Series(all_government_entities).value_counts()
print("\nUnique government entities:\n", unique_government_entities)

# %%

# Top 10 Entities:
# Top 10 entities - Registrants
N=10
top_registrants = df['registrant_name'].value_counts().head(N)
print("Top", N, "registrants:")
print(top_registrants)

# Top 10 entities - Clients
top_clients = df['client_name'].value_counts().head(N)
print("\nTop", N, "clients:")
print(top_clients)
# %%

# %%
# Lobbying Activities Analysis:
# Most common general issue codes
common_issue_codes = df['lobbying_activities'].explode('general_issue_code_display').value_counts().head(N)
print("Most common general issue codes:")
print(common_issue_codes)

