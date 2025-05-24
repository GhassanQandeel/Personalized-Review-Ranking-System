import pandas as pd
import json
import gzip

import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('Movies_and_TV.json.gz')

df = df[['reviewerID', 'reviewText']]


# Step 1: Get 30 unique reviewerIDs
selected_reviewers = df['reviewerID'].drop_duplicates().head(30)

# Step 2: Filter the DataFrame to keep only rows with those reviewerIDs
df_30 = df[df['reviewerID'].isin(selected_reviewers)]

# Step 3: (Optional) Reset index for clean output
df_30 = df_30.reset_index(drop=True)
df_30.to_csv('Movies_and_TV_User.csv')
# Preview result
print(df_30.head())