import pandas as pd
import spacy
from collections import defaultdict

df = pd.read_csv('Movies_and_TV_User.csv')

nlp = spacy.load('en_core_web_sm')

user_profiles = defaultdict(set)

# Process each row
for _, row in df.iterrows():
    user = row['user_id']
    review = row['review']
    doc = nlp(str(review))

    #stemming, tokinize , normlize , extract POS,
    keywords = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop]


    user_profiles[user].update(keywords)


user_profiles = {user: list(keywords) for user, keywords in user_profiles.items()}



import json

with open('user_profiles.json', 'w') as f:
    json.dump(user_profiles, f, indent=2)
