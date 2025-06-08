import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import json


def create_user_profiles_for_your_data(user_id):
    with open('user_profiles.json', 'r', encoding='utf-8') as f:
        user_profiles = json.load(f)
    user_queries=[]
    for profile in user_profiles['user_profiles']:
        user_queries.append(profile)


import pandas as pd

# Load the data (assuming it's a CSV file)
df = pd.read_csv("Movies_and_TV_Reviews_ar.csv")  # Replace with your actual file path

# Remove duplicates based on 'reviewText' and keep the first occurrence
df_cleaned = df.drop_duplicates(subset=['reviewText_ar'], keep='first')

# Reset index (optional)
df_cleaned = df_cleaned.reset_index(drop=True)

# Save the cleaned data (if needed)
df_cleaned.to_csv("Movies_and_TV_Reviews_ar1.csv", index=False)

# Display the cleaned DataFrame
print(df_cleaned)