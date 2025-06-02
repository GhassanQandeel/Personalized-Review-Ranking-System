# Install the necessary libraries before running this script (only once)
# pip install googletrans==4.0.0-rc1 tqdm pandas

import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import os
import time
"""
tqdm.pandas()  # Enable tqdm for pandas

# === SET YOUR FILE PATHS HERE ===
input_csv_path = 'filtered_reviews/B000WGWQG8.csv'



output_csv_path = 'Translated_Reviews/B000WGWQG8_O.csv'

# Load the CSVs
df = pd.read_csv(input_csv_path)


# Step 2: Select 10,000 records
df_selected = df.copy()

# Initialize Google Translate client
translator = Translator()

# Translation function to Arabic
def translate_to_arabic(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    try:
        translated = translator.translate(text, src='en', dest='ar')
        time.sleep(0.5)  # Optional: prevent rate limiting
        return translated.text
    except Exception as e:
        print(f"Error translating: {e}")
        return ""

# Translate reviews with progress bar
df_selected['reviewText_ar'] = df_selected['reviewText'].progress_apply(translate_to_arabic)

# Save the translated results
df_selected.to_csv(output_csv_path, index=False)

print(f"Translation complete. File saved to: {os.path.abspath(output_csv_path)}")
"""
import pandas as pd
import os
df=pd.read_csv('Combined_Translated_Reviews.csv')
# Drop rows where 'reviewText_ar' is null
cleaned_df = df.dropna(subset=['reviewText_ar'])

# Save the cleaned dataset
cleaned_df.to_csv('Cleaned_Translated_Reviews.csv', index=False)