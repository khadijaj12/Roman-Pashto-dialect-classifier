import pandas as pd
import numpy as np
import os

# Create folders for organized data
os.makedirs('roman_script_data', exist_ok=True)
os.makedirs('pashto_script_data', exist_ok=True)

# Load the raw data
df = pd.read_csv('rawdata(north) (2).csv')

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# ==================== DATA CLEANING ====================

# 0. Remove Timestamp column
df = df.drop('Timestamp', axis=1)
print("0. Removed Timestamp column")

# 0.5. Rename and standardize dialect column
df = df.rename(columns={'Which dialect do you speak': 'Dialect'})
# Extract dialect type and standardize
df['Dialect'] = df['Dialect'].str.lower()
df['Dialect'] = df['Dialect'].apply(lambda x: 'north' if 'noth' in str(x) else ('south' if 'south' in str(x) else 'other'))
print("0.5. Renamed dialect column and standardized entries")

# 1. Remove duplicate rows
df = df.drop_duplicates()
print("1. Shape after removing duplicates:", df.shape)

# 2. Drop rows with all missing values
df = df.dropna(how='all')
print("2. Shape after removing all-null rows:", df.shape)

# ==================== OUTLIER REMOVAL ====================

# 3. Remove rows with more than 50% missing values (outliers)
threshold = 0.5
missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
df = df[missing_ratio < threshold]
print("\n3. Shape after outlier removal (>50% missing):", df.shape)

# 4. Remove rows with very short text responses (likely invalid entries)
answer_columns = df.columns[4:]  # Answer columns start from index 4
for col in answer_columns:
    if df[col].dtype == 'object':
        # Keep only rows where text responses have meaningful length (>3 characters)
        df = df[df[col].isna() | (df[col].astype(str).str.len() > 3)]

print("4. Shape after removing short/invalid responses:", df.shape)

# ==================== CONCATENATE ANSWER COLUMNS ====================

# 5. Concatenate all answer columns into one column
answer_columns = df.columns[4:]
df['Concatenated_Responses'] = df[answer_columns].apply(
    lambda row: ' | '.join(row.dropna().astype(str)), axis=1
)

print("\n5. Concatenated answer columns into single column")

# ==================== SEPARATE BY SCRIPT TYPE ====================

# 6. Create separate tables for Roman (Latin) script vs Pashto script
# Roman script: contains Latin characters (a-z, A-Z)
# Pashto script: contains Pashto/Dari characters (Arabic-based)

def detect_script_type(text):
    """Detect if text contains primarily Latin or Arabic script"""
    if pd.isna(text):
        return 'Unknown'
    
    text = str(text)
    # Check for Pashto/Arabic characters
    arabic_range = range(0x0600, 0x06FF)  # Arabic Unicode range
    pashto_chars = sum(1 for char in text if ord(char) in arabic_range)
    
    # Check for Latin characters
    latin_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    
    if pashto_chars > latin_chars:
        return 'Pashto_Script'
    elif latin_chars > pashto_chars:
        return 'Roman_Script'
    else:
        return 'Mixed'

# Detect script for concatenated responses
df['Script_Type'] = df['Concatenated_Responses'].apply(detect_script_type)

print("\n6. Script type detection:")
print(df['Script_Type'].value_counts())

# 7. Create separate tables
roman_script_df = df[df['Script_Type'] == 'Roman_Script'].copy()
pashto_script_df = df[df['Script_Type'] == 'Pashto_Script'].copy()
mixed_script_df = df[df['Script_Type'] == 'Mixed'].copy()

print("\nRoman script entries:", len(roman_script_df))
print("Pashto script entries:", len(pashto_script_df))
print("Mixed script entries:", len(mixed_script_df))

# ==================== SAVE CLEANED DATA ====================

# Save main cleaned data
df.to_csv('cleaned_data.csv', index=False)
print("\n✓ Cleaned data saved to 'cleaned_data.csv'")

# Save script-separated tables to respective folders
# Roman script data (remove Script_Type column as it's redundant)
roman_script_df_export = roman_script_df.drop('Script_Type', axis=1)
roman_script_df_export.to_csv('roman_script_data/cleaned_data_roman_script.csv', index=False)

pashto_script_df.to_csv('pashto_script_data/cleaned_data_pashto_script.csv', index=False)
mixed_script_df.to_csv('pashto_script_data/cleaned_data_mixed_script.csv', index=False)

print("✓ Roman script data saved to 'roman_script_data/cleaned_data_roman_script.csv'")
print("✓ Pashto script data saved to 'pashto_script_data/cleaned_data_pashto_script.csv'")
print("✓ Mixed script data saved to 'pashto_script_data/cleaned_data_mixed_script.csv'")

# ==================== CREATE ORGANIZED TABLES ====================

# Create a clean summary table with key information (remove Script_Type for Roman)
summary_table_roman = roman_script_df[['Dialect', 'District']].copy()
summary_table_roman.insert(0, 'Id', range(1, len(summary_table_roman) + 1))
summary_table_roman = summary_table_roman[['Id', 'Dialect', 'District']]

# Summary table for Pashto (keep Script_Type)
summary_table_pashto = pashto_script_df[['Dialect', 'District', 'Script_Type']].copy()
summary_table_pashto.insert(0, 'Id', range(1, len(summary_table_pashto) + 1))
summary_table_pashto = summary_table_pashto[['Id', 'Dialect', 'District', 'Script_Type']]

# Save organized summary tables
summary_table_roman.to_csv('roman_script_data/summary_table_roman.csv', index=False)
summary_table_pashto.to_csv('pashto_script_data/summary_table_pashto.csv', index=False)

print("✓ Roman script summary table saved to 'roman_script_data/summary_table_roman.csv'")
print("✓ Pashto script summary table saved to 'pashto_script_data/summary_table_pashto.csv'")

# Create detailed response tables
response_table_roman = roman_script_df[['Concatenated_Responses']].copy()
response_table_roman.insert(0, 'Id', range(1, len(response_table_roman) + 1))
response_table_roman.to_csv('roman_script_data/responses_table_roman.csv', index=False)

response_table_pashto = pashto_script_df[['Concatenated_Responses', 'Script_Type']].copy()
response_table_pashto.insert(0, 'Id', range(1, len(response_table_pashto) + 1))
response_table_pashto = response_table_pashto[['Id', 'Concatenated_Responses', 'Script_Type']]
response_table_pashto.to_csv('pashto_script_data/responses_table_pashto.csv', index=False)

print("✓ Roman script responses table saved to 'roman_script_data/responses_table_roman.csv'")
print("✓ Pashto script responses table saved to 'pashto_script_data/responses_table_pashto.csv'")

# ==================== DISPLAY ORGANIZED TABLES ====================

print("\n" + "="*80)
print("ORGANIZED DATA SUMMARY TABLE")
print("="*80)
summary_all = df[['Dialect', 'District', 'Script_Type']].copy()
summary_all.insert(0, 'Id', range(1, len(summary_all) + 1))
summary_all = summary_all[['Id', 'Dialect', 'District', 'Script_Type']]
print(summary_all.to_string(index=False))

print("\n" + "="*80)
print("ROMAN SCRIPT DATA SUMMARY")
print("="*80)
print(summary_table_roman.to_string(index=False))

print("\n" + "="*80)
print("PASHTO SCRIPT DATA SUMMARY")
print("="*80)
print(summary_table_pashto.to_string(index=False))

# ==================== SUMMARY ====================
print("\n" + "="*50)
print("CLEANING SUMMARY")
print("="*50)
print(f"Original rows: 27")
print(f"Final cleaned rows: {len(df)}")
print(f"Rows removed: {27 - len(df)}")
print("\nFinal dataset shape:", df.shape)
print("\nColumns in cleaned data:")
print(df.columns.tolist())
