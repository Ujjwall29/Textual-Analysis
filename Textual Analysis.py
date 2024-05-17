#!/usr/bin/env python
# coding: utf-8

# Loading the required libraries

# In[ ]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import os


# Lets remove the stop words from the articles using respective excel files

# In[ ]:


# Function to load stop words from multiple files into a set
def load_stop_words_from_files(file_paths):
    stop_words = set()
    for file_path in file_paths:
        with open(file_path, "r", encoding="latin-1") as file:
            try:
                for line in file:
                    # Split each line based on the '|' character and add individual words to the set
                    words = line.strip().lower().split('|')
                    for word in words:
                        stop_words.add(word.strip())  # Convert to lowercase and remove leading/trailing whitespace
            except UnicodeDecodeError:
                print(f"Error decoding file: {file_path}. Trying alternative encodings.")
                with open(file_path, "r", encoding="latin-1") as file_alt:
                    for line in file_alt:
                        words = line.strip().lower().split('|')
                        for word in words:
                            stop_words.add(word.strip())
    return stop_words

# List of file paths containing stop words
stop_word_files = ["C:\\Users\\Ujjawal\\Downloads\\StopWords_Auditor.txt",
                  "C:\\Users\\Ujjawal\\Downloads\\StopWords_Currencies.txt",
                  "C:\\Users\\Ujjawal\\Downloads\\StopWords_DatesandNumbers.txt",
                  "C:\\Users\\Ujjawal\\Downloads\\StopWords_Generic.txt",
                  "C:\\Users\\Ujjawal\\Downloads\\StopWords_GenericLong.txt",
                  "C:\\Users\\Ujjawal\\Downloads\\StopWords_Geographic.txt",
                  "C:\\Users\\Ujjawal\\Downloads\\StopWords_Names.txt"]

# Load stop words from multiple files into a set
stop_words = load_stop_words_from_files(stop_word_files)


# Making dictionaries for the positive and negative words

# In[ ]:


# Function to load words from file into a dictionary
def load_words_from_file(file_path):
    word_dict = {}
    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            word = line.strip().lower()  # Convert to lowercase and remove whitespace
            word_dict[word] = 1  # Assign a sentiment score of 1 (positive) or -1 (negative)
    return word_dict

# Load positive and negative words from files into dictionaries
positive_dict = load_words_from_file("C:\\Users\\Ujjawal\\Downloads\\positive-words.txt")
negative_dict = load_words_from_file("C:\\Users\\Ujjawal\\Downloads\\negative-words.txt")


# Extracting variables for Sentimental analysis

# But before that, lets extract article text from the links given in the excel file

# In[ ]:


# Function to fetch text content from article links
def fetch_article_text(article_link):
    try:
        response = requests.get(article_link)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from HTML content (you might need to adjust this based on the structure of the articles)
        article_text = soup.get_text(separator=' ', strip=True)
        return article_text
    except Exception as e:
        print(f"Error fetching article from link: {article_link}. Error: {e}")
        return None

# Load the DataFrame containing articles
df = pd.read_excel("C:\\Users\\Ujjawal\\Downloads\\Input.xlsx")

df['URL'] = df['URL'].apply(fetch_article_text)


# Extracting following variables using their given formulas:
# 1.) Positivity score
# 2.) Negativity score
# 3.) Polarity score
# 4.) Subjectivity score

# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to assign sentiment scores to words in the article
def assign_sentiment_scores(article_text, positive_dict, negative_dict):
    positive_score = 0
    negative_score = 0
    total_words = 0
    
    words = word_tokenize(article_text.lower())  # Tokenize the text and convert to lowercase
    for word in words:
        if word in positive_dict:
            positive_score += 1
        elif word in negative_dict:
            negative_score += 1
        total_words += 1
    
    return positive_score, negative_score, total_words

# Function to analyze sentiment for each article
def analyze_sentiment(article_text, positive_dict, negative_dict):
    positive_score, negative_score, total_words = assign_sentiment_scores(article_text, positive_dict, negative_dict)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / ((total_words) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

# Calculate sentiment scores for each article
sentiment_scores = df['URL'].apply(lambda x: analyze_sentiment(x, positive_dict, negative_dict) if x else (0, 0, 0, 0))

# Add sentiment scores to the DataFrame
df[['Positive_Score', 'Negative_Score', 'Polarity_Score', 'Subjectivity_Score']] = pd.DataFrame(sentiment_scores.tolist())

# Print or save the DataFrame with sentiment scores
print(df)


# Calculating Average sentence length, Percentage of complex words, and Gunning Fog Index

# In[ ]:


from nltk.tokenize import sent_tokenize

# Function to calculate average sentence length
def calculate_average_sentence_length(text):
    sentences = sent_tokenize(text)
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    total_sentences = len(sentences)
    return total_words / total_sentences if total_sentences > 0 else 0

# Function to calculate percentage of complex words
def calculate_percentage_complex_words(text):
    words = word_tokenize(text)
    complex_words = [word for word in words if len(word) > 2 and word.lower() not in stopwords.words('english')]
    return (len(complex_words) / len(words)) * 100 if len(words) > 0 else 0

# Function to calculate Gunning Fog Index
def calculate_gunning_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

# Calculate average sentence length for each article
df['Average_Sentence_Length'] = df['URL'].apply(lambda x: calculate_average_sentence_length(x) if x else 0)

# Calculate percentage of complex words for each article
df['Percentage_Complex_Words'] = df['URL'].apply(lambda x: calculate_percentage_complex_words(x) if x else 0)

# Calculate Gunning Fog Index for each article
df['Gunning_Fog_Index'] = df.apply(lambda row: calculate_gunning_fog_index(row['Average_Sentence_Length'], row['Percentage_Complex_Words']), axis=1)

# Print or save the DataFrame with the new columns
print(df)


# Average number of words per sentence

# In[ ]:


# Function to calculate average number of words per sentence
def calculate_average_words_per_sentence(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences) if len(sentences) > 0 else 0

# Calculate average number of words per sentence for each article
df['Average_Words_Per_Sentence'] = df['URL'].apply(lambda x: calculate_average_words_per_sentence(x) if x else 0)

# Print or save the DataFrame with the new column
print(df)


# Total number of complex words

# In[ ]:


# Function to count number of syllables in a word
def count_syllables(word):
    vowels = 'aeiouy'
    syllables = 0
    prev_char = ''
    for char in word:
        if char.lower() in vowels and prev_char not in vowels:
            syllables += 1
        prev_char = char
    if word.endswith('e'):
        syllables -= 1
    if syllables == 0:
        syllables += 1
    return syllables

# Function to calculate total number of complex words
def calculate_total_complex_words(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    complex_words = [word for word in words if count_syllables(word) > 2 and word.lower() not in stop_words]
    return len(complex_words)

# Calculate total number of complex words for each article
df['Total_Complex_Words'] = df['URL'].apply(lambda x: calculate_total_complex_words(x) if x else 0)

# Print or save the DataFrame with the new column
print(df)


# Total number of clean words

# In[ ]:


def count_total_cleaned_words(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
    return len(cleaned_words)

# Calculate total cleaned words for each article
df['Total_Cleaned_Words'] = df['URL'].apply(lambda x: count_total_cleaned_words(x) if x else 0)

# Print or save the DataFrame with the new column
print(df)


# Total number of syllables in the text

# In[ ]:


import string

# Function to count number of syllables in a word
def count_syllables(word):
    exceptions = ["es", "ed"]  # Exceptions to syllable count
    word = word.lower().rstrip(string.punctuation)  # Convert to lowercase and remove trailing punctuation
    vowels = 'aeiouy'
    syllables = 0
    prev_char = ''
    for char in word:
        if char.lower() in vowels and prev_char not in vowels:
            syllables += 1
        prev_char = char
    if word.endswith('e') and word[-2:] not in exceptions:
        syllables -= 1
    if syllables == 0:
        syllables += 1
    return syllables

# Function to count syllables in each word of the text
def count_syllables_in_text(text):
    words = word_tokenize(text)
    syllable_counts = [count_syllables(word) for word in words]
    return sum(syllable_counts)

# Calculate total number of syllables for each article
df['Total_Syllables'] = df['URL'].apply(lambda x: count_syllables_in_text(x) if x else 0)

# Print or save the DataFrame with the new column
print(df)


# Total number of personal pronouns

# In[ ]:


import re

# Function to count personal pronouns in the text
def count_personal_pronouns(text):
    # Define regex pattern for personal pronouns
    pattern = r'\b(?:I|we|my|ours|us)\b'
    # Compile regex pattern
    regex = re.compile(pattern, flags=re.IGNORECASE)
    # Find all matches in the text
    matches = regex.findall(text)
    # Return the count of matches
    return len(matches)

# Calculate total number of personal pronouns for each article
df['Total_Personal_Pronouns'] = df['URL'].apply(lambda x: count_personal_pronouns(x) if x else 0)

# Print or save the DataFrame with the new column
print(df)


# Average word length

# In[ ]:


# Function to calculate average word length
def calculate_average_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    return total_characters / total_words if total_words > 0 else 0

# Calculate average word length for each article
df['Average_Word_Length'] = df['URL'].apply(lambda x: calculate_average_word_length(x) if x else 0)

# Print or save the DataFrame with the new column
print(df)


# In[ ]:


# Converting dataframe into an excel file

df.to_excel('output.xlsx', index=False)


# In[ ]:


# Checking if the Excel file exists
if os.path.exists('output.xlsx'):
    print("Excel file has been successfully created.")
else:
    print("Excel file creation failed.")


# In[ ]:




