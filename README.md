## Project3A23

The purpose of this project is to successfully develop a sentiment analysis system to analyse public opinions toward artificial inteligence (AI) from various resources such as websites, podcasts, videos, etc. The goal is to understand general sentiment (positive, negative, or neutral) towards AI-realted topics. This README file has been created to make it easier to use this repository. You will find information here about the structure and what files to use to create the final product.

## Team members

* Justyna Dobersztajn
* Diba Dabiransari
* Aleeza Azad

## Development Methodology
For this assignment we used the Cross Industry Standard Process for Data Mining (CRISP DM) to move on with our project. There are various steps to this model which are:
* Business Understanding
* Data Understanding
* Data Preparation                                                                                                     
* Modelling
* Evaluation
* Deployment

![CRISP-DM](https://github.com/dibadabir/Project3A23/assets/152966994/3ceeae0c-6898-4fb1-892b-ce43fb3af03b)

## 1. What does the business need?
As mentioned earlier, the aim of this project is to analyse public opinion and gain a better understanding of their thoughts about AI. We identified categories in which AI is utilised and did some in-depth research about them.
Moreover, this project is supposed to be a good practice for us to improve our skills, both technical and communicative skills. We were able to explore and develop a machine learning model whilst being exposed to diffrent platforms suh as google colab and pycharm.

## 2. What data do we need?
For this analysis, we chose six categories representing sectors impacted by artificial intelligence:
* AI in Healthcare
* AI in Travel Industry
* AI in Fashion
* AI in E-Commerce / Marketing
* AI in Education
* Generative AI
  
Our aim was to gather data that both describes and critiques the utilisation of artificial systems within these domains.

## 3. Cleaning, Processing, and labeling
After doing some research and choosing our subject, we started looking for websites and scraping the text out of them. We used BeautifulSoup to extract text from websites and Deepgram to transcribe podcasts.

The data that were scraped from websites and audios, were raw data and we were not able to use it just on its own. We had to process the textual data in a way that would be understandable for the machine. We first cleaned the collected data, the punctuation marks, special characters and stop words were removed. Instead of completely deleting numeric numbers, we replaced them with their written forms. After that, split paragraphs into sentences so that the machine can process them better.

The next step was to label our data. We used to diffrent libraries to tag our data with Positive, Negative, or Neutral. The libraries used were TextBlob and Vader, both widely recognized in the field of Natural Language Processing (NLP). We used them to compute the polarity of the sentences and then label them based on that criteria.

![Web-scraping ML Pipeline Diagram](https://github.com/dibadabir/Project3A23/assets/152966994/a239306c-ebb3-4523-9bfc-27c035498349)


```
def scrape_clean (url, div_class,id_name=None):
  # Load the website
  website = requests.get(url).text
  soup = bs(website,'html.parser')

  # Find the div section that is the parent of all paragraphs
  div = soup.find_all('div', attrs={'class':div_class, 'id':id_name})

  # Get the text from paragraphs
  extracted_text = []
  for div in div:
      paragraphs = div.find_all('p')
      for paragraph in paragraphs:
          extracted_text.append(paragraph.get_text(strip=True))  # Remove leading/trailing whitespace

  # Splitting the text into sentences, remove the stopwords and punctuations, and save the cleaned version in a list
  sentences = []
  for text in extracted_text:
    sentence = sent_tokenize(text)
    for item in sentence:
      # Convert to lowercase
      text_lowercase = item.lower()
      # Remove punctuation
      text_without_punctuation = re.sub(r"[^\w\s]", "", text_lowercase)
      # Remove stopwords and stem words
      tokens = word_tokenize(text_without_punctuation)
      new_tokens = []
      for word in tokens:
        if word.isnumeric():
          word = num2words(word)
          new_tokens.append(lemmatizer.lemmatize(word))
          continue
        elif word not in stop_words:
          new_tokens.append(lemmatizer.lemmatize(word))
          continue
      # Join tokens back into a string
      cleaned_text = " ".join(new_tokens)
      sentences.append(cleaned_text)

  return sentences
```

### Example of **web scraping** implementation - Education

#### Using TextBlob
* [Web-Scraping Code File](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/Education_webscrape_code%20file%20without%20numbers%20in%20the%20dataset.ipynb)
* [Scraped Results](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/education%20dataset%20(no%20numbers).csv)

#### Using Vader
* [Web-Scraping Code File](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/Education_webscrape_(without_numbers)_Vader_ver_.ipynb)
* [Scraped Results](https://github.com/dibadabir/Project3A23/blob/main/Web%20Scraping/Education/Education%20dataset%20(no%20numbers)%20-%20Vader%20ver.csv)

### Example of **audio scraping** implementation - Education
* [Audio-Scraping Code File](https://github.com/dibadabir/Project3A23/blob/main/Speech%20to%20Text/Education/Audio_Scraping_Using_DEEPGRAM(Education).ipynb)
* [Scraped Results](https://github.com/dibadabir/Project3A23/blob/main/Speech%20to%20Text/Fashion/Audio_Scraping_Using_DEEPGRAM(Fashion).ipynb)

## 4. Choosing the best model
Following the machine learning pipeline, the next phase would be choosing an appropriate model for sentiment analysis. We explored a variety of supervised learning algorithms such as **Naive Bayes**, **Logistic Regression**, **Random Forest**, and **Decision Tree** using two vectorizing methods, CountVectorizer and TF-IDF Vectorizer and then we compared the accuracy of these models and how diffrnt each model performs. We then demonstrated the results to gain a better understanding of them and make a better decision.

[Training Models using CountVectorizer](https://github.com/dibadabir/Project3A23/blob/main/Final%20(Everything%20combined!)/Compare_Models_(CountVectorizer).ipynb)

[Training Models using TF-IDF Vectorizer](https://github.com/dibadabir/Project3A23/blob/main/Final%20(Everything%20combined!)/Compare_Models_(TF_IDFVectorizer).ipynb)

## 5. Evaluating Models
We then assess the performance of the trained sentiment analysis models using appropriate matrices such as F1-Score, Recall, Precision, and Accuracy to understand our model and how it performs. Then we used matplotlib to visualise the classification report and choose the best model to use.

![image](https://github.com/dibadabir/Project3A23/assets/152966994/8ec4f527-3106-45b6-bed0-7b357245032d)
![image](https://github.com/dibadabir/Project3A23/assets/152966994/c625eabe-0a75-49df-87b6-eb539190deac)

## 6. Deployment
we used our chosen classifier to train our model and perform the sentiment analysis on the unlabeled data scraped from podcasts to find out what is public's opinion about AI in general.

![image](https://github.com/dibadabir/Project3A23/assets/152966994/6098fec9-a821-4a14-b94c-a8289767e2d5)

## 7. Product Compatibility

We made sure that our code files and models are compatible and can run smoothly on other platforms such as PyCharm, ensuring seamless integration and ease of use for developers across different environments.

![image-b62dff59-eb99-4d61-85ab-dd7d9233a375](https://github.com/dibadabir/Project3A23/assets/152966994/76e2937c-3a1c-45b0-bcc2-ede259c62392)



## Libraries used in this project

```
import re
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import requests
import nltk
import os
import httpx
import json
import pickle
import logging, verboselogs
from bs4 import BeautifulSoup as bs
from num2words import num2words
from textblob import TextBlob
from dotenv import load_dotenv
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepgram import ( DeepgramClient, DeepgramClientOptions, PrerecordedOptions, FileSource)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
```

## Version History

- Web-Scraping with TextBlob, including numbers
- Web-Scraping with TextBlob, without numeric numbers
- Web-scraping with Vader, without numeric numbers
- Audio Scraping with Deepgram
- Compare models with the final data, CountVectorizer
- Compare models with the final data, TF-IDFVectorizer
- Sentiment Analysis using Random Forest
