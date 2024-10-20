import os
from extractingcomments import YouTubeCommentsExtractor
from flask import Flask, request, render_template
from onlycomments import OnlyComments
import pandas
import re
import nltk
import string
from nltk.corpus import stopwords
from textblob import TextBlob
import yt_video
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

class YouTubeCommentsAnalyzer:
    punctuation = punctuation + "\n"
    def __init__(self, api_key):
        self.api_key = api_key
        self.extractor = YouTubeCommentsExtractor(api_key)
        self.stop_words = set(stopwords.words('english'))
        self.emoji_pattern = re.compile("["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F700-\U0001F77F"  # alchemical symbols
                                        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                        u"\U00002702-\U000027B0"  # Dingbats
                                        u"\U000024C2-\U0001F251"
                                        "]+", flags=re.UNICODE)

    def extract_comments(self, video_id):
        max_results = 1000  # You can adjust this number
        comments = self.extractor.extract_comments(video_id, max_results)
        output_file = "youtube_comments.csv"
        self.extractor.save_to_csv(comments, output_file)
        return output_file

    def remove_stopwords(self, text):
        return " ".join([word for word in str(text).split() if word.lower() not in self.stop_words])

    def remove_emojis(self, text):
        return self.emoji_pattern.sub(r'', text)

    def clean_text(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'@\w+', '', text)
        return text.strip()

    def analyze_sentiment(self, comment):
        analysis = TextBlob(comment)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    def get_result(self, videoID):
        output_file = self.extract_comments(videoID)
        only_comments = OnlyComments()
        only_comments.extract_only_comments(output_file)

        data = pandas.read_csv("extracted_comments_new.csv")
        removed_stop_word = data["Comment"].apply(self.remove_stopwords)
        removed_emoji = self.remove_emojis(removed_stop_word.to_string())
        cleaned_text = self.clean_text(removed_emoji)
        cleaned_comments_list = [comment.strip(" ") for comment in cleaned_text.split("\n")]

        sentiments = [self.analyze_sentiment(comment) for comment in cleaned_comments_list]
        positive_count = sentiments.count('Positive')
        neutral_count = sentiments.count('Neutral')
        negative_count = sentiments.count('Negative')

        return {
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "negative_count": negative_count
        }
        
    def negative_comments_extract(self,videoID):
        output_file=self.extract_comments(videoID)
        only_comments=OnlyComments()
        only_comments.extract_only_comments(output_file)
        data=pandas.read_csv("extracted_comments_new.csv")
        removed_stop_word = data["Comment"].apply(self.remove_stopwords)
        removed_emoji = self.remove_emojis(removed_stop_word.to_string())
        cleaned_text = self.clean_text(removed_emoji)
        cleaned_comments_list = [comment.strip(" ") for comment in cleaned_text.split("\n")]
        sentiments=[self.analyze_sentiment(comment) for comment in cleaned_comments_list]
        negative_comments=[]
        for comment, sentiment in zip(cleaned_comments_list, sentiments):
            if sentiment == 'Negative':
                negative_comments.append(comment)

        return negative_comments

    # def generate_negative_summary(self,videoID):
    #     negative_comments = self.negative_comments_extract(videoID)
    #     text = " ".join(negative_comments)
    #     stopwords=list(STOP_WORDS)
    #     nlp=spacy.load('en_core_web_sm')
    #     doc=nlp(text)
    #     tokens=[token.text for token in doc]
    #     word_frequency={}
    #     for word in doc:
    #         if word.text.lower() not in stopwords:
    #             if word.text.lower() not in punctuation:
    #                 if word.text not in word_frequency.keys():
    #                     word_frequency[word.text]=1
    #                 else:
    #                     word_frequency[word.text]+=1
    #     max_frequency=max(word_frequency.values()) 
    #     for word in word_frequency.keys():
    #         word_frequency[word]=word_frequency[word]/max_frequency
    #     sentence_tokens=[sent for sent in doc.sents]          

    #     sentence_scores={}
    #     for sent in sentence_tokens:
    #         for word in sent:
    #             if word.text.lower() in word_frequency.keys():
    #                 if sent not in sentence_scores.keys():
    #                     sentence_scores[sent]=word_frequency[word.text.lower()]
    #                 else:
    #                     sentence_scores[sent]+=word_frequency[word.text.lower()]
    #     select_length=int(len(sentence_tokens)*0.3)
    #     summary=nlargest(select_length,sentence_scores,key=sentence_scores.get)
    #     final_summary=[word.text for word in summary]

    #     summary=' '.join(final_summary)
    #     return summary

    def generate_negative_summary(self, videoID):
        negative_comments = self.negative_comments_extract(videoID)
        print("Negative Comments:", negative_comments)

        text = " ".join(negative_comments)
        print("Text:", text)

        stopwords = list(STOP_WORDS)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)

        tokens = [token.text for token in doc]
        print("Tokens:", tokens)

        word_frequency = {}
        for word in doc:
            if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
                if word.text not in word_frequency.keys():
                    word_frequency[word.text] = 1
                else:
                    word_frequency[word.text] += 1
        print("Word Frequency:", word_frequency)

        max_frequency = max(word_frequency.values())
        for word in word_frequency.keys():
            word_frequency[word] = word_frequency[word] / max_frequency

        sentence_tokens = [sent for sent in doc.sents]
        print("Sentence Tokens:", sentence_tokens)

        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequency.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequency[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequency[word.text.lower()]
        print("Sentence Scores:", sentence_scores)

        select_length = int(len(sentence_tokens) * 0.3)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word.text for word in summary]

        summary = ' '.join(final_summary)
        print("Final Summary:", summary)
        return summary
