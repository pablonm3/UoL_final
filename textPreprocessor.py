import string
import spacy;
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

ps = PorterStemmer()

nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
nltk.download('stopwords')
nltk.download('punkt')
nltk_stops = stopwords.words("english")

class TextPreprocessor:

    def __init__(self):
        self.tfidf = TfidfVectorizer(sublinear_tf=True)

    def fit_tfidf(self, corpus):
        self.tfidf.fit(corpus)

    def _get_high_tfidf_words(self, test_text, top_n):
        tfidf_matrix = self.tfidf.transform(test_text)

        # Getting the words from the vocabulary
        feature_names = self.tfidf.get_feature_names_out()

        # Get the tfidf scores for each word in each document
        df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        # Get the sum of tfidf scores for each word across all documents
        sum_scores = df_tfidf.sum(axis=0)

        # Get the top_n words with highest TF-IDF scores
        top_words = sum_scores.nlargest(top_n)

        return top_words.index.values.tolist()

    def _remove_punctuation(self, sentences):
        # inspired by source: https://www.kaggle.com/code/pablomarino/from-shallow-learning-to-2020-sota-gpt-2-roberta
        r = []
        for text in sentences:
            table = str.maketrans('', '', string.punctuation)
            new_text = text.translate(table)
            r.append(new_text)
        return r;

    def _remove_stopwords(self, sentences, stopwords=nltk_stops):
        # inspired by source: https://www.kaggle.com/code/pablomarino/from-shallow-learning-to-2020-sota-gpt-2-roberta
        r = []
        for text in sentences:
            text_nlp = nlp(text)
            words = [];
            for token in text_nlp:
                if token.text not in stopwords:
                    words.append(token.text)
            clean_text = " ".join(words)
            r.append(clean_text)
        return r;

    def _lemmatize(self, sentences):
        # inspired by source: https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
        r = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            # using reduce to apply stemmer to each word and join them back into a string
            stemmed_sentence = reduce(lambda x, y: x + " " + lemmatizer.lemmatize(y), words, "")
            r.append(stemmed_sentence.strip())
        return r;

    def _steeming(self, sentences):
        # source: https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
        r = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            # using reduce to apply stemmer to each word and join them back into a string
            stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")
            r.append(stemmed_sentence.strip())
        return r

    def _TFIDF_word_removal(self, sentences):
        # this is like stopword removal, but rather than using fixed list common words are obtained using TF-IDF
        top_n = len(nltk_stops)
        common_words = self._get_high_tfidf_words(sentences, top_n=top_n)
        return self._remove_stopwords(sentences, common_words)

    def _lowercase(self, sentences):
        r = []
        for text in sentences:
            r.append(text.lower())
        return r;

    def run(self, sentences, rem_stopwords, word_normalization, lowercasing, remove_punctuation, TFIDF_word_removal):
        r = sentences
        if rem_stopwords:
            r = self._remove_stopwords(r)
        if word_normalization=="lemmatization":
            r = self._lemmatize(r)
        elif word_normalization=="stemming":
            r = self._steeming(r)
        if lowercasing:
            r = self._lowercase(r)
        if remove_punctuation:
            r = self._remove_punctuation(r)
        if TFIDF_word_removal:
            r = self._TFIDF_word_removal(r)
        return r;