import string

class TextPreprocessor:

    def _remove_punctuation(self, sentences):
        #TODO: does this work?
        return sentences.translate(str.maketrans('', '', string.punctuation))

    def _remove_stopwords(self, sentences):
        pass;

    def _lemmatize(self, sentences):
        pass;

    def _steeming(self, sentences):
        pass;

    def _TFIDF_word_removal(self, sentences):
        pass;

    def _lowercase(self, sentences):
        pass;

    def run(self, sentences, rem_stopwords, _word_normalization, lowercasing, remove_punctuation, TFIDF_word_removal):
        pass;