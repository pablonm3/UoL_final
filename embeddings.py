import hashlib

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

from cache import RedisCache


def average_pool(last_hidden_states):
    # adapted for Tensorflow from: https://www.kaggle.com/code/pablomarino/from-shallow-learning-to-2020-sota-gpt-2-roberta?scriptVersionId=38895686&cellId=108
    return tf.reduce_mean(last_hidden_states, axis=1)

def sum_pool(last_hidden_states):
    # adapted for Tensorflow from: https://www.kaggle.com/code/pablomarino/from-shallow-learning-to-2020-sota-gpt-2-roberta?scriptVersionId=38895686&cellId=108
    return tf.reduce_sum(last_hidden_states, axis=1)

def max_pool(last_hidden_states):
    return tf.reduce_max(last_hidden_states, axis=1)
def concat_pool(last_hidden_states):
    # adapted for Tensorflow from: https://www.kaggle.com/code/pablomarino/from-shallow-learning-to-2020-sota-gpt-2-roberta?scriptVersionId=38895686&cellId=108
    batch_size, max_tokens, emb_dim = last_hidden_states.shape
    return tf.reshape(last_hidden_states, (batch_size, max_tokens * emb_dim))


class EmbeddingGenerator:
    cache = None
    def __init__(self, config, model_name):
        self.model_name = model_name
        self.config = config
        if(not EmbeddingGenerator.cache):
            clear_cache = self.config["CLEAR_CACHE"]
            EmbeddingGenerator.cache = RedisCache(clear_cache)
        # Load pre-trained model tokenizer (BERT-base uncased)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModel.from_pretrained(model_name)
        self.max_tokens = 512  # FIXME this depends on the model
    def sentence_embedding(self, sentences, comb_strategy):
        sentence_text = ".".join(sentences) # convert list to string, md5 doesn't support lists
        hash = hashlib.md5(sentence_text.encode('utf-8')).hexdigest()
        key = hash + "_" + self.model_name
        if(self.cache.exists(key)):
            print("sentence_embedding cache hit, hash:", key)
            last_hidden_state = self.cache.get(key)
        else:
            print("sentence_embedding cache miss hash: ", key)
            # adapted for tensorflow from source: https://huggingface.co/intfloat/e5-small-v2#usage
            # removed normalization step due to research that suggest that vector length may contain valuable information: https://arxiv.org/abs/1508.02297
            # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            _sentences = sentences[:] # clone sentences
            if("e5" in self.model_name):
                _sentences = [f"query: {s}" for s in _sentences] # recommended usage for this model: https://huggingface.co/intfloat/e5-small-v2#usage

            batch_dict = self.tokenizer(sentences, add_special_tokens=True, return_tensors='tf', max_length=self.max_tokens, truncation=True, padding=True)
            outputs = self.model(**batch_dict)
            last_hidden_state = outputs.last_hidden_state
            self.cache.set(key, last_hidden_state)
        if(comb_strategy == "mean"):
            embeddings = average_pool(last_hidden_state)
        elif (comb_strategy == "sum"):
            embeddings = sum_pool(last_hidden_state)
        elif (comb_strategy == "max"):
            embeddings = max_pool(last_hidden_state)
        elif (comb_strategy == "concat"):
            embeddings = concat_pool(last_hidden_state)
        elif(comb_strategy == "first_token"):
            # Get the embeddings of the [CLS] token (it's the first one)
            # only available for bert based models, e5 will give embedding for "query" token
            embeddings = last_hidden_state[:,0,:]
        r = embeddings.numpy()
        return r