import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


def sentence_embedding(sentences):
    MAX_TOKENS = 512 #FIXME this depends on the model
    # Load pre-trained model tokenizer (BERT-base uncased)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained("bert-base-uncased")

    sentence_embeddings = []

    for sentence in sentences:
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        #encoded_input = tokenizer(text, return_tensors='tf')
        input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='tf')
        input_ids = input_ids[0][:MAX_TOKENS] # truncate to max tokens supported by model
        input_ids = tf.expand_dims(input_ids, 0)  # Batch size 1
        outputs = model(input_ids)
        # Get the embeddings of the [CLS] token (it's the first one)
        cls_embedding = outputs.last_hidden_state[0][0]
        sentence_embeddings.append(cls_embedding)

    return sentence_embeddings
