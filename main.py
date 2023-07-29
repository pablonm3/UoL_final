import argparse
import math
import time

from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import pathlib
from configReader import get_config

from custom_deap_tools import my_mutGaussian, my_HallOfFame
from embeddings import EmbeddingGenerator
from textPreprocessor import TextPreprocessor
from utils import save_log
from sklearn.metrics import f1_score

project_path = pathlib.Path(__file__).parent.resolve()

RANDOM_SEED = 42




# Register parameters for GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

GENOTYPE_SPEC = [
    {"name": "learning_rate", "type": "float_range", "bounds": [0.001, 0.1]},
    {"name": "n_layers", "type": "int_range", "bounds": [0, 11]}, # max n layers=10
    {"name": "max_neurons", "type": "int_range", "bounds": [50, 301]}, # max value=300
    {"name": "per_dropout", "type": "float_range", "bounds": [0, 0.2]},
    {"name": "dropout_in_layers", "type": "cat", "options": ["none", "all", "input"]},
    {"name": "epochs", "type": "int_range", "bounds": [1, 11]},# max epochs=10
    {"name": "batch_size", "type": "cat", "options": [1, 2, 4, 8, 16]},
    {"name": "emb_model_name", "type": "cat", "options": ["bert-base-uncased", "roberta-base", "intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2"]},
    {"name": "emb_comb_strategy", "type": "cat", "options": ["mean", "first_token", "sum", "concat", "max"]},
    {"name": "preprop_rem_stopwords", "type": "cat", "options": [True, False]},
    {"name": "preprop_word_normalization", "type": "cat", "options": ["lemmatization", "stemming", None]},
    {"name": "preprop_lowercasing", "type": "cat", "options": [True, False]},
    {"name": "preprop_remove_punctuation", "type": "cat", "options": [True, False]},
    {"name": "preprop_TFIDF_word_removal", "type": "cat", "options": [True, False]},
]

for gene in GENOTYPE_SPEC:
    toolbox.register("attr_" + gene["name"], np.random.uniform, 0, 1)

class GA:
# GA class runner
  def __init__(self, config_name):
        self.config = get_config(config_name, f"{project_path}/config")
        self.fitness_cache = {}
        self.vectorizers_cache = {}
        DS_PATH = self.config["DS_PATH"]
        MAX_SAMPLE_SIZE_DS = self.config["MAX_SAMPLE_SIZE_DS"]
        TEXT_COLUMN = self.config["TEXT_COLUMN"]
        LABEL_COLUMN = self.config["LABEL_COLUMN"]
        # Load your data
        # Assuming df is your DataFrame and it has columns 'text' and 'label'
        df = pd.read_csv(DS_PATH)
        # downsize df to max 3000 rows
        df = df.sample(n=MAX_SAMPLE_SIZE_DS, random_state=RANDOM_SEED)
        # Preprocess text data to convert it into numerical data
        X = df[TEXT_COLUMN].tolist()
        y = df[LABEL_COLUMN]
        # Encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)

        # Train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, encoded_Y, test_size=0.2, random_state=RANDOM_SEED)
        # Convert integers to dummy variables (i.e. one hot encoded)
        self.y_train_ohe = np_utils.to_categorical(self.y_train)
        self.textPreprocessor = TextPreprocessor()

  def sentence_vectorizer(self, model_name, X_train, X_test, comb_strategy):
        # Preprocess text data to convert it into numerical data
        if(model_name in self.vectorizers_cache):
            vectorizer = self.vectorizers_cache[model_name]
        else:
            vectorizer = EmbeddingGenerator(self.config, model_name)
            self.vectorizers_cache[model_name] = vectorizer
        X_train_emb = vectorizer.sentence_embedding(X_train, comb_strategy)
        X_test_emb = vectorizer.sentence_embedding(X_test, comb_strategy)
        return X_train_emb, X_test_emb

  def parse_genotype(self, individual):
    # convert framework gene: between 0 and 1 to fenotype, real value used by my pipeline/model
    r = {}
    FROZEN_PARAMETERS = self.config["FROZEN_PARAMETERS"]
    for i, gene in enumerate(individual):
        gene_name = GENOTYPE_SPEC[i]["name"]
        if(gene_name in FROZEN_PARAMETERS):
            print(f"OVERWRITING PARAMETER {gene_name} FOR TESTING WITH VALUE: {FROZEN_PARAMETERS[gene_name]}")
            r[gene_name] = FROZEN_PARAMETERS[gene_name]
        elif GENOTYPE_SPEC[i]["type"] == "float_range":
            # scale the gene value between 0(inclusive) and 1(exclusive) to a value between both bounds low(inclusive) and high(exlusive)
            r[gene_name] = GENOTYPE_SPEC[i]["bounds"][0] + gene * (
                        GENOTYPE_SPEC[i]["bounds"][1] - GENOTYPE_SPEC[i]["bounds"][0])
        elif GENOTYPE_SPEC[i]["type"] == "int_range":
            # scale the gene value between 0(inclusive) and 1(exclusive) to a value between both bounds low(inclusive) and high(exlusive)
            r[gene_name] = int(GENOTYPE_SPEC[i]["bounds"][0] + gene * (
                        GENOTYPE_SPEC[i]["bounds"][1] - GENOTYPE_SPEC[i]["bounds"][0]) )
        elif GENOTYPE_SPEC[i]["type"] == "cat":
            options = GENOTYPE_SPEC[i]["options"]
            index = math.floor(gene * len(options))
            r[gene_name] = options[index]

    return r
  def create_model(self, input_dim, learning_rate, n_layers, max_neurons, dropout_in_layers, per_dropout):
      # Define method to create model
      NEURONS_CHANGE_FACTOR = self.config["NEURONS_CHANGE_FACTOR"]
      model = Sequential()
      neurons = max_neurons
      model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
      if(dropout_in_layers in ["all", "input"]):
          model.add(Dropout(per_dropout))
      for i in range(n_layers):
          neurons = int(neurons * NEURONS_CHANGE_FACTOR)  # reduce neurons by a fixed rate each layer
          model.add(Dense(neurons, activation='relu'))
          if (dropout_in_layers in ["all"]):
              model.add(Dropout(per_dropout))
      model.add(Dense(self.y_train_ohe.shape[1], activation='softmax'))
      optimizer = Adam(learning_rate=learning_rate)
      model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
      return model

  def eval_nn(self, individual):
      # method for evaluation
      props = self.parse_genotype(individual)
      print("raw values individual: ", individual)
      print("evaluating individual with props: ", props)
      if (tuple(individual) in self.fitness_cache):
          print("using cached fitness value for key: ", tuple(individual) )
          return self.fitness_cache[tuple(individual)],
      learning_rate = props["learning_rate"]
      n_layers = props["n_layers"]
      max_neurons = props["max_neurons"]
      dropout_in_layers = props["dropout_in_layers"]
      per_dropout = props["per_dropout"]
      epochs = props["epochs"]
      batch_size = props["batch_size"]
      emb_model_name = props["emb_model_name"]
      emb_comb_strategy = props["emb_comb_strategy"]
      preprop_rem_stopwords = props["preprop_rem_stopwords"]
      preprop_word_normalization = props["preprop_word_normalization"]
      preprop_lowercasing = props["preprop_lowercasing"]
      preprop_remove_punctuation = props["preprop_remove_punctuation"]
      preprop_TFIDF_word_removal = props["preprop_TFIDF_word_removal"]
      X_train_preproped = self.textPreprocessor.run(self.X_train, preprop_rem_stopwords, preprop_word_normalization, preprop_lowercasing, preprop_remove_punctuation, preprop_TFIDF_word_removal)
      X_test_preproped = self.textPreprocessor.run(self.X_test, preprop_rem_stopwords, preprop_word_normalization, preprop_lowercasing, preprop_remove_punctuation, preprop_TFIDF_word_removal)
      X_train_emb, X_test_emb = self.sentence_vectorizer(emb_model_name, X_train_preproped, X_test_preproped, emb_comb_strategy)
      model = KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=10, verbose=0)
      model = model.set_params(input_dim=X_train_emb[0].shape[0], learning_rate=learning_rate, n_layers=n_layers,
                               max_neurons=max_neurons, dropout_in_layers=dropout_in_layers, per_dropout=per_dropout)
      X_train_emb = tf.stack(X_train_emb)
      X_test_emb = tf.stack(X_test_emb)
      y_train_ohe_stacked = tf.stack(self.y_train_ohe)
      model.fit(X_train_emb, y_train_ohe_stacked, batch_size=batch_size, epochs=epochs)
      predictions = model.predict(X_test_emb)
      f1_macro_score = f1_score(self.y_test, predictions, average='macro')
      print("f1_macro_score: ", f1_macro_score)
      self.fitness_cache[tuple(individual)] = f1_macro_score
      return f1_macro_score,
  def run(self):
    PROB_MUTATION = self.config["PROB_MUTATION"]
    N_POPULATION = self.config["N_POPULATION"]
    GENERATIONS = self.config["GENERATIONS"]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_learning_rate, toolbox.attr_n_layers, toolbox.attr_max_neurons, toolbox.attr_dropout_in_layers, toolbox.attr_per_dropout, toolbox.attr_epochs, toolbox.attr_batch_size, toolbox.attr_emb_model_name,
                      toolbox.attr_emb_comb_strategy, toolbox.attr_preprop_rem_stopwords, toolbox.attr_preprop_word_normalization, toolbox.attr_preprop_lowercasing,
                      toolbox.attr_preprop_remove_punctuation, toolbox.attr_preprop_TFIDF_word_removal), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", self.eval_nn)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", my_mutGaussian, mu=0, sigma=0.1, indpb=PROB_MUTATION)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=N_POPULATION)
    hof = my_HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    start_epoch_time = int(time.time())
    stats.register("time", lambda _: int(time.time()) - start_epoch_time)
    result, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=PROB_MUTATION, ngen=GENERATIONS, stats=stats,
                                      halloffame=hof, verbose=True)

    best_individual = hof[0]
    print("Best individual: ", self.parse_genotype(best_individual))
    print("best individual fitness: " + str(best_individual.fitness))
    print(log)
    save_log(log, self.config["LOG_FILE_DIR"])
    print("surviving population:")
    print(pop)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='Config file to use')
    args = parser.parse_args()
    ga = GA(args.config)
    ga.run()