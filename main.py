import argparse

from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import pathlib
from configReader import get_config

from custom_deap_tools import my_mutGaussian, my_HallOfFame
from embeddings import sentence_embedding
from utils import save_log

project_path = pathlib.Path(__file__).parent.resolve()

RANDOM_SEED = 42



def sentence_vectorizer(X_train, X_test):
    # Preprocess text data to convert it into numerical data
    X_train_emb = sentence_embedding(X_train)
    X_test_emb = sentence_embedding(X_test)
    return X_train_emb, X_test_emb

# Register parameters for GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

GENOTYPE_SPEC = [
    {"name": "learning_rate", "type": "float_range", "bounds": [0.001, 0.1]},
    {"name": "n_layers", "type": "int_range", "bounds": [0, 10]},
    {"name": "max_neurons", "type": "int_range", "bounds": [50, 300]}
]

for gene in GENOTYPE_SPEC:
    if gene["type"] == "float_range":
        toolbox.register("attr_" + gene["name"], np.random.uniform, 0, 1)
    elif gene["type"] == "int_range":
        toolbox.register("attr_" + gene["name"], np.random.uniform, 0, 1)

# GA class runner
class GA:
  def __init__(self, config_name):
        self.config = get_config(config_name, f"{project_path}/config")
        self.fitness_cache = {}
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
        X = np.array(df[TEXT_COLUMN])
        y = df[LABEL_COLUMN]
        # Encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        # Convert integers to dummy variables (i.e. one hot encoded)
        y = np_utils.to_categorical(encoded_Y)

        # Train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

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
            r[gene_name] = GENOTYPE_SPEC[i]["bounds"][0] + gene * (
                        GENOTYPE_SPEC[i]["bounds"][1] - GENOTYPE_SPEC[i]["bounds"][0])
        elif GENOTYPE_SPEC[i]["type"] == "int_range":
            r[gene_name] = int(GENOTYPE_SPEC[i]["bounds"][0] + gene * (
                        GENOTYPE_SPEC[i]["bounds"][1] - GENOTYPE_SPEC[i]["bounds"][0]))

    return r
  def create_model(self, input_dim, learning_rate=0.01, n_layers=0, max_neurons=100):
      # Define method to create model
      NEURONS_CHANGE_FACTOR = self.config["NEURONS_CHANGE_FACTOR"]
      model = Sequential()
      neurons = max_neurons
      model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
      for i in range(n_layers):
          neurons = int(neurons * NEURONS_CHANGE_FACTOR)  # reduce neurons by a fixed rate each layer
          model.add(Dense(neurons, activation='relu'))
      model.add(Dense(self.y_train.shape[1], activation='softmax'))
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
      X_train_emb, X_test_emb = sentence_vectorizer(self.X_train, self.X_test)
      model = KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=10, verbose=0)
      model = model.set_params(input_dim=X_train_emb[0].shape[0], learning_rate=learning_rate, n_layers=n_layers,
                               max_neurons=max_neurons)
      X_train_emb = tf.stack(X_train_emb)
      X_test_emb = tf.stack(X_test_emb)
      y_train_stacked = tf.stack(self.y_train)
      y_test_stacked = tf.stack(self.y_test)
      model.fit(X_train_emb, y_train_stacked)
      accuracy = model.score(X_test_emb, y_test_stacked)
      print("accuracy: ", accuracy)
      self.fitness_cache[tuple(individual)] = accuracy
      return accuracy,
  def run(self):
    PROB_MUTATION = self.config["PROB_MUTATION"]
    N_POPULATION = self.config["N_POPULATION"]
    GENERATIONS = self.config["GENERATIONS"]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_learning_rate, toolbox.attr_n_layers, toolbox.attr_max_neurons), n=1)
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