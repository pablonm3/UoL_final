from deap import base, creator, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
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

from custom_deap_tools import my_mutGaussian, my_HallOfFame
from embeddings import sentence_embedding

# Load your data
# Assuming df is your DataFrame and it has columns 'text' and 'label'
DS_PATH = 'datasets/IMDB_Dataset.csv'
RANDOM_SEED = 42
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
GENERATIONS = 15
N_POPULATION = 3
PROB_MUTATION = 0.2
MAX_SAMPLE_SIZE_DS = 150
NEURONS_CHANGE_FACTOR = 0.8 # reduce neurons by 20% each layer
df = pd.read_csv(DS_PATH)
#downsize df to max 3000 rows
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Define method to create model
def create_model(input_dim, learning_rate=0.01, n_layers=0, max_neurons=100):
    model = Sequential()
    neurons = max_neurons
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    for i in range(n_layers):
        neurons = int(neurons * NEURONS_CHANGE_FACTOR) # reduce neurons by a fixed rate each layer
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def parse_genotype(individual):
    # convert framework gene: between 0 and 1 to fenotype, real value used by my pipeline/model
    r = {}
    for i, gene in enumerate(individual):
        if GENOTYPE_SPEC[i]["type"] == "float_range":
            r[GENOTYPE_SPEC[i]["name"]] = GENOTYPE_SPEC[i]["bounds"][0] + gene * (
                        GENOTYPE_SPEC[i]["bounds"][1] - GENOTYPE_SPEC[i]["bounds"][0])
        elif GENOTYPE_SPEC[i]["type"] == "int_range":
            r[GENOTYPE_SPEC[i]["name"]] = int(GENOTYPE_SPEC[i]["bounds"][0] + gene * (
                        GENOTYPE_SPEC[i]["bounds"][1] - GENOTYPE_SPEC[i]["bounds"][0]))

    return r

def sentence_vectorizer(X_train, X_test):
    # Preprocess text data to convert it into numerical data
    X_train_emb = sentence_embedding(X_train)
    X_test_emb = sentence_embedding(X_test)
    return X_train_emb, X_test_emb

fitness_cache = {}
# Define method for evaluation
def eval_nn(individual):
    props = parse_genotype(individual)
    print("evaluating individual with props: ", props)
    if(tuple(individual) in fitness_cache):
        print("using cached fitness value")
        return fitness_cache[tuple(individual)],
    learning_rate = props["learning_rate"]
    n_layers = props["n_layers"]
    max_neurons = props["max_neurons"]
    X_train_emb, X_test_emb = sentence_vectorizer(X_train, X_test)
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=0)
    model = model.set_params(input_dim=X_train_emb[0].shape[0], learning_rate=learning_rate, n_layers=n_layers, max_neurons=max_neurons)
    X_train_emb = tf.stack(X_train_emb)
    X_test_emb = tf.stack(X_test_emb)
    y_train_stacked = tf.stack(y_train)
    y_test_stacked = tf.stack(y_test)
    model.fit(X_train_emb, y_train_stacked)
    accuracy = model.score(X_test_emb, y_test_stacked)
    print("accuracy: ", accuracy)
    fitness_cache[tuple(individual)] = accuracy
    return accuracy,

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

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_learning_rate, toolbox.attr_n_layers, toolbox.attr_max_neurons), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_nn)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", my_mutGaussian,  mu=0, sigma=0.1, indpb=PROB_MUTATION)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=N_POPULATION)
hof = my_HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
result, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=PROB_MUTATION, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)

best_individual = hof[0]
print("Best individual: ", parse_genotype(best_individual))
print("best individual fitness: " + str(best_individual.fitness))
print(log)
print("surviving population:")
print(pop)
