from deap import base, creator, tools, algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
import pandas as pd

from custom_deap_tools import my_mutGaussian, my_HallOfFame

# Load your data
# Assuming df is your DataFrame and it has columns 'text' and 'label'
DS_PATH = 'datasets/IMDB_Dataset.csv'
RANDOM_SEED = 42
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
GENERATIONS = 10
N_POPULATION = 3
PROB_MUTATION = 0.2
MAX_SAMPLE_SIZE_DS = 3000
df = pd.read_csv(DS_PATH)
#downsize df to max 3000 rows
df = df.sample(n=MAX_SAMPLE_SIZE_DS, random_state=RANDOM_SEED)
# Preprocess text data to convert it into numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[TEXT_COLUMN]).toarray()
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
def create_model(learning_rate=0.01, neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
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


# Define method for evaluation
def eval_nn(individual):
    print("evaluating...")
    props = parse_genotype(individual)
    learning_rate = props["learning_rate"]
    neurons = props["neurons"]
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=0)
    model = model.set_params(learning_rate=learning_rate, neurons=int(neurons))
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print("accuracy: ", accuracy)
    return accuracy,

# Register parameters for GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

GENOTYPE_SPEC = [
    {"name": "learning_rate", "type": "float_range", "bounds": [0.001, 0.1]},
    {"name": "neurons", "type": "int_range", "bounds": [1, 100]}
]

for gene in GENOTYPE_SPEC:
    if gene["type"] == "float_range":
        toolbox.register("attr_" + gene["name"], np.random.uniform, 0, 1)
    elif gene["type"] == "int_range":
        toolbox.register("attr_" + gene["name"], np.random.uniform, 0, 1)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_learning_rate, toolbox.attr_neurons), n=1)
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
print("Best individual: " + str(best_individual))
print("best individual fitness: " + str(best_individual.fitness))
print(log)
print("surviving population:")
print(pop)
