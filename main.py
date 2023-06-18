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

# Load your data
# Assuming df is your DataFrame and it has columns 'text' and 'label'
df = pd.read_csv('datasets/IMDB_Dataset.csv')
RANDOM_SEED = 42
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
#downsize df to max 3000 rows
df = df.sample(n=3000, random_state=RANDOM_SEED)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define method to create model
def create_model(learning_rate=0.01, neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define method for evaluation
def eval_nn(individual):
    print("evaluating...")
    learning_rate, neurons = individual
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
toolbox.register("attr_learning_rate", np.random.uniform, 0.001, 0.1)
toolbox.register("attr_neurons", np.random.randint, 1, 100)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_learning_rate, toolbox.attr_neurons), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_nn)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[0.001, 1], up=[0.1, 100], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=10)
result = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

best_individual = tools.selBest(pop, 1)[0]
print("Best individual: " + str(best_individual))
