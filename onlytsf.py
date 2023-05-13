from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import tensorflow as tf
import random
import yaml


app = Flask(__name__)

with open("intents.yaml") as file:
    data = yaml.safe_load(file)

lemmatizer = WordNetLemmatizer()

words = []
labels = []
docs_x = []
docs_y = []

# Tokenize and lemmatize words in patterns
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs_x.append([lemmatizer.lemmatize(word) for word in tokens])
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Sort and remove duplicates
words = sorted(set(words))
labels = sorted(set(labels))

# Build a Bag of Words for each pattern
training = []
output = []

out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [lemmatizer.lemmatize(word.lower()) for word in doc]

    for word in words:
        if word in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Convert to numpy array
training = np.array(training)
output = np.array(output)

# Build the neural network
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Train the model
model.fit(training, output, n_epoch=3000, batch_size=400, show_metric=True)
model.save("model.tflearn")

# Load the model
model.load("model.tflearn")
@app.route("/home")
def home():
    return render_template("home.html")
# Respond to user input
@app.route("/")
def index():
    return render_template("index.html")




@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")

    results = model.predict([BagOfWords(user_input, words, lemmatizer)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            responses = intent["responses"]
            break

    response = random.choice(responses)
    return response

def BagOfWords(s, words, lemmatizer):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, word in enumerate(words):
            if word == se:
                bag[i] = 1

    return np.array(bag)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080')
