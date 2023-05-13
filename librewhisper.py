from flask import Flask, render_template, request,jsonify
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import tensorflow as tf
import random
import yaml
import requests
import libretranslatepy
#transcription
import whisper
import sounddevice as sd
import soundfile as sf
import subprocess

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
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
# Load the model
model.load("model.tflearn")
try:
    cmd = ['libretranslate']
    subprocess.Popen(cmd)
except print(0):
    pass


@app.route("/home")
def home():
    return render_template("home.html")
# Respond to user input
@app.route("/")
def index():
    return render_template("index.html")



@app.route("/record", methods=["POST"])
def record():
    model = whisper.load_model("tiny.en") # Load English model
    audio_file = "recorded_audio.wav" # File to save recorded audio
    seconds = 5 # Recording duration in seconds

    # Record audio from user
    fs = 16000  # Sample rate
    print("Recording...")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    sf.write(audio_file, myrecording, fs)  # Save as WAV file

    # Transcribe audio using Whisper
    result = model.transcribe(audio_file)
    transcription = result["text"]
    print("Transcription:", transcription)

    # Send transcription to /get endpoint
    response = requests.get("http://localhost:5000/get", params={"msg": transcription})
    
    return jsonify({"transcription": transcription})

#botresponse
@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")
    # detect language and translate to English
    response = requests.post("http://localhost:5000/detect", json={"q": user_input}).json()
    lang = response[0]["language"]
    if lang != "en":
        response = requests.post("http://localhost:5000/translate", json={"q": user_input, "source": lang, "target": "en"}).json()
        user_input = response["translatedText"]
    # get model response
    results = model.predict([BagOfWords(user_input, words, lemmatizer)])
    results_index = np.argmax(results)
    tag = labels[results_index]
    for intent in data["intents"]:
        if intent["tag"] == tag:
            responses = intent["responses"]
            break
    response = random.choice(responses)
    # translate model response back to the original language
    if lang != "en":
        response = requests.post("http://localhost:5000/translate", json={"q": response, "source": "en", "target": lang}).json()["translatedText"]
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
