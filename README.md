# TenseWhisper
 This is a Flask chatbot that can translate between languages, generate text, and answer questions. It is trained on a yaml dataset which you can change to fit your needs, and it can respond to user input in a variety of ways.
Features
    very beautiful yet simple interface built with just html css and javascript
    interactive buttons
    Can translate between a variety of languages
    Can generate text
    Can answer questions
    uses completely opensource tools so translation is easier and free
    Is easy to use



EXAMPLE
![TENSE](https://github.com/danpizzy/tensewhisper/assets/53155066/cf93f41f-c90e-41da-bcbb-cb50cd20648f)
![TENSE](https://im5.ezgif.com/tmp/ezgif-5-4a3ae108b5.gif)

https://im5.ezgif.com/tmp/ezgif-5-4a3ae108b5.gif

How to use

    Clone the repository to your local machine.
    Install the required dependencies.
    Run the chatbot by using the librewhisper.py file or the onlytsf.py file the libre whisper file incorporates bot libre translate and speech recognition but is a bit slower  
    you can upload your own dataset but make sure it follows the same yaml format as the default intents.yaml

Contributing

Contributions are welcome! Please feel free to open issues or pull requests.
License

YAML Dataset

The YAML dataset is a file that contains the training data for the chatbot. The dataset should be formatted as follows:
Code snippet

intents:
  - Tag: greet
    patterns:
      - "hello"
      - "hi"
      - "good morning"
      - "good afternoon"
      - "good evening"
    responses:
      - "Hello!"
      - "Hi there!"
      - "Good morning!"
      - "Good afternoon!"
      - "Good evening!"

  - Tag: goodbye
    patterns:
      - "bye"
      - "see you later"
      - "goodbye"
    responses:
      - "Goodbye!"
      - "See you later!"
      - "Good bye!"

  - Tag: ask_name
    patterns:
      - "what is your name?"
      - "who are you?"
    responses:
      - "My name is Tens Whisper."
      - "I am a chatbot."

  - Tag: tell_joke
    patterns:
      - "tell me a joke"
      - "make me laugh"
    responses:
      - "What do you call a fish with no eyes? Fsh!"
      - "Why did the scarecrow win an award? Because he was outstanding in his field!"
      - "What do you call a lazy kangaroo? A pouch potato!"

Training the Chatbot

To train the chatbot just rerun the python file

This  will train the chatbot on the YAML dataset. The training process may take a few minutes to complete.
Running the Chatbot


This will start the chatbot on your local machine. The chatbot will be available at http://localhost:8080.

