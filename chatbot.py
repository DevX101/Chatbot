import aiml
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import speech_recognition as sr
import pyttsx3
import wikipedia
import webbrowser
import requests
from googlesearch import search

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver, ResolutionProverCommand
from simpful import *
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

from tensorflow import keras
import cv2
from detector import Detector
from azure_api import azure_image_classification, azure_image_analysis
import warnings

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="chatbot.xml")

read_expr = Expression.fromstring
kb = []
data = pd.read_csv('Files/kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

# Load image classification model
warnings.filterwarnings("ignore")
model = keras.models.load_model("Weights/model.h5")
labels = ["bear", "deer", "squirrel"]
correct_labels = dict()
path_key = ""

# Load object detection model
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
class_path = "Files/coco.names"
detect = Detector()
detect.readClasses(class_path)
detect.downloadModel(model_url)
detect.loadModel()

# Azure API
cog_key = 'YOUR_COG_KEY'
cog_endpoint = 'YOUR_COG_ENDPOINT'

def main():

    respond("\nWelcome! Pleasure to meet you. \nMy topic of interest is the Tundra Ecosystem. "
            "\nAsk me a question and I will do my best to answer.")

    while True:
        try:
            """Manual input"""
            user_input = input("> ")

            # Strip punctuation
            tokenizer = RegexpTokenizer(r'\w+')
            input_list = tokenizer.tokenize(user_input)
            user_input = ' '.join(input_list)

            # Correct spelling
            correct_input = user_input.lower()
            spell_check = TextBlob(correct_input)
            correct_input = spell_check.correct()

            # Fix incorrect spell check
            ignore_words = [("tunica", "tundra"), ("consumed", "consumes")]
            for word in ignore_words:
                correct_input = correct_input.replace(word[0], word[1])

            response = kernel.respond(correct_input)

            """Voice input"""
            if response == 'voice':
                respond("Go ahead! I'm listening...")

                vc = None
                while vc != 'stop':
                    print('🔊')
                    vc = voice_command()
                    response = kernel.respond(vc)
                    chat(user_input, response)
                respond("I will stop listening! Let me know if you would like to speak again.")

            chat(user_input, response)

        except (KeyboardInterrupt, EOFError) as e:
            respond("Bye!")
            break


def chat(user_input, response):

    if response[0] == '#':
        params = response[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            respond(params[1])
            sys.exit()
        elif cmd == 99:
            respond("I did not get that, please try again. "
                    "\nMake sure your question is related to my topic of interest.")
        elif cmd == 9:
            return None
        elif cmd == 1:
            try:
                answer = compute_similarity(str(user_input))
                respond(answer)
            except:
                # Input that is copy/pasted will result in exception!
                respond("Sorry, I have not been programmed to know that yet. Please be more specific!")

        elif cmd == 2:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                respond(wSummary)
            except:
                respond("I could not find a valid definition.")

        elif cmd == 3:
            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(params[1])

            # Distinguish whether statement is positive or negative.
            if scores['pos'] > scores['neg']:
                print('😊')
                respond("It seems you are in good spirits!")
            else:
                print('😞')
                respond("It seems you don't feel well... \nHopefully our conversation brightens up your mood.")

        elif cmd == 4:

            query = "what is " + params[1]
            r = requests.get("https://api.duckduckgo.com",
                             params={
                                 "q": query,
                                 "format": "json"
                             })
            data = r.json()
            answer = data["Abstract"]

            if answer != "":
                respond(answer)
            else:
                respond("Here is more information you could look at.")
                url = "https://www.google.com/search?q=" + params[1]
                webbrowser.open_new_tab(url)

        elif cmd == 5:
            df = pd.read_csv("Files/fuzzyData.csv", names=["Animal", "Population", "Size"])
            found = False
            index = 0
            for i in df.index:
                if df['Animal'][i] == params[1]:
                    found = True
                    index = i
                    break

            if found is True:
                tfr = fuzzy_system(df['Population'][index]/pow(10, 6), df['Size'][index])
                tfr = round(tfr, 2)
                respond("The fertility rate for an adult " + str(params[1]) + " is " + str(tfr) + " births per year.")
            else:
                print("No matches found on the database.")

        elif cmd == 6:
            respond("Here is a map showing the regions of the Tundra ecosystem.")
            tundra = plt.imread("Files/tundraMap.png")
            plt.imshow(tundra)
            plt.show()

        elif cmd == 7:
            num, query = params[1].split(',')
            respond("Here are the top " + num + " search results for your query.")
            n = int(num)
            for result in search(query, tld="co.uk", num=n, stop=n, pause=1):
                print(result)

        elif cmd == 8:
            print("Interesting opinion. Good to hear your insights on this topic.")
            tok_word = word_tokenize(params[1])
            tok_sent = sent_tokenize(params[1])
            stop_words = set(stopwords.words("english"))
            ps = PorterStemmer()
            lem = WordNetLemmatizer()

            filtered_words = [w for w in tok_word if w not in stop_words]
            stemmed_words = [ps.stem(w) for w in filtered_words]
            pos = pos_tag(stemmed_words)
            print(stemmed_words)
            print(pos)

        elif cmd in (30, 31, 32):
            logic_chat(user_input)

        elif cmd == 10:
            try:
                # Check if label has been corrected
                if params[1] in correct_labels:
                    respond("This image is a " + correct_labels[params[1]])
                else:
                    # Image Classification
                    path = os.path.join('Files/Images/', params[1])
                    result = model.predict_classes([prepare(path, 200)])[0]
                    respond("This image is a " + labels[result])

                    img = cv2.imread(path)
                    cv2.imshow("Result", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Azure Image Classification
                    prediction = azure_image_classification(params[1])
                    print("[Azure] This image is a " + prediction)
            except:
                respond("I could not find that image.")

        elif cmd == 11:
            # Video Classification
            try:
                path = os.path.join('Files/Images/', params[1])
                cap = cv2.VideoCapture(path)

                classify_vid = []
                while cap.isOpened():
                    (grabbed, frame) = cap.read()
                    cv2.imshow('Frame', frame)
                    result = model.predict_classes([prepare(frame, 200)])[0]

                    if result not in classify_vid:
                        classify_vid.append(result)

                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

                # Show the most common label classified from the video
                # conf = max(set(classify_vid), key=classify_vid.count)
                # print("This video contains a", labels[int(conf)])

                # Show labels classified from the video
                if classify_vid is not None:
                    respond("This video contains...")
                    num = 0
                    for index in classify_vid:
                        print(num, ") ", labels[index])

            except:
                respond("I could not find that video.")

        elif cmd == 12:
            # Correct image classification
            correct_labels[path_key] = params[1]
            respond("Okay, I will remember that.")

        elif cmd in (13, 14, 15):
            obj_detect(cmd, params[1])

    else:
        respond(response)


def compute_similarity(query):

    # Extract questions & answers from file.
    column_names = ["Questions", "Answers"]
    df = pd.read_csv("Files/QA.csv", names=column_names)
    questions = df.Questions.tolist()
    answers = df.Answers.tolist()

    if query in questions:
        return answers[questions.index(query)]
    else:
        questions.append(query)

        """tf-idf"""
        vectorizer = TfidfVectorizer(stop_words="english")
        question_tfidf = vectorizer.fit_transform(questions)
        df = pd.DataFrame(question_tfidf.toarray(), index=questions, columns=vectorizer.get_feature_names())

        """cosine similarity"""
        similarity = []
        for q in questions:
            score = cosine_similarity([df.loc[query].values], [df.loc[q].values])
            similarity.append(score[0][0])

        similarity.pop()
        result = zip(answers, similarity)

        answer = ""
        # Set threshold for acceptable similarity
        if np.max(similarity) < 0.2:
            respond("Here is what I have found.")
            url = "https://www.google.com/search?q=" + query
            webbrowser.open_new_tab(url)
        else:
            for ans, score in result:
                if score == np.max(similarity):
                    answer = ans

        return answer


def logic_chat(user_input):

    response = kernel.respond(user_input)
    params = response[1:].split('$')
    cmd = int(params[0])

    if cmd == 30:  # Add to KB
        obj, subject = params[1].split(' is ')
        obj, subject = obj.replace(' ', '_'), subject.replace(' ', '_')
        expr = read_expr(subject + '(' + obj + ')')
        not_expr = read_expr('-' + str(expr))

        # Check if expression is not in contradiction with KB.
        true_prover = ResolutionProverCommand(expr, kb).prove()
        false_prover = ResolutionProverCommand(not_expr, kb).prove()
        condition1 = true_prover and not false_prover
        condition2 = not true_prover and not false_prover

        if condition1 or condition2:
            kb.append(expr)
            statement = "OK, I will remember that " + obj + " is " + subject
            respond(statement)
        else:
            respond("Sorry this contradicts with what I know!")

    elif cmd == 31:  # Check KB
        obj, subject = params[1].split(' is ')
        obj, subject = obj.replace(' ', '_'), subject.replace(' ', '_')
        expr = read_expr(subject + '(' + obj + ')')
        infer_kb(expr)

    elif cmd == 32:  # Check KB with multi-valued predicates
        predator, prey = params[1].split(' consumes ')
        predator, prey = predator.replace(' ', '_'), prey.replace(' ', '_')
        expr = read_expr("consumes(" + predator + ',' + prey + ')')
        infer_kb(expr)

def infer_kb(expr):

    answer = ResolutionProver().prove(expr, kb, verbose=False)
    if answer:
        respond("Correct.")
    else:
        # Provide definitive answer given the condition of the expression using resolution.
        not_expr = read_expr('-' + str(expr))
        true_prover = ResolutionProverCommand(expr, kb).prove()
        false_prover = ResolutionProverCommand(not_expr, kb).prove()

        if not true_prover and false_prover:
            respond("Incorrect.")
        else:
            respond("Sorry, I have not been programmed to know that yet.")


def fuzzy_system(var1, var2):
    """
    Mamdani FIS
    """
    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    P_1 = FuzzySet(function=Triangular_MF(a=0, b=0.05, c=0.1), term="endangered")
    P_2 = FuzzySet(function=Trapezoidal_MF(a=0.05, b=0.1, c=0.5, d=1), term="normal")
    P_3 = FuzzySet(function=Triangular_MF(a=0.5, b=10, c=10), term="dense")
    FS.add_linguistic_variable("population", LinguisticVariable([P_1, P_2, P_3], concept="Population (millions)",
                                                        universe_of_discourse=[0, 5]))

    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=50), term="small")
    S_2 = FuzzySet(function=Triangular_MF(a=50, b=100, c=500), term="medium")
    S_3 = FuzzySet(function=Triangular_MF(a=100, b=1000, c=1000), term="large")
    FS.add_linguistic_variable("size", LinguisticVariable([S_1, S_2, S_3], concept="Size (kilograms)",
                                                        universe_of_discourse=[0, 1000]))

    # Define output fuzzy sets and linguistic variable
    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=1), term="low")
    F_2 = FuzzySet(function=Triangular_MF(a=0.5, b=1.5, c=2.5), term="average")
    F_3 = FuzzySet(function=Trapezoidal_MF(a=1, b=2, c=5, d=5), term="high")
    FS.add_linguistic_variable("fertility", LinguisticVariable([F_1, F_2, F_3], concept="Fertility Rate (births per animal per year)",
                                                            universe_of_discourse=[0, 5]))

    # Define fuzzy rules
    R1 = "IF (population IS endangered) THEN (fertility IS low)"
    R2 = "IF (size IS medium) OR (population IS normal) THEN (fertility IS average)"
    R3 = "IF (size IS small) OR (size IS large) AND (population IS dense) THEN (fertility IS high)"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("population", var1)
    FS.set_variable("size", var2)

    # Perform Mamdani inference
    output = FS.Mamdani_inference()

    return output['fertility']


def respond(command):
    # Initialize the engine
    print(command)
    voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-GB_HAZEL_11.0"
    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.say(command)
    engine.runAndWait()


def voice_command():
    r = sr.Recognizer()
    response = None
    while response is None:
        try:
            # Microphone as input source.
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                response = r.recognize_google(audio2)
                response = response.lower()

                return response

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            respond("Please speak up. I can't hear you.")


def prepare(filepath, IMG_SIZE):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, -1)

def obj_detect(cmd, path):
    if cmd == 13:
        detect.predictImage(path)
        label_count = detect.getLabels()

        respond("The image contains...")
        for label in label_count:
            observation = str(label_count[label]) + ' ' + label
            respond(observation)
    elif cmd == 14:
        detect.predictVideo(path)
    elif cmd == 15:
        detect.predictVideo(0)

if __name__ == "__main__":
    main()

"""
Azure:
- voice/face recognition 
- text translation (NLP)
- OCR (CV)
- chatbot web service
"""
