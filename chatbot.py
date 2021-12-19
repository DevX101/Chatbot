import aiml
import wikipedia
import pandas as pd
import numpy as np
import sys

import speech_recognition as sr
import pyttsx3
from googlesearch import search

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver, ResolutionProverCommand
from simpful import *
from nltk.sentiment import SentimentIntensityAnalyzer

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="chatbot.xml")

read_expr = Expression.fromstring
kb = []
data = pd.read_csv('sampleKB.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]


"""
Topic: 
1) Galaxies & Nebula
2) Jurassic Period 
"""


def main():

    # respond("Welcome! Pleasure to meet you. \nWhat shall I call you?")
    # name = input()
    # respond("Great! My topic of interest are galaxies and nebulae. \nAsk me anything and I will do my best to answer.")

    while True:
        try:
            """Manual input"""
            name = "Aditya S"  # r

            user_input = input(name + ": ")
            response = kernel.respond(user_input)

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
            respond("I did not get that, please try again.")
        elif cmd == 9:
            return None
        elif cmd == 1:
            try:
                answer = compute_similarity(user_input)
                respond(answer)
            except:
                # Input that is copy/pasted will cause errors.
                respond("Sorry, I do not know that. Please be more specific!")

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
            respond("Here are a few sources you could look at for more information.")

            # Retrieve top 10 URLs.
            for j in search(params[1], tld="co.uk", num=10, stop=10, pause=2):
                print(j)

        elif cmd == 31:  # if input pattern is "I know that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')
            not_expr = read_expr('-' + str(expr))

            # Check if expression is not in contradiction with KB.
            true_prover = ResolutionProverCommand(expr, kb).prove()
            false_prover = ResolutionProverCommand(not_expr, kb).prove()
            condition1 = true_prover and not false_prover
            condition2 = not true_prover and not false_prover

            if condition1 or condition2:
                kb.append(expr)
                statement = "OK, I will remember that", object, "is", subject
                respond(statement)
            else:
                respond("Sorry this contradicts with what I know!")

        elif cmd == 32:  # if the input pattern is "check that * is *"
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')
            answer = ResolutionProver().prove(expr, kb, verbose=True)
            if answer:
                respond('Correct.')
            else:
                not_expr = read_expr('-' + str(expr))
                true_prover = ResolutionProverCommand(expr, kb).prove()
                false_prover = ResolutionProverCommand(not_expr, kb).prove()

                # Provide definitive answer given the condition of the expression using resolution.
                if not true_prover and false_prover:
                    respond("Incorrect")
                else:
                    respond("Sorry I don't know")
    else:
        respond(response)


def read_csv(path):

    # Extract questions & answers from file.
    column_names = ["Questions", "Answers"]
    df = pd.read_csv(path, names=column_names)
    questions = df.Questions.tolist()
    answers = df.Answers.tolist()

    return questions, answers


def compute_similarity(query):

    questions, answers = read_csv("sampleQA.csv")
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
    for ans, score in result:
        if score == np.max(similarity):
            answer = ans

    return answer


def fuzzy_system(var1, var2):
    """
    Mamdani FIS (fuzzy inference system) -> more suitable for human inputs than Sugeno FIS (numerical calculation).

    - Life expectancy
    - Death of a star about to be nebula
    - Estimate when each star will go supernova
    """

    # Create a fuzzy system object
    FS = FuzzySystem()

    # Define fuzzy sets and linguistic variables
    S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term="good")
    S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term="excellent")
    FS.add_linguistic_variable("v1", LinguisticVariable([S_1, S_2, S_3], concept="Service quality",
                                                        universe_of_discourse=[0, 10]))

    F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="rancid")
    F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term="delicious")
    FS.add_linguistic_variable("v2", LinguisticVariable([F_1, F_2], concept="Food quality",
                                                        universe_of_discourse=[0, 10]))

    # Define output fuzzy sets and linguistic variable
    T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term="low")
    T_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=20), term="average")
    T_3 = FuzzySet(function=Trapezoidal_MF(a=10, b=20, c=25, d=25), term="high")
    FS.add_linguistic_variable("result", LinguisticVariable([T_1, T_2, T_3],
                                                            universe_of_discourse=[0, 25]))

    # Produce graphs of fuzzy sets
    FS.produce_figure(outputfile='output.pdf')

    # Define fuzzy rules
    R1 = "IF (v1 IS poor) OR (v2 IS rancid) THEN (result IS low)"
    R2 = "IF (v1 IS good) THEN (result IS average)"
    R3 = "IF (v1 IS excellent) OR (v2 IS delicious) THEN (result IS high)"
    FS.add_rules([R1, R2, R3])

    # Set antecedents values
    FS.set_variable("v1", var1)
    FS.set_variable("v2", var2)

    # Perform Mamdani inference
    output = FS.Mamdani_inference()

    return output['result']


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

                # Wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level.
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # Listen for the user's input.
                audio2 = r.listen(source2)

                # Recognize audio.
                response = r.recognize_google(audio2)
                response = response.lower()

                return response

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            respond("Please speak up. I can't hear you.")


if __name__ == "__main__":
    main()

"""
Tasks:
- analyse grammar
- google search if more info is requested
"""
