import aiml
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as img

import speech_recognition as sr
import pyttsx3
import wikipedia
import webbrowser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver, ResolutionProverCommand
from simpful import *
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="chatbot.xml")

read_expr = Expression.fromstring
kb = []
data = pd.read_csv('Files/kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]


def main():

    # respond("Welcome! Pleasure to meet you. \nMy topic of interest is the Tundra Ecosystem. "
    #         "\nAsk me a question and I will do my best to answer.")

    while True:
        try:
            """Manual input"""
            user_input = input("> ")
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
            print("I did not get that, please try again. "
                    "\nMake sure your question is related to my topic of interest.")
        elif cmd == 9:
            return None
        elif cmd == 1:
            try:
                answer = compute_similarity(user_input)
                print(answer)
            except:
                # Input that is copy/pasted will result in exception!
                print("Sorry, I have not been programmed to know that yet. Please be more specific!")

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
                respond("The fertility rate for an adult " + str(params[1]) + " is " + str(tfr) + " births per year")
            else:
                print("No matches found on the database.")

        elif cmd == 6:
            respond("Here is a map showing the regions of the Tundra ecosystem.")
            tundra = img.imread("Files/tundraMap.png")
            plt.imshow(tundra)
            plt.show()

        elif cmd == 7:
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

        elif cmd == 31:  # Add to KB
            object, subject = params[1].split(' is ')
            object, subject = object.replace(' ', '_'), subject.replace(' ', '_')
            expr = read_expr(subject + '(' + object + ')')
            not_expr = read_expr('-' + str(expr))

            # Check if expression is not in contradiction with KB.
            true_prover = ResolutionProverCommand(expr, kb).prove()
            false_prover = ResolutionProverCommand(not_expr, kb).prove()
            condition1 = true_prover and not false_prover
            condition2 = not true_prover and not false_prover

            if condition1 or condition2:
                kb.append(expr)
                statement = "OK, I will remember that " + object + " is " + subject
                respond(statement)
            else:
                respond("Sorry this contradicts with what I know!")

        elif cmd == 32:  # Check KB
            print(params[1])
            object, subject = params[1].split(' is ')
            object, subject = object.replace(' ', '_'), subject.replace(' ', '_')
            expr = read_expr(subject + '(' + object + ')')
            infer_kb(expr)

        elif cmd == 33:
            predator, prey = params[1].split(' consumes ')
            predator, prey = predator.replace(' ', '_'), prey.replace(' ', '_')
            expr = read_expr("consumes(" + predator + ',' + prey + ')')
            infer_kb(expr)
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

    questions, answers = read_csv("Files/QA.csv")  # Files/QA.csv
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


def infer_kb(expr):

    print("Searching...")
    answer = ResolutionProver().prove(expr, kb, verbose=False)
    print("answer: " + str(answer))
    print("Please wait... \n")
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
            print("Sorry, I have not been programmed to know that yet.")  # respond


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

    # Produce graphs of fuzzy sets
    # FS.produce_figure(outputfile='output.pdf')

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
- lemmatization dictionary
"""
