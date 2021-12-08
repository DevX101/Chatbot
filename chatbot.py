import aiml
import wikipedia
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver, ResolutionProverCommand
from simpful import *

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="chatbot.xml")

read_expr = Expression.fromstring
kb = []
data = pd.read_csv('sampleKB.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

FS = FuzzySystem()


def main():

    while True:
        try:
            user_input = input(">Human: ")
            response = kernel.respond(user_input)
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break

        if response[0] == '#':
            params = response[1:].split('$')
            cmd = int(params[0])
            if cmd == 0:
                print(params[1])
                break
            elif cmd == 99:
                print("I did not get that, please try again.")
            elif cmd == 1:
                try:
                    # wiki_summary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                    # print(wiki_summary)
                    answer = compute_similarity(user_input)
                    print(answer)
                except:
                    # Input that is copy/pasted will cause errors.
                    print("Sorry, I do not know that. Be more specific!")

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
                    print('OK, I will remember that', object, 'is', subject)
                else:
                    print("Sorry this contradicts with what I know!")

            elif cmd == 32:  # if the input pattern is "check that * is *"
                object, subject = params[1].split(' is ')
                expr = read_expr(subject + '(' + object + ')')
                answer = ResolutionProver().prove(expr, kb, verbose=True)
                if answer:
                    print('Correct.')
                else:
                    not_expr = read_expr('-' + str(expr))
                    true_prover = ResolutionProverCommand(expr, kb).prove()
                    false_prover = ResolutionProverCommand(not_expr, kb).prove()

                    # Provide definitive answer given the condition of the expression using resolution.
                    if not true_prover and false_prover:
                        print("Incorrect")
                    else:
                        print("Sorry I don't know")
        else:
            print(response)


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


if __name__ == "__main__":
    main()

"""
Tasks:
- set name variable in aiml
- fuzzy logic (B)
- sentiment analysis (B)
- web service (A)
"""
