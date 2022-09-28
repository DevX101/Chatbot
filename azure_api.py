from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.ai.language.questionanswering import models as qna
from python_code import vision
from python_code import faces

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
import uuid
import json

cog_key = 'f91c9d8f2ecb4ca4a3bb7f8f571397e6'
cog_endpoint = 'https://t0266882-imgclassification.cognitiveservices.azure.com/'
cog_region = 'uksouth'

def azure_image_classification(img_path):
    project_id = 'd6b20a2f-fa01-40cf-8f0b-bec81be9e693'
    cv_key = cog_key
    cv_endpoint = cog_endpoint
    model_name = 'tundraClassifier'
    path = os.path.join('Files', 'Images', img_path)
    file = open(path, "rb")

    # Create an instance of the prediction service
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": cv_key})
    custom_vision_client = CustomVisionPredictionClient(endpoint=cv_endpoint, credentials=credentials)

    classification = custom_vision_client.classify_image(project_id, model_name, file.read())
    prediction = classification.predictions[0].tag_name
    # img = Image.open(path)
    # img.show()

    return prediction

def azure_image_analysis(img_path):

    path = os.path.join('Files', 'Images', img_path)

    # Get a client for the computer vision service
    computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

    # Get a description from the computer vision service
    image_stream = open(path, "rb")
    description = computervision_client.describe_image_in_stream(image_stream)
    caption = vision.show_image_caption(path, description)

    # Specify the features we want to analyze
    image_stream = open(path, "rb")
    features = ['Description', 'Tags', 'Adult', 'Objects', 'Faces']
    analysis = computervision_client.analyze_image_in_stream(image_stream, visual_features=features)
    vision.show_image_analysis(path, analysis)

    return caption

def azure_ocr(img_path):

    computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))
    path = os.path.join('Files', 'Images/Text', img_path)
    image_stream = open(path, "rb")

    # Use the Computer Vision service to find text in the image
    read_results = computervision_client.recognize_printed_text_in_stream(image_stream)

    # Open image to display it.
    fig = plt.figure(figsize=(7, 7))
    img = Image.open(path)
    draw = ImageDraw.Draw(img)

    # Process the text line by line
    text = ""
    for region in read_results.regions:
        for line in region.lines:

            # Show the position of the line of text
            l, t, w, h = list(map(int, line.bounding_box.split(',')))
            draw.rectangle(((l, t), (l + w, t + h)), outline='magenta', width=5)

            # Read the words in the line of text
            line_text = ''
            for word in line.words:
                line_text += word.text + ' '
            text = text + line_text.rstrip() + '\n'

    # Show the image with the text locations highlighted
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    return text

def azure_face(path):

    # Create a face detection client.
    face_client = FaceClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

    # Open an image
    image_path = os.path.join('Files/Images/Faces', path)
    image_stream = open(image_path, "rb")

    # Detect faces and specified facial attributes
    attributes = ['age', 'emotion']
    detected_faces = face_client.face.detect_with_stream(image=image_stream, return_face_attributes=attributes)

    # Display the faces and attributes (code in python_code/faces.py)
    faces.show_face_attributes(image_path, detected_faces)

def azure_translateText(text, to_lang, from_lang='en'):

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region': cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    return response[0]["translations"][0]["text"]

def azure_text_analysis(doc):

    # Get a client for your text analytics cognitive service resource
    text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint,
                                                credentials=CognitiveServicesCredentials(cog_key))

    language_analysis = text_analytics_client.detect_language(documents=doc)
    key_phrase_analysis = text_analytics_client.key_phrases(documents=doc)
    sentiment_analysis = text_analytics_client.sentiment(documents=doc)
    entity_analysis = text_analytics_client.entities(documents=doc)

    for num in range(len(doc)):
        print('--- {} ---'.format(doc[num]['id']))

        """Detect Language"""
        lang = language_analysis.documents[num].detected_languages[0]
        print('Language: ', lang.name)

        """Extract Key Phrases"""
        print('\nKey Phrases:')
        key_phrases = key_phrase_analysis.documents[num].key_phrases
        for key_phrase in key_phrases:
            print(key_phrase)

        """Determine Sentiment"""
        print('\nSentiment Score:')
        sentiment_score = sentiment_analysis.documents[num].score
        if sentiment_score < 0.5:
            sentiment = 'negative'
        else:
            sentiment = 'positive'
        print('{} ({})'.format(sentiment, sentiment_score))

        """Extract Known Entities"""
        print("\nNamed Entities:")
        entities = entity_analysis.documents[num].entities
        for entity in entities:
            if entity.type in ['DateTime', 'Location', 'Person', 'Geographical', 'Natural']:
                link = '(' + entity.wikipedia_url + ')' if entity.wikipedia_id is not None else ''
                print(' - {}: {} {}'.format(entity.type, entity.name, link))
        print('\n')

def azure_qna(question):

    endpoint = "https://t0266882-language.cognitiveservices.azure.com/"
    credential = AzureKeyCredential("a352651987c14e2b82f26adf26b3242e")

    column_names = ["Questions", "Answers"]
    df = pd.read_csv("Files/QA.csv", names=column_names)
    answers = df.Answers.drop_duplicates()
    answers = answers.head().tolist()

    client = QuestionAnsweringClient(endpoint, credential)
    with client:
        input = qna.AnswersFromTextOptions(
            question=question,
            text_documents=answers
        )

        output = client.get_answers_from_text(input)

    best_answer = [a for a in output.answers if a.confidence > 0.9][0]

    # print(u"Q: {}".format(input.question))
    # print(u"A: {}".format(best_answer.answer))
    # print("Confidence Score: {}".format(output.answers[0].confidence))

    return best_answer.answer
