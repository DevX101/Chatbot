from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from python_code import vision

import os
from PIL import Image

def azure_image_classification(img_path):
    project_id = 'd6b20a2f-fa01-40cf-8f0b-bec81be9e693'
    cv_key = 'f91c9d8f2ecb4ca4a3bb7f8f571397e6'
    cv_endpoint = 'https://t0266882-imgclassification.cognitiveservices.azure.com/'
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
    cog_key = 'YOUR_COG_KEY'
    cog_endpoint = 'YOUR_COG_ENDPOINT'
    path = os.path.join('Files', 'Images', img_path)

    # Get a client for the computer vision service
    computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))

    # Get a description from the computer vision service
    image_stream = open(path, "rb")
    description = computervision_client.describe_image_in_stream(image_stream)
    vision.show_image_caption(path, description)

    # Specify the features we want to analyze
    features = ['Description', 'Tags', 'Adult', 'Objects', 'Faces']
    analysis = computervision_client.analyze_image_in_stream(image_stream, visual_features=features)
    vision.show_image_analysis(path, analysis)
