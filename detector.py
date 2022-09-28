import tensorflow as tf
import numpy as np
import cv2
import os
import time
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self):
        self.obj = dict()

    def readClasses(self, path):
        with open(path, 'r') as f:
            self.labels = f.readlines()

        # Set colours of bounding boxes
        self.colours = np.random.uniform(low=0, high=255, size=(len(self.labels), 3))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.dir = "Weights/Pretrained"
        get_file(fname=fileName,
                 origin=modelURL,
                 cache_dir=self.dir,
                 cache_subdir="checkpoints",
                 extract=True)

    def loadModel(self):
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.dir, "checkpoints", self.modelName, "saved_model"))

    def createBoundingBox(self, image):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)
        bboxs = detections['detection_boxes'][0].numpy()
        index = detections['detection_classes'][0].numpy().astype(np.int32)
        score = detections['detection_scores'][0].numpy()
        height, width, channels = image.shape
        boxIdx = tf.image.non_max_suppression(bboxs, score,
                                              max_output_size=50,
                                              iou_threshold=0.5,
                                              score_threshold=0.5)
        if len(boxIdx) != 0:
            for i in boxIdx:
                bbox = tuple(bboxs[i].tolist())
                conf = round(100*score[i])
                classIndex = index[i]
                classLabel = self.labels[classIndex]
                classColour = self.colours[classIndex]
                displayText = '{}: {}%'.format(classLabel, conf)

                if classLabel in self.obj.keys():
                    self.obj[classLabel] = self.obj[classLabel] + 1
                else:
                    self.obj[classLabel] = 1

                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * width, xmax * width, ymin * height, ymax * height)
                xmin, xmax, ymin, ymax= int(xmin), int(xmax), int(ymin), int(ymax)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColour, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColour, 2)

        return image

    def getLabels(self):
        return self.obj

    def predictImage(self, path):
        image = cv2.imread(path)
        bboxImg = self.createBoundingBox(image)
        cv2.imwrite(self.modelName+".jpg", bboxImg)
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Error opening file...")
            return

        (success, image) = cap.read()
        startTime = 0
        while success:
            currentTime = time.time()
            fps = 1/(currentTime-startTime)
            startTime = currentTime
            bboxImg = self.createBoundingBox(image)

            cv2.putText(bboxImg, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.imshow("Result", bboxImg)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()
        cv2.destroyAllWindows()
