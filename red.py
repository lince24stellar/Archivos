import cv2
import numpy as np
import unittest

class TestObjectDetection(unittest.TestCase):

    def setUp(self):
        self.config = "data/yolov3.cfg"
        self.weights = "data/yolov3.weights"
        self.labels = open("data/coco.names").read().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        self.image = cv2.imread("img/coche.jpg")
        self.height, self.width, _ = self.image.shape

    def test_object_detection(self):
        blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[:4] * np.array([self.width, self.height, self.width, self.height])
                    (x_center, y_center, w, h) = box.astype("int")
                    x = int(x_center - (w / 2))
                    y = int(y_center - (h / 2))

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Perform assertions based on expected results
        self.assertGreater(len(boxes), 0)
        self.assertGreater(len(confidences), 0)
        self.assertGreater(len(classIDs), 0)

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
