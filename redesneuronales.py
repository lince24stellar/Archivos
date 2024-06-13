import cv2
import numpy as np
import unittest
config = "data/yolov3.cfg"

weights = "data/yolov3.weights"
LABELS = open("data/coco.names").read().split("\n")
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config, weights)

image = cv2.imread("img/coche.jpg")
height, width, _ = image.shape

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                              swapRB=True, crop=False)

ln = net.getLayerNames()

ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

net.setInput(blob)
outputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for output in outputs:
     for detection in output:
          scores = detection[5:]
          classID = np.argmax(scores)
          confidence = scores[classID]

          if confidence > 0.5:
          
               box = detection[:4] * np.array([width, height, width, height])
               (x_center, y_center, w, h) = box.astype("int")
               x = int(x_center - (w / 2))
               y = int(y_center - (h / 2))

               boxes.append([x, y, w, h])
               confidences.append(float(confidence))
               classIDs.append(classID)

idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
print("idx:", idx)

if len(idx) > 0:
     for i in idx:
          (x, y) = (boxes[i][0], boxes[i][1])
          (w, h) = (boxes[i][2], boxes[i][3])

          color = colors[classIDs[i]].tolist()
          text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
          cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
          cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, color, 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

class TestStringMethods(unittest.TestCase):

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