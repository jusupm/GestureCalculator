from cvzone.ClassificationModule import Classifier
from PIL import Image
from numpy import asarray

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["+","-","*","/","=","numbers"]

tp=0
fn=0

for i in range (1,121):

    img=Image.open("Data/Evaluation/plus/Image_{}.jpg".format(i),"r")
    data=asarray(img)

    prediction, index = classifier.getPrediction(data, draw=False)
    if (index==0):
        tp+=1
    else:
        fn+=1
print(tp)
print(fn)