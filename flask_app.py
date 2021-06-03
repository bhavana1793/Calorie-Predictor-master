import os
import cv2
import time

import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

conf_thresh = 0.3
nms_thresh = 0.1
path = "./"
nets = cv2.dnn.readNet('yolov2-food100.weights', 'yolov2-food100.cfg')
predicted_class = None
glob_image = None


def get_prediction(image, net, LABELS, COLORS):
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > conf_thresh:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh,
                            nms_thresh)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            # print(boxes)
            # print(LABELS[classIDs[i]])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image, confidences, LABELS[classIDs[np.argmax(np.array(confidences))]]


def predict_image(image2):
    classes = []
    global glob_image
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()

    if image2 is None:
        image2 = cv2.imread("./pizza_41.jpg")
    colors = np.random.uniform(0, 255, size=(100, 3))
    res, confidences, predicted_class = get_prediction(image2, nets, classes, colors)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    print(predicted_class)
    glob_image = 'Image' + predicted_class
    cv2.imwrite(f'static/{glob_image}.jpg', res)
    return predicted_class


@app.route('/')
def home():
    return render_template('HomePage.html')


@app.route('/upload')
def upload():
    return render_template('UploadImg.html')


@app.route('/predict', methods=['POST'])
def predict():
    global predicted_class
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        image2 = cv2.imread(file_path)
        print(file_path)

        predicted_class = predict_image(image2=image2)
        print(predicted_class)

        return render_template('InputWeight.html', predicted_class=predicted_class, imagepath=f'static/{glob_image}.jpg')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        weight = int(request.form['weight'])

        # os.remove(f'static/{glob_image}.jpg')
        '''
        cal_dict = {'rice': 5,
'eels-on-rice' : 10,
'pilaf
chicken-n-egg-on-rice
pork-cutlet-on-rice
beef-curry
sushi
chicken-rice
fried-rice
tempura-bowl
bibimbap
toast
croissant
roll-bread
raisin-bread
chip-butty
hamburger
pizza
sandwiches
udon-noodle
tempura-udon
soba-noodle
ramen-noodle
beef-noodle
tensin-noodle
fried-noodle
spaghetti
Japanese-style-pancake
takoyaki
gratin
sauteed-vegetables
croquette
grilled-eggplant
sauteed-spinach
vegetable-tempura
miso-soup
potage
sausage
oden
omelet
ganmodoki
jiaozi
stew
teriyaki-grilled-fish
fried-fish
grilled-salmon
salmon-meuniere
sashimi
grilled-pacific-saury-
sukiyaki
sweet-and-sour-pork
lightly-roasted-fish
steamed-egg-hotchpotch
tempura
fried-chicken
sirloin-cutlet
nanbanzuke
boiled-fish
seasoned-beef-with-potatoes
hambarg-steak
beef-steak
dried-fish
ginger-pork-saute
spicy-chili-flavored-tofu
yakitori
cabbage-roll
rolled-omelet
egg-sunny-side-up
fermented-soybeans
cold-tofu
egg-roll
chilled-noodle
stir-fried-beef-and-peppers
simmered-pork
boiled-chicken-and-vegetables
sashimi-bowl
sushi-bowl
fish-shaped-pancake-with-bean-jam
shrimp-with-chill-source
roast-chicken
steamed-meat-dumpling
omelet-with-fried-rice
cutlet-curry
spaghetti-meat-sauce
fried-shrimp
potato-salad
green-salad
macaroni-salad
Japanese-tofu-and-vegetable-chowder
pork-miso-soup
chinese-soup
beef-bowl
kinpira-style-sauteed-burdock
rice-ball
pizza-toast
dipping-noodles
hot-dog
french-fries
mixed-rice
goya-chanpuru
 }
 '''

        #calories= cal_dict['predicted_class'] * weight
        return render_template('result.html', calories=80)

if __name__ == '__main__':
    app.run(debug=False)