import os
import cv2
import time
import matplotlib.image as img
import numpy as np
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras import models
from PIL import Image
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import random
app = Flask(__name__)

conf_thresh = 0.3
nms_thresh = 0.1
path = "./"
nets = cv2.dnn.readNet('yolov2-food100.weights', 'yolov2-food100.cfg')
predicted_class = None
glob_image = None

#commenting
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

#predicts food class with model, image as input

def predict_class(model, image2,imageorig, show = True):
   global glob_image
   print("my IMAGE 2 is as below----------------------------")
   print(image2)
   #preporcessing of image tha twas uplaoded 
   img = load_img(image2, target_size=(299, 299))
   imgarray = img_to_array(img)                    
   imgarray2 = np.expand_dims(imgarray, axis=0)         
   imgarray2 /= 255.   
   #img = image2.resize(target_size, refcheck=False)
   pred = model.predict(imgarray2)
   index = np.argmax(pred)
   print(index)
   #creatinglist of food classes 
   food_list = ['apple_pie', 'beef_carpaccio', 'bibimbap', 'cup_cakes', 'foie_gras', 
                'french_fries', 'garlic_bread', 'pizza', 'spring_rolls', 
                'spaghetti_carbonara', 'strawberry_shortcake']
   food_list.sort()
   predicted_class = food_list[index]

   print(predicted_class)
   
   glob_image = 'Image' + predicted_class
   cv2.imwrite(f'static/{glob_image}.jpg', imageorig)
  
   print(predicted_class)
   return predicted_class

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

        image2path=file_path
        print('filepath is------')
        print(file_path)
        #f = open("lentil-samosa-recipe-01.jpg", 'rb')
        imageorig = Image.open(f)
        imageorig = cv2.imread(os.path.join(basepath, 'uploads', secure_filename(f.filename)))
        print('image originalis-----')
        print(imageorig)

        model_best = models.load_model('best_model_11class.hdf5',compile = False)
        predicted_class=predict_class(model_best, image2path,imageorig, True)

        #OLDpredicted_class = predict_image(image2=image2)
        print(predicted_class)
       # predicted_class = predict_image(image2=image2)
        #print(predicted_class)
        imagepath = f'static/{glob_image}.jpg'
        print('image path is')
        print(imagepath)
        return render_template('InputWeight.html', predicted_class=predicted_class, imagepath=f'static/{glob_image}.jpg')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        weight = int(request.form['weight'])

        # os.remove(f'static/{glob_image}.jpg')
        print(predicted_class)
        cal_dict = {'apple_pie': 2.37, 'beef_carpaccio':1.26, 'bibimbap':1.46, 'cup_cakes':3.05, 'foie_gras':4.62, 
                'french_fries':3.19, 'garlic_bread':3.49, 'pizza':2.66, 'spring_rolls':1.53, 
                'spaghetti_carbonara':2.0, 'strawberry_shortcake':3.46}
        calories= cal_dict[predicted_class] * weight
        print(calories)
        imgs = os.listdir('static/generated-images/')
        #imgs = [ file for file in imgs]
        #imgrand = random.sample(imgs,k=5)
       

        d=random.choice(imgs)
        e=random.choice(imgs)
        f=random.choice(imgs)
        g=random.choice(imgs)
        #os.startfile(d)
        weightofperson = int(request.form['weightofperson'])
        yogatime=round(calories/(0.049*weightofperson),2)
        print(yogatime)
        danceminutes=round(calories/(0.084*weightofperson),2)
        print(danceminutes)
        return render_template('result.html', calories=calories, d=d,e=e,f=f,g=g,yogatime=yogatime,danceminutes=danceminutes)

if __name__ == '__main__':
    app.run(debug=False)