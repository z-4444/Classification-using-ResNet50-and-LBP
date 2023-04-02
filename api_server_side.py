#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import traceback
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from skimage.feature import local_binary_pattern
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from joblib import load
import base64
import io

model = load('svm_model.joblib')

def solution(img_base64):
    # Preprocess the image
    image_bytes = base64.b64decode(img_base64)
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    resnet_features = resnet_model.predict(np.expand_dims(img, axis=0))[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, 8, 1)
    (hist, _) = np.histogram(lbp.ravel(),
                              bins=np.arange(0, 10),
                              range=(0, 10))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    img_height, img_width, _ = img.shape
    features = np.concatenate([resnet_features, [img_height, img_width], hist ])
    features= features.reshape(-1,features.shape[0])

    # Make a prediction on the image
    pred = model.predict(features)

    # Get the predicted class label
    class_idx = pred[0]
    if class_idx == 0:
        class_label = 'Boot'
    elif class_idx == 1:
        class_label = 'Sandal'
    elif class_idx == 2:
        class_label = 'Shoe'

    # Return the predicted class label
    return class_label

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    try:
        # Get the image data from the JSON payload
        data = request.get_json()
        img_base64 = data['image']
        
        # Call the solution function with the image data
        class_label = solution(img_base64)
        
        # Return the predicted class label as a JSON response
        return jsonify({'class_label': class_label})
    except Exception as e:
        return str(traceback.format_exc())

app.run(host='0.0.0.0', port=5000)

