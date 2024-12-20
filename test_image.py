import tensorflow
import cv2
import numpy as np

model = tensorflow.keras.models.load_model(r'C:\Users\pa662\PycharmProjects\HGM\V11\gestpred.h5')
img = cv2.imread( r'C:\Users\pa662\PycharmProjects\HGM\V11\datasets\test\7\1199.jpg')

def predict(model):

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayimg, (64, 64))
    reshaped = resized.reshape(1, 64, 64, 1)
    prediction = model.predict(reshaped)
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        return "ZERO"
    elif predicted_class == 1:
        return "ONE"
    elif predicted_class == 2:
        return "TWO"
    elif predicted_class == 3:
        return "THREE"
    elif predicted_class == 4:
        return "THUMBSUP"
    elif predicted_class == 5:
        return "FIVE"
    elif predicted_class == 6:
        return "SIX"
    elif predicted_class == 7:
        return "L-SHAPE"

preds = predict(model)
print(preds)
