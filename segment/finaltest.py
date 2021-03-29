import numpy as np
import cv2
import os
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import keras

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   #Restrict TensorFlow to only allocate 3GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]) #Cai nay minh dung de chia bo nho GPU may, moi nguoi co the tuy chinh
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


def my_IoU(y_true, y_pred):
    y_pred = K.argmax(y_pred)
    y_pred = K.cast(y_pred, 'float32')
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    intersection = K.sum(y_true * y_pred)
    IoU = (intersection+0.0000001) / (K.sum(y_true) + K.sum(y_pred) - intersection + 0.001)
    return IoU

model1 = load_model('segment/new_dlv3+_hand_segment_v2.h5', custom_objects={'my_IoU': my_IoU})
model2 = load_model('segment/classify_hand_90,7_89,9_2431712.h5')

classes=['Thump Up', 'OK', 'Lucky Flower', 'Chinese Hello', 'Bird', 'Take Photo', 'Hand Up', 'Hand Down', 'I Love You', 'Flower', 'Muscle', 'Cross Hand Over Hand', 'Hi', 'Stop', 'Rock', 'Sleepy', 'Pray', 'Heart', 'Big Heart', 'Small Heart', 'Binocullar', 'Plus Sign', 'Dab', 'Shut Up', 'Rabbit', 'Pistol', 'Hand Over Head', 'Touch Cheek', 'Touch Head', 'Wait', 'Calling', 'No Gesture']

def segment(path_name):

    # dir = r'C:\Users\ad\Desktop\new_test'
    #link = sorted([os.path.join(path_name, fname) for fname in os.listdir(path_name)])
    imginput = cv2.imread(path_name)
    imginput.astype('float32')
    imginput = cv2.resize(imginput, (256, 256))
    imginput = imginput/255.0*2-1
    x = np.zeros((1,) + (256, 256) + (3,))
    x[0] = imginput
    pred = model1.predict(x)

    pred = pred[0]
    print(pred)
    pred1 = np.zeros((256, 256))
    for k in range(256):
        for j in range(256):
            if pred[k][j][1]>0.3:
                pred1[k][j]=255
    img = cv2.imread(path_name)
    img = cv2.resize(img, (256, 256))
    pred1 = np.uint8(pred1)
    result1 = cv2.bitwise_and(img, img, mask=pred1)
    segment_img = cv2.imwrite('1.jpg', result1)
    image = tf.keras.preprocessing.image.load_img('1.jpg', target_size=(224, 224))
    input_arr = keras.preprocessing.image.img_to_array(image)
    #os.remove('1.jpg')
    input_arr = input_arr * 1. / 255
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    final_result = model2.predict(input_arr)
    print(classes[np.argmax(final_result)])
    class_seg = classes[np.argmax(final_result)]
    if segment_img == True:
        res = "1.jpg"    
    return res, class_seg

