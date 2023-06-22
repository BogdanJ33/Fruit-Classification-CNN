import imghdr

from PIL import Image
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import keras
from keras.utils import np_utils

def make_image_into_array(path_to_image, data, labels, i):
    images = os.listdir(path_to_image)
    for img in images:
        if img.endswith(".jpeg") or img.endswith(".png"):

            img_path = os.path.join(path_to_image, img)
            image_format = imghdr.what(img_path)
            if image_format is not None:
                imag = cv2.imread(img_path)
                if imag is not None:
                    img_from_ar = Image.fromarray(imag, 'RGB')
                    resized_image = img_from_ar.resize((50, 50))
                    data.append(np.array(resized_image))
                    labels.append(i)

    return data, labels

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)
def get_fruit_name(label):
    if label==0:
        return "apple"
    if label==1:
        return "avocado"
    if label==2:
        return "banana"
    if label==3:
        return "cherry"
    if label==4:
        return"kiwi"
    if label==5:
        return"mango"
    if label==6:
        return"orange"
    if label==7:
        return"pineapple"
    if label==8:
        return"strawberry"
    if label==9:
        return"watermelon"
def predict_fruit(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    fruit=get_fruit_name(label_index)
    print(fruit)
    print("The predicted Fruit is a "+fruit+" with accuracy =    "+str(acc))

if __name__ == '__main__':
    # data = []
    # dataTest = []
    # labels = []
    # labelsTest = []
   # appleFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//Apple"
   # avocadoFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//avocado"
   # bananaFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//Banana"
   # cherryFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//cherry"
   # kiwiFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//kiwi"
   # mangoFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//mango"
   # orangeFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//orange"
   # pinenappleFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//pinenapple"
   # strawberriesFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//strawberries"
   # watermelonFolderPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//train//watermelon"

    # appleFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//apple"
    # avocadoFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//avocado"
    # bananaFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//banana"
    # cherryFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//cherry"
    # kiwiFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//kiwi"
    # mangoFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//mango"
    # orangeFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//orange"
    # pinenappleFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//pinenapple"
    # strawberriesFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//stawberries"
    # watermelonFolderTestPath = "C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//test//watermelon"

  #  data, labels = make_image_into_array(appleFolderPath, data, labels, 0)
   # data, labels = make_image_into_array(avocadoFolderPath, data, labels, 1)
   # data, labels = make_image_into_array(bananaFolderPath, data, labels, 2)
   # data, labels = make_image_into_array(cherryFolderPath, data, labels, 3)
   # data, labels = make_image_into_array(kiwiFolderPath, data, labels, 4)
   # data, labels = make_image_into_array(mangoFolderPath, data, labels, 5)
   # data, labels = make_image_into_array(orangeFolderPath, data, labels, 6)
   # data, labels = make_image_into_array(pinenappleFolderPath, data, labels, 7)
   # data, labels = make_image_into_array(strawberriesFolderPath, data, labels, 8)
   # data, labels = make_image_into_array(watermelonFolderPath, data, labels, 9)
   #  dataTest, labelsTest = make_image_into_array(appleFolderTestPath, dataTest, labelsTest, 0)
   #  dataTest, labelsTest = make_image_into_array(avocadoFolderTestPath, dataTest, labelsTest, 1)
   #  dataTest, labelsTest = make_image_into_array(bananaFolderTestPath, dataTest, labelsTest, 2)
   #  dataTest, labelsTest = make_image_into_array(cherryFolderTestPath, dataTest, labelsTest, 3)
   #  dataTest, labelsTest = make_image_into_array(kiwiFolderTestPath, dataTest, labelsTest, 4)
   #  dataTest, labelsTest = make_image_into_array(mangoFolderTestPath, dataTest, labelsTest, 5)
   #  dataTest, labelsTest = make_image_into_array(orangeFolderTestPath, dataTest, labelsTest, 6)
   #  dataTest, labelsTest = make_image_into_array(pinenappleFolderTestPath, dataTest, labelsTest, 7)
   #  dataTest, labelsTest = make_image_into_array(strawberriesFolderTestPath, dataTest, labelsTest, 8)
   #  dataTest, labelsTest = make_image_into_array(watermelonFolderTestPath, dataTest, labelsTest, 9)
    # fruits = np.array(data)
    # labels = np.array(labels)
    # np.save("fruits", fruits)
    # np.save("labels", labels)
    # fruitsTest = np.array(dataTest)
    # labelsTest = np.array(labelsTest)
    # np.save("fruitsTest", fruitsTest)
    # np.save("labelsTest", labelsTest)
    fruits = np.load("fruits.npy")
    labels = np.load("labels.npy")
    fruitsTest = np.load("fruitsTest.npy")
    labelsTest = np.load("labelsTest.npy")
    num_classes = len(np.unique(labels))
    data_length = len(fruits)
    num_classes_Test = len(np.unique(labelsTest))
    data_length_Test = len(fruitsTest)

    labels = keras.utils.to_categorical(labels, num_classes)
    labelsTest = keras.utils.to_categorical(labelsTest, num_classes_Test)
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50, 50, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.fit(fruits, labels, batch_size=50, epochs=10, verbose=1)
    score = model.evaluate(fruitsTest, labelsTest, verbose=1)

    predict_fruit("C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//predict//00.jpeg")
    predict_fruit("C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//predict//0.jpeg")
    predict_fruit("C://Users//Bogdan//FruitClassification//Fruit-Classification-CNN//MY_data//predict//1.jpeg")
