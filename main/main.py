import imghdr

from PIL import Image
import numpy as np
import os
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import keras
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import load_model

IMAGE_SIZE = 40


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
                    resized_image = img_from_ar.resize((IMAGE_SIZE, IMAGE_SIZE))
                    data.append(np.array(resized_image))
                    labels.append(i)

    return data, labels


def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image)


def get_fruit_name(label):
    if label == 0:
        return "apple"
    if label == 1:
        return "avocado"
    if label == 2:
        return "banana"
    if label == 3:
        return "cherry"
    if label == 4:
        return "kiwi"
    if label == 5:
        return "mango"
    if label == 6:
        return "orange"
    if label == 7:
        return "pineapple"
    if label == 8:
        return "strawberry"
    if label == 9:
        return "watermelon"


def predict_fruit(file, model):
    print("Predicting .................................")
    ar = convert_to_array(file)
    ar = ar / 255
    a = []
    a.append(ar)
    a = np.array(a)
    score = model.predict(a, verbose=1)
    label_index = np.argmax(score)
    acc = np.max(score)
    fruit = get_fruit_name(label_index)
    print(str(label_index) + " - " + fruit)
    file = file.split("\\")[1]
    print("The predicted Fruit from file" + file + " is a " + fruit + " with accuracy =    " + str(acc))


def create_model(fruits, labels, num_classes):
    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))

    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(fruits, labels, batch_size=32, epochs=10, verbose=1)
    model.save("model.h5")
    return model


def create_train_data_labels():
    data = []
    labels = []

    appleFolderPath = "..//MY_data//train//Apple"
    avocadoFolderPath = "..//MY_data//train//avocado"
    bananaFolderPath = "..//MY_data//train//Banana"
    cherryFolderPath = "..//MY_data//train//cherry"
    kiwiFolderPath = "..//MY_data//train//kiwi"
    mangoFolderPath = "..//MY_data//train//mango"
    orangeFolderPath = "..//MY_data//train//orange"
    pinenappleFolderPath = "..//MY_data//train//pinenapple"
    strawberriesFolderPath = "..//MY_data//train//strawberries"
    watermelonFolderPath = "..//MY_data//train//watermelon"

    data, labels = make_image_into_array(appleFolderPath, data, labels, 0)
    data, labels = make_image_into_array(avocadoFolderPath, data, labels, 1)
    data, labels = make_image_into_array(bananaFolderPath, data, labels, 2)
    data, labels = make_image_into_array(cherryFolderPath, data, labels, 3)
    data, labels = make_image_into_array(kiwiFolderPath, data, labels, 4)
    data, labels = make_image_into_array(mangoFolderPath, data, labels, 5)
    data, labels = make_image_into_array(orangeFolderPath, data, labels, 6)
    data, labels = make_image_into_array(pinenappleFolderPath, data, labels, 7)
    data, labels = make_image_into_array(strawberriesFolderPath, data, labels, 8)
    data, labels = make_image_into_array(watermelonFolderPath, data, labels, 9)
    fruits = np.array(data)
    labels = np.array(labels)
    np.save("fruits", fruits)
    np.save("labels", labels)


def create_test_data_labels():
    dataTest = []
    labelsTest = []
    appleFolderTestPath = "..//MY_data//test//apple"
    avocadoFolderTestPath = "..//MY_data//test//avocado"
    bananaFolderTestPath = "..//MY_data//test//banana"
    cherryFolderTestPath = "..//MY_data//test//cherry"
    kiwiFolderTestPath = "..//MY_data//test//kiwi"
    mangoFolderTestPath = "..//MY_data//test//mango"
    orangeFolderTestPath = "..//MY_data//test//orange"
    pinenappleFolderTestPath = "..//MY_data//test//pinenapple"
    strawberriesFolderTestPath = "..//MY_data//test//stawberries"
    watermelonFolderTestPath = "..//MY_data//test//watermelon"

    dataTest, labelsTest = make_image_into_array(appleFolderTestPath, dataTest, labelsTest, 0)
    dataTest, labelsTest = make_image_into_array(avocadoFolderTestPath, dataTest, labelsTest, 1)
    dataTest, labelsTest = make_image_into_array(bananaFolderTestPath, dataTest, labelsTest, 2)
    dataTest, labelsTest = make_image_into_array(cherryFolderTestPath, dataTest, labelsTest, 3)
    dataTest, labelsTest = make_image_into_array(kiwiFolderTestPath, dataTest, labelsTest, 4)
    dataTest, labelsTest = make_image_into_array(mangoFolderTestPath, dataTest, labelsTest, 5)
    dataTest, labelsTest = make_image_into_array(orangeFolderTestPath, dataTest, labelsTest, 6)
    dataTest, labelsTest = make_image_into_array(pinenappleFolderTestPath, dataTest, labelsTest, 7)
    dataTest, labelsTest = make_image_into_array(strawberriesFolderTestPath, dataTest, labelsTest, 8)
    dataTest, labelsTest = make_image_into_array(watermelonFolderTestPath, dataTest, labelsTest, 9)

    fruitsTest = np.array(dataTest)
    labelsTest = np.array(labelsTest)
    np.save("fruitsTest", fruitsTest)
    np.save("labelsTest", labelsTest)

    return dataTest, labelsTest


def evaluate_with_custom_epochs(model, epoch_num):
    evaluation_results = []
    accuracy_results = []
    for i in range(epoch_num):
        print("----------------------------------------\n")
        print("Epoch number: " + str(i) + "\n")
        model_values = model.fit(fruits, labels, batch_size=32, epochs=1, verbose=1)  # Train the model for 1 epoch
        score = model.evaluate(fruitsTest, labelsTest, verbose=1)
        evaluation_results.append(score)  # Store the evaluation result for the current epoch
        accuracy_results.append(model_values.history['accuracy'])
        # Print and plot the validation accuracy after each epoch
        print("Validation Accuracy after Epoch {}: {:.2f}%".format(i + 1, score[1] * 100))

    epochs = range(1, epoch_num+1)
    validation_accuracy = [result[1] for result in
                           evaluation_results]  # Extracting validation accuracy from the evaluation results
    plt.figure()
    plt.plot(epochs, validation_accuracy, label = 'Validation accuracy')
    plt.plot(epochs, accuracy_results, label = 'Training accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Epochs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # create_train_data_labels()
    # create_test_data_labels()
    fruits = np.load("fruits.npy")
    labels = np.load("labels.npy")
    fruitsTest = np.load("fruitsTest.npy")
    labelsTest = np.load("labelsTest.npy")
    num_classes = len(np.unique(labels))
    fruits = fruits.astype('float32') / 255
    fruitsTest = fruitsTest.astype("float32") / 255
    num_classes_Test = len(np.unique(labelsTest))
    labels = keras.utils.to_categorical(labels, num_classes)
    labelsTest = keras.utils.to_categorical(labelsTest, num_classes_Test)
    # model = create_model(fruits, labels, num_classes)
    model = load_model("model.h5")
    # evaluate_with_custom_epochs(model, 25)
    # model.fit(fruits, labels, batch_size=32, epochs=25, verbose=1)  # Train the model for 1 epoch
    score = model.evaluate(fruitsTest, labelsTest, verbose=1)
    # model.save("model.h5")

    images = os.listdir("..//MY_data//predict")
    for img in images:
        img_path = os.path.join("..//MY_data//predict", img)
        predict_fruit(img_path, model)
