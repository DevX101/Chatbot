from tensorflow.keras import Sequential, Input, losses, optimizers, applications, callbacks
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import keras_tuner as kt

EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 200
CATEGORIES = 3
LEARNING_RATE = 0.001

labels = ["bear", "deer", "squirrel"]
data = []
for label in labels:
    path = os.path.join("Dataset", label)
    class_num = labels.index(label)
    for img in os.listdir(path):
        try:
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            data.append([resized_arr, class_num])
        except Exception as e:
            print(e)

dataset = np.array(data, dtype=object)

X = []
y = []
for feature, label in dataset:
    X.append(feature)
    y.append(label)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

x_train = np.array(x_train) / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.array(x_test) / 255.0
x_test = np.expand_dims(x_test, -1)
y_train = np.array(y_train)
y_test = np.array(y_test)

data_gen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)
data_gen.fit(x_train)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(CATEGORIES, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

learn = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

# model.save("Weights/model.h5")
# model.summary()

"""Evaluation"""

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
print("Test loss: ", test_loss)

# Graph
acc = learn.history['accuracy']
val_acc = learn.history['val_accuracy']
loss = learn.history['loss']
val_loss = learn.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Classification Report
predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1, -1)[0]
print(classification_report(y_test, predictions, target_names=['Bear', 'Deer', "Squirrel"]))

# k-Fold Cross-Validation
def cross_validation(X, y):
    k_fold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_scores = []
    for train, test in k_fold.split(X, y):
        model.fit(X[train], y[train], epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X[test], y[test]))
        scores = model.evaluate(X[test], y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cv_scores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (float(np.mean(cv_scores)), float(np.std(cv_scores))))

"""Uncomment to perform k-Fold cross-validation"""
# cross_validation(X, y)

# Transfer Learning
def transfer_model(model):
    vgg_model = applications.vgg16.VGG16(weights="imagenet", include_top=False,
                                         input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
    for layer in vgg_model.layers[0:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False

    return model

"""Hyperparameter Optimized Model"""
def build_model(hp):
    model = Sequential()
    # model.add(transfer_model)

    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Hyperparameter Tuning
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CATEGORIES, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def tune_model(model):
    tuner = kt.Hyperband(model, objective='val_accuracy', max_epochs=EPOCHS, factor=3,
                         directory='Weights/Tuning', project_name="tuning")
    tuner.search_space_summary()
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=[stop_early])

    best_hp = tuner.get_best_hyperparameters()[0]
    h_model = tuner.hypermodel.build(best_hp)
    h_model.summary()
    history = h_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    eval_result = h_model.evaluate(x_test, y_test, return_dict=True)
    print("[test loss, test accuracy]:", eval_result)

"""Uncomment to perform hyperparameter tuning"""
# tune_model(build_model)
