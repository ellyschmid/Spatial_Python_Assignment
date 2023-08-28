import os
import cv2
import keras
import random
from cv2.ximgproc import segmentation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# set seed
random.seed(1404)

# set path to input
path = ""  # path to images
annot = ""  # path to corresponding image files

# Iterate through images and annotation files and join them
for e, i in enumerate(os.listdir(annot)):
    if e < 10:
        filename = i.split(".")[0] + ".png"
        print(filename)
        img = cv2.imread(os.path.join(path, filename))
        df = pd.read_csv(os.path.join(annot, i))
        plt.imshow(img)

        # Draw rectangles on the image
        for row in df.iterrows():
            x1 = int(row[1][4])
            y1 = int(row[1][5])
            x2 = int(row[1][6])
            y2 = int(row[1][7])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plt.figure()
        plt.imshow(img)
        break

# Create selective search segmentation
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

im = cv2.imread(os.path.join(path, "ship_1.png"))
ss.setBaseImage(im)
ss.switchToSelectiveSearchFast()
rects = ss.process()
imOut = im.copy()

# Draw rectangles on the image based on selective search results
for i, rect in enumerate(rects):
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

plt.imshow(imOut)
plt.show()

# Create empty lists for training images and labels
train_images = []
train_labels = []


# Define a function to calculate Intersection over Union (IoU) between bounding boxes
def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# Iterate through annotation files
for e, i in enumerate(os.listdir(annot)):
    try:
        if i.startswith("ship_"):
            filename = i.split(".")[0] + ".png"
            print(e, filename)
            image = cv2.imread(os.path.join(path, filename))
            df = pd.read_csv(os.path.join(annot, i))
            gtvalues = []

            # Extract ground truth bounding boxes
            for row in df.iterrows():
                for row in df.iterrows():
                    x1 = int(row[1][4])
                    y1 = int(row[1][5])
                    x2 = int(row[1][6])
                    y2 = int(row[1][7])
                gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

            # Run selective search on the image
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0

            # Process selective search results
            for e, result in enumerate(ssresults):
                if e < 2000 and flag == 0:
                    for gtval in gtvalues:
                        x, y, w, h = result
                        iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                        if counter < 30:
                            if iou > 0.70:
                                timage = imout[y:y + h, x:x + w]
                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(1)
                                counter += 1
                        else:
                            fflag = 1
                        if falsecounter < 30:
                            if iou < 0.3:
                                timage = imout[y:y + h, x:x + w]
                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                        else:
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1
    except Exception as e:
        print(e)
        print("error in " + filename)
        continue

# Convert lists to NumPy arrays
X_new = np.array(train_images)
y_new = np.array(train_labels)

# Load VGG16 model with pre-trained weights
vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()

# Freeze layers up to layer 15
for layers in vggmodel.layers[:15]:
    layers.trainable = False

# Define custom output layer
X = vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)

# Create the final model
model_final = Model(inputs=vggmodel.input, outputs=predictions)

# Compile the model
opt = Adam(learning_rate=0.0001)
model_final.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
model_final.summary()


# Define a custom LabelBinarizer
class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


# Apply the custom LabelBinarizer to convert labels
lenc = MyLabelBinarizer()
Y = lenc.fit_transform(y_new)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Data augmentation
aug = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest"
)

BS = 5
EPOCHS = 5

# Train the model
print("[INFO] training head...")
hist = model_final.fit(
    aug.flow(X_train, y_train, batch_size=BS),
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BS,
    epochs=EPOCHS
)

# Save the trained model
model_final.save("")

# Plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig("plot")

# Test on test images
for im in X_test:
    img = np.expand_dims(im, axis=0)
    out = model_final.predict(img)
    if out[0][0] > out[0][1]:
        print("ship")
    else:
        print("no ship")

# test on test image
# Initialize a counter variable z
z = 0

# Iterate through files in the specified directory (path)
for e, i in enumerate(os.listdir(path)):
    # Check if the file name starts with "test"
    if i.startswith("test"):
        # Increment the counter z
        z += 1

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(path, i))

        # Set the base image for selective search
        ss.setBaseImage(img)

        # Switch to the selective search mode optimized for speed
        ss.switchToSelectiveSearchFast()

        # Perform selective search and get the list of proposed regions (rectangles)
        ssresults = ss.process()

        # Create a copy of the original image
        imout = img.copy()

        # Iterate through the selective search results
        for e, result in enumerate(ssresults):
            # Limit the number of processed regions to 2000 (if there are more)
            if e < 2000:
                # Extract the coordinates and dimensions of the region
                x, y, w, h = result

                # Crop the region from the image
                timage = imout[y:y + h, x:x + w]

                # Resize the cropped region to a fixed size (224x224 pixels)
                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)

                # Expand the dimensions of the resized region to match the model input shape
                img = np.expand_dims(resized, axis=0)

                # Make a prediction using the trained model
                out = model_final.predict(img)

                # Check if the model predicts "ship" (based on a threshold of 0.85)
                if out[0][0] > 0.5:
                    # If predicted as "ship," draw a green rectangle around the region
                    cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

        # Display the image with rectangles (regions predicted as "ship")
        plt.figure()
        plt.imshow(imout)
        plt.show()
