import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
from tensorflow.keras import layers, models


def load_and_preprocess_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            preprocessed_image = cv2.resize(image, (224, 224))  # Resize to a common size
            images.append(preprocessed_image)
            labels.append(label)
    return images, labels


tiger_images, tiger_labels = load_and_preprocess_images(r"F:\CapstoneII\tigerdetectionsystem\image", label=1)
non_tiger_images, non_tiger_labels = load_and_preprocess_images(r"F:\CapstoneII\tigerdetectionsystem\image", label=0)
all_images = tiger_images + non_tiger_images
all_labels = tiger_labels + non_tiger_labels

X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
X_train = np.array(X_train)
y_train = np.array(y_train)
model.fit(X_train, y_train, epochs=10, batch_size=32)
X_test = np.array(X_test)
y_test = np.array(y_test)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)

app = Flask(__name__)  # Create a Flask app instance


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            uploaded_file = request.files["file"]
            if uploaded_file.filename != "":
                image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
                preprocessed_image = cv2.resize(image, (224, 224))
                preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
                prediction = model.predict(preprocessed_image)[0][0]
                if prediction > 0.5:
                    result = "Tiger detected with confidence: {:.2f}".format(prediction)
                else:
                    result = "No tiger detected."
                return render_template("index.html", result=result, image_uploaded=True)
        return render_template("index.html", image_uploaded=False)
    except Exception as e:
        error_message = "An error occurred: {}".format(e)
        return render_template("index.html", error_message=error_message, image_uploaded=False)


if __name__ == "__main__":
    app.run(debug=True)
