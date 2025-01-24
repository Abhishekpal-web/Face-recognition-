# Face-recognition-
This project builds a simple face recognition system using OpenCV and a pre-trained deep learning model for face embeddings.

# Importing necessary libraries
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

# Step 1: Load the pre-trained face detector and face embedding model
def load_models():
    face_detector = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",  # Prototxt file path
        "res10_300x300_ssd_iter_140000.caffemodel"  # Pre-trained weights
    )

    # Load face embedding model (e.g., FaceNet or OpenCV's DNN face embeddings)
    face_embedding_model = cv2.dnn.readNetFromTorch("openface.nn4.small2.v1.t7")

    return face_detector, face_embedding_model

# Step 2: Load and preprocess dataset (images and labels)
def load_dataset(dataset_path):
    embeddings = []
    labels = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            embeddings.append(get_face_embedding(image, face_detector, face_embedding_model))
            labels.append(person_name)

    return np.array(embeddings), np.array(labels)

# Step 3: Extract face embeddings from an image
def get_face_embedding(image, face_detector, face_embedding_model):
    # Detect faces
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]

            # Generate face embedding
            blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            face_embedding_model.setInput(blob)
            return face_embedding_model.forward().flatten()

    return None

# Step 4: Train a classifier using face embeddings
def train_classifier(embeddings, labels):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, labels)
    return knn

# Step 5: Perform face recognition on live video
def recognize_faces(knn, face_detector, face_embedding_model):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        embedding = get_face_embedding(frame, face_detector, face_embedding_model)
        if embedding is not None:
            label = knn.predict([embedding])[0]
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    dataset_path = "./face_dataset"  # Path to your dataset folder

    global face_detector, face_embedding_model
    face_detector, face_embedding_model = load_models()

    print("Loading dataset...")
    embeddings, labels = load_dataset(dataset_path)

    print("Training classifier...")
    knn = train_classifier(embeddings, labels)

    print("Starting face recognition...")
    recognize_faces(knn, face_detector, face_embedding_model)

# Run the project
if __name__ == "__main__":
    main()
