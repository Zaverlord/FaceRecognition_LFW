import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# ------------------------------
# Helper function to process known faces
# ------------------------------
def process_known_faces(lfw_people):
    """
    Process the LFW dataset to extract face encodings and names.
    """
    known_face_encodings = []
    known_face_names = []
    
    for i, image in enumerate(lfw_people.images):
        image_uint8 = image.astype(np.uint8)
        rgb_image = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        
        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(lfw_people.target_names[lfw_people.target[i]])
    
    return known_face_encodings, known_face_names

# ------------------------------
# Main function for face recognition
# ------------------------------
def recognize_faces(input_image_path, tolerance=0.6):
    """
    Perform face recognition and display landmarks on the input image.
    """
    print("Downloading the LFW dataset...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    print("Processing known faces...")
    known_face_encodings, known_face_names = process_known_faces(lfw_people)
    
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: Unable to load the image from {input_image_path}")
        return
    
    rgb_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_input)
    face_landmarks_list = face_recognition.face_landmarks(rgb_input)
    face_encodings = face_recognition.face_encodings(rgb_input, face_locations)
    
    for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        name = "Unknown"
        if any(matches):
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        cv2.rectangle(rgb_input, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(rgb_input, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        
        for feature, points in landmarks.items():
            for point in points:
                cv2.circle(rgb_input, point, 2, (255, 255, 255), -1)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_input)
    plt.axis("off")
    plt.show()

# Example usage:
recognize_faces("known_faces\person_trump.jpg")
