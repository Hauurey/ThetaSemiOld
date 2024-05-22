import cv2
import numpy as np
import os
from PIL import Image

# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the face detection classifier
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Input ID and Name for face labeling
id = input("Enter ID: ")
name = input("Enter Name: ")

# Create the data directory if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Save ID and Name to id_name.txt inside the data directory, each pair on a new line
with open(os.path.join(data_dir, "id_name.txt"), "a") as file:
    file.write(f"{id},{name}\n")

# Initialize counter for collected images
count = 0

# Maximum number of images to collect
max_count = 500

while count < max_count:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Increment image count
        count += 1
        
        # Save the detected face region
        cv2.imwrite(r'C:\Users\user\Desktop\Theta\datasets\User.' + str(id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w])
        
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed or desired number of images is collected
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_count:
        break

# Release the video capture
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Dataset Collection Done")

# Function to get image IDs, names, and faces
def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    names = []
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        faceNP = np.array(faceImage)
        filename = os.path.split(imagePaths)[-1]
        Id = int(filename.split(".")[1])
        name = filename.split(".")[2]
        faces.append(faceNP)
        ids.append(Id)
        names.append(name)
        cv2.imshow("Training", faceNP)
        cv2.waitKey(1)
    return ids, names, faces

# Path to the dataset directory
path = r'C:\Users\user\Desktop\Theta\datasets'

# Get image IDs, names, and faces
IDs, Names, facedata = getImageID(path)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
recognizer.train(facedata, np.array(IDs))

# Save the trained model
recognizer.write(r'C:\Users\user\Desktop\Theta\Trainer.yml')

# Close OpenCV windows
cv2.destroyAllWindows()

print("Training Completed")
