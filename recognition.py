import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
from PIL import Image
import threading
import queue

# Email configuration
Sender_Email = "202110035@fit.edu.ph"
Reciever_Email = "anonuevo.harry.s@gmail.com"
Password = "uppy slbu ksai ocgy"  # Use app-specific password if using Gmail

# Thread-safe queue to manage image tasks
image_queue = queue.Queue()
new_image_event = threading.Event()

# Dictionary to track last print times
last_print_times = {
    "motion": datetime.min,
    "face": datetime.min,
    "unknown_face": datetime.min,
}

# Minimum time between prints for each event type
print_interval = timedelta(seconds=10)

# Create a directory for logs
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Function to get the log file path for the current date
def get_log_file_path():
    current_date = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(log_dir, f'{current_date}.txt')

# Open the initial log file
log_file_path = get_log_file_path()
log_file = open(log_file_path, 'a')

def log_message(message):
    global log_file, log_file_path
    current_date = datetime.now().strftime('%m-%d-%Y')
    new_log_file_path = os.path.join(log_dir, f'{current_date}.txt')
    # Check if we need to switch to a new log file
    if new_log_file_path != log_file_path:
        log_file.close()
        log_file_path = new_log_file_path
        log_file = open(log_file_path, 'a')
    current_time = datetime.now().strftime('%H:%M')
    log_entry = f"[{current_time}]: {message}\n"
    log_file.write(log_entry)
    log_file.flush()  # Ensure the message is written immediately to the file

def send_email_with_image(image_path):
    newMessage = EmailMessage()
    newMessage['Subject'] = "Visitor Image"
    newMessage['From'] = Sender_Email
    newMessage['To'] = Reciever_Email
    newMessage.set_content('Let me know what you think. Image attached!')

    with open(image_path, 'rb') as f:
        image_data = f.read()
        image = Image.open(f)
        image_type = image.format.lower()
        image_name = os.path.basename(image_path)

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, Password)
        smtp.send_message(newMessage)

def handle_image_tasks():
    last_image_path = None
    while True:
        face_img, date_folder, current_time = image_queue.get()
        if face_img is None:
            break
        try:
            # Save the unknown face
            face_filename = os.path.join(date_folder, f"unknown_face_{current_time}.jpg")
            cv2.imwrite(face_filename, face_img)
            last_image_path = face_filename  # Keep track of the last saved image path
        except Exception as e:
            log_message(f"Failed to process image task: {e}")
        finally:
            image_queue.task_done()

    # After processing all images, send the last saved image
    if last_image_path:
        try:
            send_email_with_image(last_image_path)
        except Exception as e:
            log_message(f"Failed to send email with image: {e}")

# Create a base folder for storing unknown faces
unknown_faces_dir = 'unknown_faces'
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

# Create the data directory if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Load ID and Name from file
with open(os.path.join(data_dir, "id_name.txt"), "r") as f:
    id, name = f.readline().strip().split(',')

# Initialize video capture
video = cv2.VideoCapture(0)

# Initialize MOG2 background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Load face detection classifier
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'C:\Users\user\Desktop\Theta\Trainer.yml')

# Start the worker thread to handle image tasks
worker_thread = threading.Thread(target=handle_image_tasks)
worker_thread.start()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Apply MOG2 background subtraction
    fgMask = backSub.apply(frame)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the foreground mask
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Count non-zero pixels in the foreground mask
    nonZeroCount = cv2.countNonZero(fgMask)

    # If non-zero count exceeds threshold, log "Motion Detected"
    if nonZeroCount > 5000:  # Adjust this threshold as needed
        current_time = datetime.now()
        if current_time - last_print_times["motion"] > print_interval:
            log_message("Motion Detected")
            last_print_times["motion"] = current_time

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        text = f"Unknown - Confidence: {round(100 - conf, 2)}%"
        if conf <= 80:
            text = f"{name} - Confidence: {round(100 - conf, 2)}%"
            current_time = datetime.now()
            if current_time - last_print_times["face"] > print_interval:
                log_message(f"Face Detected: {name}")
                last_print_times["face"] = current_time
        else:
            # Create a date-based folder
            current_date = datetime.now().strftime('%Y-%m-%d')
            date_folder = os.path.join(unknown_faces_dir, current_date)
            if not os.path.exists(date_folder):
                os.makedirs(date_folder)

            # Save the unknown face and signal the worker thread
            face_img = frame[y:y+h, x:x+w]
            image_queue.put((face_img, date_folder, current_time.strftime('%m-%d-%Y_%H-%M')))
            new_image_event.set()
            current_time = datetime.now()
            if current_time - last_print_times["unknown_face"] > print_interval:
                log_message("Face Detected: Unknown")
                last_print_times["unknown_face"] = current_time

        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    frame = cv2.resize(frame, (640, 480))
    fgMask = cv2.resize(fgMask, (640, 480))

    # Display frames in separate windows
    cv2.imshow("Original Frame", frame)
    #cv2.imshow("Foreground Mask", fgMask)

    k = cv2.waitKey(1)
    if k == ord("q"):
        break

# Signal the worker thread to exit and wait for it to finish
image_queue.put((None, None, None))
worker_thread.join()

# Release video capture
video.release()
cv2.destroyAllWindows()

# Close the log file
log_file.close()
