import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import pygame 

def resource_path(relative_path):
    """ Get the absolute path to resource, handling development and PyInstaller environments """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

print("Loading AI Brain...")
model = tf.keras.models.load_model(resource_path('drowsiness_model.h5'))

# --- AUDIO SETUP ---
pygame.mixer.init()
pygame.mixer.music.load(resource_path('AlarmSound.mp3')) 
is_alarm_playing = False
# ----------------------

face_cascade = cv2.CascadeClassifier(resource_path('haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(resource_path('haarcascade_eye.xml'))

cap = cv2.VideoCapture(0)

# Keep the lightweight resolution framework to stop pixel lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sleepy_frames = 0
ALARM_THRESHOLD = 5 
frame_counter = 0

# These variables hold the state so the screen doesn't flicker
display_label = "WAITING..."
display_color = (0, 255, 0)
x, y = 50, 50 # Default text position

print("\n🚨 BALANCED LIVE FEED TURNING ON! 🚨")
print("--> Click on the video window that pops up, and press 'q' on your keyboard to stop it.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    frame_counter += 1
    
    # Restored to your original 3rd frame calculation loop
    if frame_counter % 3 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # RESTORED: Original classic thresholds to ensure closed eyes aren't ignored
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # If no face is found at all, we assume the person fell completely over or head dropped
        if len(faces) == 0:
            sleepy_frames += 1

        for (fx, fy, w, h) in faces:
            x, y = fx, fy # Update text position to track the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # RESTORED: Original eye cascade thresholds
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(eyes) == 0:
                # The camera lost the eye (crucial indicator for a closed eye!)
                sleepy_frames += 1
            else:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                    resized_eye = cv2.resize(eye_roi, (224, 224))
                    
                    # Matches your training dataset scaling
                    img_array = tf.cast(tf.expand_dims(resized_eye, 0), tf.float32)
                    
                    prediction = model.predict(img_array, verbose=0)
                    score = prediction[0][0] 
                    
                    if score > 0.5: 
                        sleepy_frames += 1
                    else:
                        sleepy_frames = 0 # Eyes are open, reset the alarm timer!
                        
                    break # Process one eye to save CPU speed
                    
    # --- VISUAL STABILITY BUFFER ---
    if sleepy_frames >= 2:
        display_label = "SLEEPY"
        display_color = (0, 0, 255)
    else:
        display_label = "AWAKE"
        display_color = (0, 255, 0)
        
    # --- ALARM LOGIC ---
    if sleepy_frames >= ALARM_THRESHOLD:
        cv2.putText(frame, "ALARM: WAKE UP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if not is_alarm_playing:
            pygame.mixer.music.play(-1) 
            is_alarm_playing = True
    else:
        if is_alarm_playing:
            pygame.mixer.music.stop()
            is_alarm_playing = False
            
    cv2.putText(frame, display_label, (x, max(30, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2)
    cv2.imshow('Drowsiness Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
print("\nCamera safely turned off.")