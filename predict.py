import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('model/waste_model.h5')

# Class names (adjust order if different)
CLASSES = ['biodegradable', 'e-waste', 'glass', 'metal', 'paper', 'plastic']


# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press SPACE to capture and predict. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the live video feed
    cv2.imshow("Waste Classification - Press SPACE to predict", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break
    elif key == 32:  # SPACE to capture and predict
        # Preprocess the captured frame
        image = cv2.resize(frame, (224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Predict
        preds = model.predict(image)[0]
        class_index = np.argmax(preds)
        label = f"{CLASSES[class_index]} ({preds[class_index]*100:.2f}%)"

        print("Prediction:", label)

        # Display prediction on the frame
        output_frame = frame.copy()
        cv2.putText(output_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", output_frame)
        cv2.waitKey(3000)  # Wait 3 seconds to show prediction

cap.release()
cv2.destroyAllWindows()
