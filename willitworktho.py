import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

def load_model():
    # Load model from JSON file
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights into the model
    model.load_weights('model.weights.h5')
    print("Model loaded successfully")
    return model

def start_webcam():
    # Load the pre-trained model
    model = load_model()

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start webcam
    cap = cv2.VideoCapture(0)

    # Emotion labels
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    print("Starting webcam... Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # Create overlay window
        sub_img = frame[0:int(height / 6), 0:int(width)]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)

        # Add title
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.8
        FONT_THICKNESS = 2
        label_color = (10, 10, 255)

        title = "Emotion Detection"
        title_dimensions = cv2.getTextSize(title, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        textX = int((res.shape[1] - title_dimensions[0]) / 2)
        textY = int((res.shape[0] + title_dimensions[1]) / 2)
        cv2.putText(res, title, (textX, textY), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        try:
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract and preprocess face region
                roi_gray = gray[y - 5:y + h + 5, x - 5:x + w + 5]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = roi_gray.astype('float') / 255.0

                # Make prediction
                prediction = model.predict(roi_gray)
                emotion_idx = np.argmax(prediction[0])
                emotion = emotions[emotion_idx]
                confidence = prediction[0][emotion_idx] * 100

                # Display results
                cv2.putText(res, f"Emotion: {emotion}", (0, textY + 22 + 5), FONT, 0.7, label_color, 2)
                cv2.putText(res, f"Confidence: {confidence:.1f}%",
                            (width - 180, textY + 22 + 5), FONT, 0.7, label_color, 2)

        except Exception as e:
            print(f"Error: {str(e)}")
            pass

        # Add the overlay to the main frame
        frame[0:int(height / 6), 0:int(width)] = res

        # Display the frame
        cv2.imshow('Facial Emotion Recognition', frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_webcam() 