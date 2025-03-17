import cv2
from fer import FER

# Initialize the emotion detector
detector = FER()

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect emotions in the frame
    emotions = detector.detect_emotions(frame)

    # If emotions are detected, get the dominant emotion
    if emotions:
        # The first detected face
        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        score = emotions[0]['emotions'][dominant_emotion]

        # Draw rectangle around face
        (x, y, w, h) = emotions[0]["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the emotion on the frame
        cv2.putText(frame, f"{dominant_emotion} ({score:.2f})", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
