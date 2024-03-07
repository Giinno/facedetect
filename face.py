import cv2

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture using webcam (device index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale (optional, but recommended for Haar cascades)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame for faster processing (optional)
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw bounding boxes around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Live Face Detection', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt detected.")

finally:
    # Release capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    print('Live face detection terminated!')
