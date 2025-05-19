import cv2
from deepface import DeepFace

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_COLOR = (0, 255, 255)
BOX_COLOR = (0, 128, 255)
LINE_THICKNESS = 2

def detect_faces(gray_frame):
    return face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80))

def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception as e:
        print(f"[Error] DeepFace failed: {e}")
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("🔍 Starting Emotion Detection — Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces(gray)

        for (x, y, w, h) in faces:
            face_img = rgb[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (224, 224))  # DeepFace input size
            emotion = analyze_emotion(face_img_resized)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR, LINE_THICKNESS)
            cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), FONT, FONT_SCALE, FONT_COLOR, 2)
            print(f" Detected Emotion: {emotion}")

        cv2.imshow(" Real-time Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(" Exiting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
