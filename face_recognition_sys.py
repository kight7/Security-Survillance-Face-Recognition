import cv2
import os
from datetime import datetime
from deepface import DeepFace
import shutil
import time

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
        self.refresh_known_faces()

    def refresh_known_faces(self):
        self.known_face_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.known_faces_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def add_new_face(self, image_path, name):
        # Save the image to the known_faces directory with the person's name
        ext = os.path.splitext(image_path)[1]
        dest_path = os.path.join(self.known_faces_dir, f"{name}{ext}")
        shutil.copy(image_path, dest_path)
        self.refresh_known_faces()
        print(f"Added {name} to known faces")
        return True

    def recognize_faces_in_frame(self, frame):
        # Save the current frame temporarily
        temp_img = "temp_frame.jpg"
        cv2.imwrite(temp_img, frame)
        try:
            # Use DeepFace to find the face in the known_faces_dir
            result = DeepFace.find(img_path=temp_img, db_path=self.known_faces_dir, enforce_detection=False)
            if len(result) > 0 and len(result[0]) > 0:
                # Get the best match
                best_match = result[0].iloc[0]
                name = os.path.splitext(os.path.basename(best_match['identity']))[0]
                confidence = 1 - best_match['distance']  # Lower distance = better match
            else:
                name = "Unknown"
                confidence = 0.0
        except Exception as e:
            name = "Unknown"
            confidence = 0.0

        # Print result in terminal
        print(f"Detected: {name} (Confidence: {confidence:.2f})")

        # Draw label on the frame
        label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
        # Clean up temp image
        if os.path.exists(temp_img):
            os.remove(temp_img)
        return frame

    def start_realtime_recognition(self):
        print("Starting real-time face recognition...")
        print("Press 'q' to quit, 's' to save current frame")

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return

        start_time = time.time()
        last_result = None

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = self.recognize_faces_in_frame(frame)
            cv2.imshow('Face Recognition', frame)

            # Optionally, store the last result for returning
            last_result = frame

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

            # Stop after 10 seconds
            if time.time() - start_time > 10:
                print("Time limit reached (10 seconds).")
                break

        video_capture.release()
        cv2.destroyAllWindows()
        print("Face recognition stopped")
        return last_result

def main():
    face_system = FaceRecognitionSystem()
    if len(face_system.known_face_names) == 0:
        print("No known faces found!")
        print("Please add some images to the 'known_faces' directory.")
        print("Image files should be named with the person's name (e.g., 'john_doe.jpg')")
        return

    print(f"Loaded {len(face_system.known_face_names)} known faces:")
    for name in face_system.known_face_names:
        print(f"  - {name}")

    face_system.start_realtime_recognition()

if __name__ == "__main__":
    main()