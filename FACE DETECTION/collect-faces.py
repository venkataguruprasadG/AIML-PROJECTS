import cv2
import os

# Ask for name
person_name = input("Enter the name of the person: ").strip().lower().replace(" ", "_")
save_dir = os.path.join("data", person_name)
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

count = 0
print(f"[INFO] Collecting images for {person_name}. Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        file_path = os.path.join(save_dir, f"{person_name}_{count}.jpg")
        cv2.imwrite(file_path, face)

        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_name}_{count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Collecting Faces", frame)

    if count >= 50 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"[INFO] Collected {count} face images for {person_name}")
cap.release()
cv2.destroyAllWindows()
