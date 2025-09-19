from collect_faces import collect_faces
from train_model import train_model
from recognize import recognize_faces

if __name__ == "__main__":
    # Step 1: Collect data
    collect_faces()
    # Step 2: Train model
    train_model()
    # Step 3: Run recognition
    recognize_faces()
