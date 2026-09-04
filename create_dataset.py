from pathlib import Path
import pickle

import cv2
import mediapipe as mp


DATA_DIR = Path("data")
DATA_FILE = Path("data.pickle")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not DATA_DIR.is_dir():
        print(
            "Error: Dataset directory 'data' was not found. "
            "Run collect_imgs.py first."
        )
        return 1

    class_dirs = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())
    if not class_dirs:
        print("Error: Dataset directory 'data' contains no class folders.")
        return 1

    data = []
    labels = []
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )

    try:
        for class_dir in class_dirs:
            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Warning: Could not read image {image_path}; skipping it.")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                if not results.multi_hand_landmarks:
                    continue

                landmarks = results.multi_hand_landmarks[0].landmark
                x_values = [landmark.x for landmark in landmarks]
                y_values = [landmark.y for landmark in landmarks]
                min_x = min(x_values)
                min_y = min(y_values)

                features = []
                for landmark in landmarks:
                    features.extend((landmark.x - min_x, landmark.y - min_y))

                if len(features) != 42:
                    print(
                        f"Warning: Expected 42 features for {image_path}, "
                        f"but found {len(features)}; skipping it."
                    )
                    continue

                data.append(features)
                labels.append(class_dir.name)
    finally:
        hands.close()

    if not data:
        print("Error: No hands were detected in the dataset images.")
        return 1

    with DATA_FILE.open("wb") as file:
        pickle.dump({"data": data, "labels": labels}, file)

    print(f"Created {DATA_FILE} with {len(data)} samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
