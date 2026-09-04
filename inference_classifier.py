from pathlib import Path
import pickle

import cv2
import mediapipe as mp
import numpy as np


MODEL_FILE = Path("model.p")
CAMERA_INDEX = 0
LABELS = {
    0: "OK",
    1: "PEACE",
    2: "I LOVE YOU",
    3: "HI",
    4: "HIT ME UP",
}


def load_model():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            "Model file 'model.p' was not found. Restore the supplied model or "
            "run create_dataset.py and train_classifier.py."
        )

    try:
        with MODEL_FILE.open("rb") as file:
            model_dict = pickle.load(file)
        return model_dict["model"]
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, TypeError) as error:
        raise RuntimeError(f"Could not load a valid model from {MODEL_FILE}: {error}") from error


def main():
    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        print(f"Error: Could not open webcam at camera index {CAMERA_INDEX}.")
        return 1

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not capture a frame from the webcam.")
                return 1

            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

                x_values = [landmark.x for landmark in hand_landmarks.landmark]
                y_values = [landmark.y for landmark in hand_landmarks.landmark]
                min_x = min(x_values)
                min_y = min(y_values)

                features = []
                for landmark in hand_landmarks.landmark:
                    features.extend((landmark.x - min_x, landmark.y - min_y))

                if len(features) != 42:
                    print(f"Error: Expected 42 hand features, but found {len(features)}.")
                    return 1

                try:
                    prediction = model.predict(np.asarray([features]))
                    predicted_gesture = LABELS[int(prediction[0])]
                except (ValueError, KeyError, TypeError) as error:
                    print(f"Error: Model prediction failed: {error}")
                    return 1

                x1 = max(0, int(min(x_values) * width) - 10)
                y1 = max(0, int(min(y_values) * height) - 10)
                x2 = min(width - 1, int(max(x_values) * width) + 10)
                y2 = min(height - 1, int(max(y_values) * height) + 10)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.putText(
                    frame,
                    predicted_gesture,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Webcam inference stopped.")
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
