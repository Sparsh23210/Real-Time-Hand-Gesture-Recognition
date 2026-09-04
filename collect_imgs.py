from pathlib import Path

import cv2


DATA_DIR = Path("data")
CAMERA_INDEX = 0
NUMBER_OF_CLASSES = 5
DATASET_SIZE = 100


def main():
    DATA_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        print(f"Error: Could not open webcam at camera index {CAMERA_INDEX}.")
        return 1

    try:
        for class_index in range(NUMBER_OF_CLASSES):
            class_dir = DATA_DIR / str(class_index)
            class_dir.mkdir(exist_ok=True)

            print(f"Collecting data for class {class_index}")
            while True:
                success, frame = cap.read()
                if not success:
                    print("Error: Could not capture a frame from the webcam.")
                    return 1

                cv2.putText(
                    frame,
                    'Ready? Press "q" to begin.',
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.imshow("frame", frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            for image_index in range(DATASET_SIZE):
                success, frame = cap.read()
                if not success:
                    print("Error: Could not capture a frame from the webcam.")
                    return 1

                cv2.imshow("frame", frame)
                cv2.waitKey(25)

                file_path = class_dir / f"{image_index}.jpg"
                if not cv2.imwrite(str(file_path), frame):
                    print(f"Error: Could not write image to {file_path}.")
                    return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
