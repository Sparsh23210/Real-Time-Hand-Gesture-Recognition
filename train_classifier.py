from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DATA_FILE = Path("data.pickle")
MODEL_FILE = Path("model.p")


def main():
    if not DATA_FILE.is_file():
        print(
            "Error: Dataset file 'data.pickle' was not found. "
            "Run create_dataset.py first."
        )
        return 1

    try:
        with DATA_FILE.open("rb") as file:
            data_dict = pickle.load(file)
        data = np.asarray(data_dict["data"])
        labels = np.asarray(data_dict["labels"])
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, TypeError) as error:
        print(f"Error: Could not load a valid dataset from {DATA_FILE}: {error}")
        return 1

    if data.ndim != 2 or data.shape[1] != 42 or len(data) != len(labels):
        print("Error: The dataset must contain one label per 42-feature sample.")
        return 1

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        shuffle=True,
        stratify=labels,
        random_state=42,
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    print(f"Held-out test accuracy: {score * 100:.1f}%")

    with MODEL_FILE.open("wb") as file:
        pickle.dump({"model": model}, file)

    print(f"Saved trained model to {MODEL_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
