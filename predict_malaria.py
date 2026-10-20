"""Classify one cell image with a model produced by train_malaria_classifier.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a microscopy cell image")
    parser.add_argument("--model", type=Path, default=Path("outputs/best_model.keras"))
    parser.add_argument("--model-info", type=Path, default=Path("outputs/model_info.json"))
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.model.is_file() or not args.model_info.is_file():
        parser.error("Train the model first, or pass valid --model and --model-info paths.")

    info = json.loads(args.model_info.read_text(encoding="utf-8"))
    threshold = info["decision_threshold"] if args.threshold is None else args.threshold
    image_size = int(info["image_size"])
    image = tf.keras.utils.load_img(args.image, target_size=(image_size, image_size))
    array = tf.keras.utils.img_to_array(image)[None, ...]
    probability = float(tf.keras.models.load_model(args.model).predict(array, verbose=0)[0][0])
    predicted_class = info["positive_class"] if probability >= threshold else info["negative_class"]

    print(f"Prediction: {predicted_class}")
    print(f"Estimated parasitized probability: {probability:.4f}")
    print(f"Decision threshold: {threshold:.2f}")
    print("Educational result only; this is not a medical diagnosis.")


if __name__ == "__main__":
    main()
