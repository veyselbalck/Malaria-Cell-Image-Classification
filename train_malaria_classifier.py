"""Train a small binary malaria-cell classifier on the TFDS malaria dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def make_model(image_size: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="malaria_cell_cnn",
    )


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits, info = tfds.load(
        "malaria",
        split={
            "train": "train[:80%]",
            "validation": "train[80%:90%]",
            "test": "train[90%:]",
        },
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )
    label_names = info.features["label"].names
    positive_index = next(
        (index for index, name in enumerate(label_names) if "parasitized" in name.lower()),
        None,
    )
    if positive_index is None:
        raise ValueError(f"Could not find the parasitized class in TFDS labels: {label_names}")

    image_size = (args.image_size, args.image_size)
    autotune = tf.data.AUTOTUNE

    def prepare(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.image.resize(image, image_size)
        label = tf.cast(tf.equal(label, positive_index), tf.float32)
        return image, label

    datasets: dict[str, tf.data.Dataset] = {}
    for split_name, dataset in splits.items():
        dataset = dataset.map(prepare, num_parallel_calls=autotune)
        if split_name == "train":
            dataset = dataset.shuffle(4096, seed=args.seed, reshuffle_each_iteration=True)
        datasets[split_name] = dataset.batch(args.batch_size).prefetch(autotune)

    model = make_model(args.image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    model_path = args.output_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_auc", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
    ]
    model.fit(
        datasets["train"],
        validation_data=datasets["validation"],
        epochs=args.epochs,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(model_path)
    results = best_model.evaluate(datasets["test"], return_dict=True, verbose=0)
    probabilities = best_model.predict(datasets["test"], verbose=0).reshape(-1)
    true_labels = np.concatenate(
        [labels.numpy().reshape(-1) for _, labels in datasets["test"]]
    ).astype(np.int32)
    predicted_labels = (probabilities >= 0.5).astype(np.int32)
    tp = int(np.sum((true_labels == 1) & (predicted_labels == 1)))
    tn = int(np.sum((true_labels == 0) & (predicted_labels == 0)))
    fp = int(np.sum((true_labels == 0) & (predicted_labels == 1)))
    fn = int(np.sum((true_labels == 1) & (predicted_labels == 0)))
    results["confusion_matrix"] = {"true_negative": tn, "false_positive": fp,
                                   "false_negative": fn, "true_positive": tp}

    (args.output_dir / "metrics.json").write_text(
        json.dumps({key: value for key, value in results.items()}, indent=2),
        encoding="utf-8",
    )
    model_info = {
        "image_size": args.image_size,
        "positive_class": "Parasitized",
        "negative_class": "Uninfected",
        "positive_tfds_label": label_names[positive_index],
        "decision_threshold": 0.5,
        "seed": args.seed,
        "split": "TFDS train: 80% train, 10% validation, 10% test (image-level)",
    }
    (args.output_dir / "model_info.json").write_text(
        json.dumps(model_info, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"Saved model and metadata under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
