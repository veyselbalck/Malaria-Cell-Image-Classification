# Medical Image Analysis Projects

An educational computer vision repository containing two independent experiments from a multimedia computing course:

1. **Malaria cell classification** — a TensorFlow/Keras CNN classifies individual red blood cell images as parasitized or uninfected.
2. **Mucosal image detection dataset** — a Roboflow YOLOv7-format export with train/validation/test images and bounding-box labels. It is kept as a separate dataset because its images and annotations do not belong to the malaria cell classifier.

These experiments use different image types, labels, and modeling tasks. The malaria training script does not read the YOLO dataset.

## Malaria cell classification

The notebook in [`malaria_cell_images_classification.ipynb`](malaria_cell_images_classification.ipynb) walks through the runnable pipeline. The training and prediction code is also available in [`train_malaria_classifier.py`](train_malaria_classifier.py) and [`predict_malaria.py`](predict_malaria.py). It uses the public Malaria dataset distributed through TensorFlow Datasets; the first run downloads the dataset automatically.

### Setup

Python 3.10 or 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Train and evaluate

```bash
python train_malaria_classifier.py --epochs 20 --batch-size 128
```

The script creates `outputs/best_model.keras`, `outputs/metrics.json`, and `outputs/model_info.json`. The first run needs an internet connection to download the dataset. Training can run on CPU, though it will be slower than on a supported GPU.

### Predict one cell image

```bash
python predict_malaria.py path/to/cell.png
```

The model reports the predicted class and its estimated parasitized probability. The default decision threshold is 0.5; adjust it with `--threshold` if needed.

## Mucosal image detection dataset

The [`mucosa-yolo-dataset/`](mucosa-yolo-dataset/) directory contains 690 images and matching YOLO text labels: 585 train, 45 validation, and 60 test. Its dataset-specific README describes the export, license, structure, and current annotation uncertainties.

The image filenames include `positive` and `neg`, but the exported YAML defines three classes (`-1`, `0`, `1`). Their exact relationship to healthy/diseased tissue has not been verified. Treat the dataset as an archived export until its class mapping is confirmed; do not use it to train the malaria classifier or infer a medical diagnosis.

## Previous notebook result

The archived original notebook at [`notebooks/archive/original_colab_experiment.ipynb`](notebooks/archive/original_colab_experiment.ipynb) contains a historical run that reached **96.14% validation accuracy** after 20 epochs. This is a validation result from a random image-level split, not an independent test result or evidence of clinical performance. That experiment does not report an independent test set, sensitivity, specificity, or patient-level evaluation. Run the script above to obtain fresh metrics; do not compare results across pipelines without matching the split and evaluation method.

## Data and license notes

- The malaria cell images come from the [Malaria Cell Images Dataset on Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria), accessed through TensorFlow Datasets. Review the dataset source terms before reuse.
- The separate mucosal dataset was exported from [Roboflow Universe – odev2 v1](https://universe.roboflow.com/dev-v4x4g/odev2/dataset/1). Its export metadata states **CC BY 4.0**; attribution is included in the dataset notes.
- No trained model weights are committed. Training outputs are local artifacts and are excluded from Git.
- The source code does not yet declare a reuse license. The mucosal dataset's CC BY 4.0 license applies to that dataset, not automatically to the code or malaria images.
- This repository is for education and research. It is not a medical device or diagnostic tool.

## Türkçe özet

Bu depoda iki ayrı çalışma bulunur: sıtmalı hücre görüntülerini sınıflandıran CNN kodu ve mukoza görüntülerine ait bağımsız bir YOLOv7 veri seti. İki veri seti aynı modelde kullanılmaz. Sıtma modeli `python train_malaria_classifier.py` komutuyla eğitilir; ilk çalıştırmada veri TensorFlow Datasets üzerinden indirilir. Mukoza verisinin sınıf eşleşmesi doğrulanmadığı için tanı veya model başarısı iddiası olarak yorumlanmamalıdır.
