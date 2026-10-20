# Mucosal Image Detection Dataset

This directory contains a YOLO-format image dataset that was found alongside the malaria-cell classification notebook. It is included as a separate dataset component in the same course repository; **it is not malaria-cell data and is not used by the malaria classifier**.

## Contents

| Split | Images | Label files |
|---|---:|---:|
| Train | 585 | 585 |
| Validation | 45 | 45 |
| Test | 60 | 60 |
| **Total** | **690** | **690** |

The images are 416 × 416 JPEGs. Each text label uses YOLO bounding-box format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image dimensions. `data.yaml` points to the local `train/images`, `valid/images`, and `test/images` folders.

## Class mapping needs confirmation

The image filenames contain `positive` or `neg`, consistent with the project owner's description of diseased and healthy mucosal images. However, the Roboflow export lists three class names (`-1`, `0`, `1`), and the label files use numeric IDs `0`, `1`, and `2`. Negative-named images contain IDs 0 and 1, while positive-named images contain ID 2. The source files do not explain those labels. Confirm the intended class meanings and correct the class names before training or interpreting predictions.

Because of that ambiguity, this repository does not claim that the labels are a validated disease annotation set, and the malaria classifier must not be trained on these images.

## Source and attribution

Exported from [Roboflow Universe – odev2 v1](https://universe.roboflow.com/dev-v4x4g/odev2/dataset/1). The export metadata states a **CC BY 4.0** license. Attribute the original dataset source when redistributing it and check the source page for current terms. The provided `README.roboflow.txt` and `README.dataset.txt` are retained as provenance.

The export notes say images were resized to 416 × 416 and that training images received augmentation. No preprocessing or training code for this dataset is included here.

## Optional YOLOv7 training

After confirming the class meanings, clone the [official YOLOv7 repository](https://github.com/WongKinYiu/yolov7), follow its dependency and pretrained-weight setup, and run this from the YOLOv7 checkout (adjust the dataset path for your machine):

```bash
python train.py --data /path/to/mucosa-yolo-dataset/data.yaml --cfg cfg/training/yolov7-tiny.yaml --weights yolov7-tiny.pt --img-size 416 416 --batch-size 16 --epochs 50
```

This trains the three numeric classes as exported. It does not resolve their medical meaning or validate the model for clinical use.
