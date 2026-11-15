# Face Recognition

A webcam demo that matches detected faces against reference images and asks recognized people to blink.

## Setup

Install the Python dependencies:

```sh
python3 -m pip install -r requirements.txt
```

The first run downloads InsightFace's `buffalo_l` model pack. The pack is cached under `~/.insightface/models`. See the [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) for model details and current terms; its pretrained packs are provided for non-commercial research use.

## Reference images

Put a `.jpg`, `.jpeg`, or `.png` image for each person in `known_faces/`. The image filename (without its extension) is used as the displayed name. If an image contains multiple faces, the largest detected face is used.

## Run

```sh
python3 face_recognizer.py
```

Use `--camera 1` to select another webcam, `--cpu` to force CPU inference, or `--threshold 0.55` to change the face-match cosine-similarity threshold. The video overlay shows the measured eye aspect ratio (EAR) and the current closed-eye cutoff. If a blink is missed, try a higher closure ratio, such as `--eye-closure-ratio 0.8`. Press `q` in the video window to quit.

The blink indicator is a simple eye-landmark heuristic. It is useful as a demo prompt, but it does not provide secure liveness or spoof protection.
