"""Recognize faces from a webcam and ask recognized users to blink."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import dlib
import insightface
import numpy as np
import onnxruntime as ort
from imutils import face_utils
from numpy.linalg import norm


BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
PREDICTOR_PATH = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"
FACE_MODEL_NAME = "buffalo_l"
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_EYE_CLOSURE_RATIO = 0.72
MIN_EYE_ASPECT_RATIO = 0.08
MIN_CLOSED_EYE_FRAMES = 1
MAX_TRACK_MISSED_FRAMES = 3


@dataclass
class FaceTrack:
    bbox: np.ndarray
    last_seen: int
    closed_eye_frames: int = 0
    open_eye_baseline: Optional[float] = None
    has_seen_open_eyes: bool = False
    blink_confirmed: bool = False


def eye_aspect_ratio(eye: np.ndarray) -> Optional[float]:
    """Return the eye aspect ratio, or None for degenerate landmarks."""
    vertical_a = norm(eye[1] - eye[5])
    vertical_b = norm(eye[2] - eye[4])
    horizontal = norm(eye[0] - eye[3])
    if horizontal <= 1e-6:
        return None
    return float((vertical_a + vertical_b) / (2.0 * horizontal))


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate cosine similarity without dividing by a zero norm."""
    first_norm = norm(first)
    second_norm = norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return -1.0
    similarity = float(np.dot(first, second) / (first_norm * second_norm))
    if not np.isfinite(similarity):
        return -1.0
    return float(np.clip(similarity, -1.0, 1.0))


def largest_face(faces):
    """Choose the largest detected face in an image."""
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: max(0.0, face.bbox[2] - face.bbox[0])
        * max(0.0, face.bbox[3] - face.bbox[1]),
    )


def create_face_model(force_cpu: bool = False):
    """Load InsightFace, retrying on CPU if GPU initialization fails."""
    model = insightface.app.FaceAnalysis(name=FACE_MODEL_NAME)
    if force_cpu or "CUDAExecutionProvider" not in ort.get_available_providers():
        model.prepare(ctx_id=-1)
        print("Using CPU for face analysis.")
        return model

    try:
        model.prepare(ctx_id=0)
        print("Using the default GPU context for face analysis.")
    except Exception as gpu_error:
        print("GPU initialization failed; retrying face analysis on CPU:", gpu_error)
        try:
            model.prepare(ctx_id=-1)
        except Exception as cpu_error:
            raise RuntimeError(
                "Could not initialize InsightFace on either GPU or CPU. "
                "Check the InsightFace and ONNX Runtime installation."
            ) from cpu_error
        print("Using CPU for face analysis.")
    return model


def load_known_faces(model, folder: Path) -> dict:
    """Load one (largest) face embedding for each image in ``folder``."""
    if not folder.is_dir():
        print("Known-faces folder is missing:", folder)
        return {}

    known_faces = {}
    for image_path in sorted(folder.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png",
        ):
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print("Warning: Could not read image:", image_path.name)
            continue

        try:
            face = largest_face(model.get(image))
        except Exception as error:
            print("Warning: Could not analyze", image_path.name, "-", error)
            continue

        if face is None:
            print("Warning: No face found in", image_path.name)
            continue

        embedding = np.asarray(face.embedding, dtype=np.float32)
        if embedding.size == 0 or not np.all(np.isfinite(embedding)) or norm(embedding) <= 1e-12:
            print("Warning: Invalid face embedding in", image_path.name)
            continue

        name = image_path.stem
        if name in known_faces:
            print("Warning: Duplicate name; skipping", image_path.name)
            continue
        known_faces[name] = embedding
        print("Loaded face for:", name)

    return known_faces


def recognize_face(embedding: np.ndarray, known_faces: dict, threshold: float) -> Tuple[Optional[str], float]:
    """Return the best matching name if its similarity meets the threshold."""
    best_name = None
    best_score = -1.0
    for name, reference_embedding in known_faces.items():
        score = cosine_similarity(embedding, reference_embedding)
        if score > best_score:
            best_name = name
            best_score = score

    if best_score < threshold:
        return None, max(0.0, best_score)
    return best_name, best_score


def _intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class FaceTracker:
    """Keep blink state with a nearby face across adjacent webcam frames."""

    def __init__(self) -> None:
        self.tracks = {}
        self.next_track_id = 0

    def update(self, boxes: Sequence[np.ndarray], frame_number: int) -> List[FaceTrack]:
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if frame_number - track.last_seen > MAX_TRACK_MISSED_FRAMES
        ]
        for track_id in expired:
            del self.tracks[track_id]

        matched_ids = set()
        current_tracks = []
        for box in boxes:
            box = np.asarray(box, dtype=np.float32)
            box_center = (box[:2] + box[2:]) / 2.0
            box_scale = max(1.0, norm(box[2:] - box[:2]))
            candidates = []

            for track_id, track in self.tracks.items():
                if track_id in matched_ids:
                    continue
                previous = track.bbox
                previous_center = (previous[:2] + previous[2:]) / 2.0
                previous_scale = max(1.0, norm(previous[2:] - previous[:2]))
                distance = norm(box_center - previous_center) / max(box_scale, previous_scale)
                overlap = _intersection_over_union(box, previous)
                if overlap >= 0.05 or distance <= 0.65:
                    candidates.append((distance - overlap, track_id))

            if candidates:
                _, track_id = min(candidates)
                track = self.tracks[track_id]
                track.bbox = box
                track.last_seen = frame_number
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                track = FaceTrack(bbox=box, last_seen=frame_number)
                self.tracks[track_id] = track

            matched_ids.add(track_id)
            current_tracks.append(track)

        return current_tracks


def face_eye_aspect_ratio(predictor, gray_frame: np.ndarray, bbox: np.ndarray) -> Optional[float]:
    """Find eye landmarks inside a clipped face box and calculate the EAR."""
    height, width = gray_frame.shape[:2]
    left = max(0, min(width - 1, int(bbox[0])))
    top = max(0, min(height - 1, int(bbox[1])))
    right = max(0, min(width, int(bbox[2])))
    bottom = max(0, min(height, int(bbox[3])))
    if right <= left or bottom <= top:
        return None

    rectangle = dlib.rectangle(left, top, right - 1, bottom - 1)
    try:
        shape = predictor(gray_frame, rectangle)
    except (RuntimeError, dlib.error):
        return None

    landmarks = face_utils.shape_to_np(shape)
    left_eye = eye_aspect_ratio(landmarks[36:42])
    right_eye = eye_aspect_ratio(landmarks[42:48])
    if left_eye is None or right_eye is None:
        return None
    return (left_eye + right_eye) / 2.0


def update_blink_state(
    track: FaceTrack, ear: Optional[float], closure_ratio: float
) -> Optional[float]:
    """Calibrate to open-eye EAR and record a blink; return the closed-eye cutoff."""
    if ear is None:
        return None

    if track.open_eye_baseline is None:
        # Use the first sample as a starting point. If it was a closed eye,
        # the next clearly more-open sample will quickly recalibrate it.
        track.open_eye_baseline = ear
        track.has_seen_open_eyes = True
        return max(MIN_EYE_ASPECT_RATIO, ear * closure_ratio)

    cutoff = max(MIN_EYE_ASPECT_RATIO, track.open_eye_baseline * closure_ratio)
    if ear < cutoff:
        if track.has_seen_open_eyes:
            track.closed_eye_frames += 1
        return cutoff

    if track.has_seen_open_eyes and track.closed_eye_frames >= MIN_CLOSED_EYE_FRAMES:
        track.blink_confirmed = True
    track.has_seen_open_eyes = True
    track.closed_eye_frames = 0

    if ear > track.open_eye_baseline * 1.2:
        track.open_eye_baseline = ear
    else:
        track.open_eye_baseline = 0.98 * track.open_eye_baseline + 0.02 * ear
    return max(MIN_EYE_ASPECT_RATIO, track.open_eye_baseline * closure_ratio)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize known faces from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="minimum cosine similarity for a recognized face (default: 0.5)",
    )
    parser.add_argument(
        "--eye-closure-ratio",
        type=float,
        default=DEFAULT_EYE_CLOSURE_RATIO,
        help=(
            "closed-eye cutoff as a fraction of your open-eye EAR "
            "(default: 0.72; increase if blinks are missed)"
        ),
    )
    parser.add_argument("--cpu", action="store_true", help="run face analysis on CPU")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not -1.0 <= args.threshold <= 1.0:
        print("Error: --threshold must be between -1.0 and 1.0.")
        return 2
    if not 0.4 <= args.eye_closure_ratio <= 0.95:
        print("Error: --eye-closure-ratio must be between 0.4 and 0.95.")
        return 2
    if not PREDICTOR_PATH.is_file():
        print("Error: dlib landmark model is missing:", PREDICTOR_PATH)
        return 1

    try:
        model = create_face_model(force_cpu=args.cpu)
        predictor = dlib.shape_predictor(str(PREDICTOR_PATH))
    except Exception as error:
        print("Error: Could not initialize face-recognition models:", error)
        return 1

    known_faces = load_known_faces(model, KNOWN_FACES_DIR)
    if not known_faces:
        print("No usable reference faces found in:", KNOWN_FACES_DIR)
        return 1

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        print("Error: Could not open webcam index", args.camera)
        return 1

    tracker = FaceTracker()
    frame_number = 0
    print("Starting webcam. Press 'q' to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Warning: Could not read a frame from the webcam.")
                break

            frame_number += 1
            faces = model.get(frame)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            recognized = []
            for face in faces:
                label, score = recognize_face(
                    np.asarray(face.embedding, dtype=np.float32),
                    known_faces,
                    args.threshold,
                )
                if label is not None:
                    recognized.append((face, label, score))

            tracks = tracker.update(
                [face.bbox.astype(np.float32) for face, _, _ in recognized], frame_number
            )

            for (face, label, score), track in zip(recognized, tracks):
                bbox = face.bbox.astype(int)
                ear = face_eye_aspect_ratio(predictor, gray_frame, face.bbox)
                cutoff = update_blink_state(track, ear, args.eye_closure_ratio)
                if ear is None:
                    status = "EAR unavailable; blink check unavailable"
                elif track.blink_confirmed:
                    status = "Blink detected"
                else:
                    status = "Blink to confirm"

                color = (0, 255, 0) if track.blink_confirmed else (0, 165, 255)
                if ear is None:
                    label_text = "{} ({:.2f}) | {}".format(label, score, status)
                else:
                    label_text = "{} ({:.2f}) | EAR {:.2f}/<{:.2f} | {}".format(
                        label, score, ear, cutoff, status
                    )
                cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                cv2.putText(
                    frame,
                    label_text,
                    (max(0, bbox[0]), max(25, bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
