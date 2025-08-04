import cv2
import numpy as np
import os
from numpy.linalg import norm
import insightface
import dlib
from imutils import face_utils

EYE_AR_THRESH = 0.24
EYE_AR_CONSEC_FRAMES = 1

def eye_aspect_ratio(eye):
    A = norm(eye[1] - eye[5])
    B = norm(eye[2] - eye[4])
    C = norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def load_known_faces(folder):
    known_faces = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            faces = model.get(img)
            if faces:
                known_faces[name] = faces[0].embedding
                print(f"Loaded embedding for: {name}")
            else:
                print(f"Warning: No face found in {filename}")
    return known_faces

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def main():
    known_faces = load_known_faces("known_faces")
    if not known_faces:
        print("No known faces loaded! Add images to 'known_faces/' and restart.")
        return

    cap = cv2.VideoCapture(0)
    print("Starting webcam. Press 'q' to quit.")

    blink_counter = {}
    liveness_confirmed = {}
    open_eyes = {}
    frame_count = 0
    last_label = "Unknown"
    last_score = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break

        faces = model.get(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for face in faces:
            bbox = face.bbox.astype(int)

            if frame_count % 5 == 0:
                embedding = face.embedding
                label = "Unknown"
                best_score = 0.0
                for name, ref_emb in known_faces.items():
                    sim = cosine_similarity(embedding, ref_emb)
                    if sim > best_score:
                        best_score = sim
                        if sim > 0.5:
                            label = name
                last_label = label
                last_score = best_score
            else:
                label = last_label
                best_score = last_score


            matched_rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
            shape = predictor(gray, matched_rect)
            shape_np = face_utils.shape_to_np(shape)
            left_eye = shape_np[36:42]
            right_eye = shape_np[42:48]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if label != "Unknown":
                blink_counter.setdefault(label, 0)
                liveness_confirmed.setdefault(label, False)
                open_eyes.setdefault(label, 0)

               
                if open_eyes[label] < EYE_AR_CONSEC_FRAMES and ear > EYE_AR_THRESH:
                    open_eyes[label] += 1

                if ear < EYE_AR_THRESH:
                    blink_counter[label] += 1
                else:
                    if blink_counter[label] >= EYE_AR_CONSEC_FRAMES and open_eyes[label] >= EYE_AR_CONSEC_FRAMES:
                        liveness_confirmed[label] = True
                    blink_counter[label] = 0

                liveness_status = "Live" if liveness_confirmed[label] else "Spoof?"
                label_text = f"{label} - {ear} {liveness_status} "
                color = (0, 255, 0) if liveness_confirmed[label] else (0, 0, 255)
            else:
                label_text = f"Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
