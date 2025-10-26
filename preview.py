from enum import Enum
import os
import cv2
import numpy as np
import psycopg2
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

class VisualStyle(Enum):
    SIMPLE = 0
    BLOCK = 1

class HoverMode(Enum):
    SHOW_ALL = 0
    HOVER_ONLY = 1
    SHOW_BELOW_THRESHOLD = 2

def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def find_closest_face(cur, embedding, threshold=0.35):
    cur.execute("""
        SELECT p.first_name, p.last_name, (f.embedding <=> %s::vector) AS distance
        FROM face_embeddings f
        JOIN persons p ON f.person_id = p.id
        ORDER BY distance ASC
        LIMIT 1;
    """, (embedding.tolist(),))
    result = cur.fetchone()
    if result: #and result[2] < threshold:
        return result[0], result[1], result[2]
    return "Unknown", "Unknown", result[2] if result else None

def compute_faces(img, app, cur, threshold=0.35):
    """Detect faces, compute embeddings, and fetch closest matches."""
    faces = app.get(img)
    faces_info = []

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        first_name, last_name, distance = find_closest_face(cur, embedding, threshold)

        name = f"{first_name} {last_name}"
        dist_str = f"{distance:.4f}" if distance is not None else "N/A"
        label = f"{name} ({dist_str})"
        color = (0, 255, 0) if distance is not None and distance <= threshold else (0, 0, 255)

        faces_info.append({"bbox": bbox, "label": label, "color": color, "distance": distance})
    return faces_info


def scale_image_and_faces(img, faces_info, max_dim=1080):
    """Scale down large images (and adjust bounding boxes)."""
    h, w = img.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        for f in faces_info:
            f["bbox"] = np.round(np.array(f["bbox"]) * scale).astype(int)
    return img, faces_info

def draw_faces(img, faces_info, visual_style=VisualStyle.SIMPLE, hover_mode=HoverMode.SHOW_ALL, hovered_face=None, threshold=0.7):
    """Draw faces and labels depending on mode and style."""
    img_draw = img.copy()

    for face in faces_info:
        (x1, y1, x2, y2) = face["bbox"]
        color = face["color"]

        # Always draw the rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # Label condition (safe comparison)
        show_label = hover_mode == HoverMode.SHOW_ALL
        if hover_mode == HoverMode.HOVER_ONLY:
            show_label = (
                hovered_face is not None
                and np.array_equal(hovered_face["bbox"], face["bbox"])
            )
        elif hover_mode == HoverMode.SHOW_BELOW_THRESHOLD:
            if face["distance"] is not None and face["distance"] <= threshold:
                show_label = True
            else:
              show_label = (
                  hovered_face is not None
                  and np.array_equal(hovered_face["bbox"], face["bbox"])
              )

        if show_label:
            if visual_style == VisualStyle.SIMPLE:  # Simple text above
                cv2.putText(img_draw, face["label"], (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            elif visual_style == VisualStyle.BLOCK:  # Block style below
                cv2.rectangle(img_draw, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img_draw, face["label"], (x1 + 6, y2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_draw

def preview(test_dir="dataset/test", threshold=0.70, step=0.01):
    """
    Interactive face preview.
    Keys:
        w/s = increase/decrease threshold
        a/d = previous/next image
        v   = toggle visual style
        h   = toggle hover / show all
        m   = move window
        q   = quit
    """
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)

    conn = connect_db()
    cur = conn.cursor()

    files = [
      f for f in os.listdir(test_dir)
      if os.path.isfile(os.path.join(test_dir, f))
      and not f.startswith('.')
    ]
    idx = 0
    visual_style = VisualStyle.BLOCK
    hover_mode = HoverMode.SHOW_ALL
    hovered_face = None

    mouse_x, mouse_y = -1, -1

    cv2.namedWindow("Face Preview")

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y, hovered_face
        mouse_x, mouse_y = x, y
        if hover_mode and event == cv2.EVENT_MOUSEMOVE:
            hovered_face = None
            for face in faces_info:
                (x1, y1, x2, y2) = face["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hovered_face = face
                    break

    cv2.setMouseCallback("Face Preview", on_mouse)

    while 0 <= idx < len(files):
        file = files[idx]
        img_path = os.path.join(test_dir, file).replace("\\", "/")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {file}")
            idx += 1
            continue

        faces_info = compute_faces(img, app, cur, threshold)
        if not faces_info:
            print(f"⚠️ No face detected in {file}")
            idx += 1
            continue

        img, faces_info = scale_image_and_faces(img, faces_info)
        print(f"\n🖼 Previewing {file} ({len(faces_info)} face(s)), threshold={threshold:.2f}")

        # Continuous display loop for hover
        while True:
            img_draw = draw_faces(img, faces_info, visual_style, hover_mode, hovered_face, threshold)
            cv2.imshow("Face Preview", img_draw)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('m'):
                cv2.moveWindow("Face Preview", 100, 100)
            elif key == ord('w'):
                threshold += step
                print(f"⬆ Increased threshold to {threshold:.2f}")
                break
            elif key == ord('s'):
                threshold = max(0, threshold - step)
                print(f"⬇ Decreased threshold to {threshold:.2f}")
                break
            elif key == ord('a'):
                idx = max(0, idx - 1)
                break
            elif key == ord('d'):
                idx += 1
                break
            elif key == ord('v'):
                visual_style = (
                    VisualStyle.BLOCK if visual_style == VisualStyle.SIMPLE else VisualStyle.SIMPLE
                )
                print(f"🔄 Visual style: {visual_style.name.title()}")
            elif key == ord('h'):
                next_mode = {
                    HoverMode.SHOW_ALL: HoverMode.HOVER_ONLY,
                    HoverMode.HOVER_ONLY: HoverMode.SHOW_BELOW_THRESHOLD,
                    HoverMode.SHOW_BELOW_THRESHOLD: HoverMode.SHOW_ALL
                }
                hover_mode = next_mode[hover_mode]
                print(f"🖱 Mode: {hover_mode.name.replace('_', ' ').title()}")
            elif key in (ord('q'), ord('Q'), 27):
                cur.close()
                conn.close()
                cv2.destroyAllWindows()
                return

    cur.close()
    conn.close()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    preview()
