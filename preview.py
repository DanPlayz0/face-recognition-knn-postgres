import os
import cv2
import psycopg2
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

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
    if result and result[2] < threshold:
        return result[0], result[1], result[2]
    return "Unknown", "Unknown", result[2] if result else None

def preview(test_dir="dataset/test", threshold=0.70, step=0.01):
    """
    Preview images interactively.
    Arrow keys:
        w  = increase threshold
        s  = decrease threshold
        a  = previous image
        d  = next image
        m  = move to (100,100)
        Q  = quit
    """
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)

    conn = connect_db()
    cur = conn.cursor()

    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    idx = 0

    while 0 <= idx < len(files):
        file = files[idx]
        img_path = os.path.join(test_dir, file).replace("\\", "/")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {file}")
            idx += 1
            continue

        faces = app.get(img)
        if not faces:
            print(f"⚠️ No face detected in {file}")
            idx += 1
            continue

        print(f"\n🖼 Previewing {file} ({len(faces)} face(s)), threshold={threshold:.2f}")

        # Draw faces and labels
        img_draw = img.copy()
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            first_name, last_name, distance = find_closest_face(cur, embedding, threshold)
            name = f"{first_name} {last_name}"
            dist_str = f"{distance:.4f}" if distance is not None else "N/A"
            label = f"{name} ({dist_str})"
            color = (0, 255, 0) if name != "Unknown Unknown" else (0, 0, 255)
            cv2.rectangle(img_draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(img_draw, label, (bbox[0], max(bbox[1]-10,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        max_dim = 1080
        h, w = img_draw.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1.0)  # only scale down, never up
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img_draw = cv2.resize(img_draw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("Face Preview", img_draw)
        key = cv2.waitKey(0)

        # Map arrow keys
        if key == ord('m'):
          cv2.moveWindow("Face Preview", 100, 100)
        elif key == ord('w'):  # Up
            threshold += step
            print(f"⬆ Increased threshold to {threshold:.2f}")
        elif key == ord('s'):  # Down
            threshold = max(0, threshold - step)
            print(f"⬇ Decreased threshold to {threshold:.2f}")
        elif key == ord('a'):  # Left
            idx = max(0, idx - 1)
        elif key == ord('d'):  # Right
            idx += 1
        elif key in (ord('q'), ord('Q')):
            break

    cur.close()
    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    preview()
