import cv2
import psycopg2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
import os

load_dotenv()

def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def find_closest_face(embedding, threshold=0.8):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.first_name, p.last_name, (f.embedding <=> %s::vector) AS distance
        FROM face_embeddings f
        JOIN persons p ON f.person_id = p.id
        ORDER BY distance ASC
        LIMIT 1;
    """, (embedding.tolist(),))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result and result[3] < threshold:
        return result[0], result[1], result[2], result[3]
    return None, "Unknown", "Unknown", None

def main():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)

    video = cv2.VideoCapture(0)
    print("Press 'q' to quit")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            (x, y, w, h) = face.bbox.astype(int)
            embedding = face.embedding
            person_id, first_name, last_name, distance = find_closest_face(embedding)
            label = f"{first_name} {last_name} ({distance:.2f})" if distance else "Unknown"

            cv2.rectangle(frame, (x, y), (w, h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            if person_id:
                print(f"Matched {first_name} {last_name} (ID {person_id}) distance={distance:.4f}")

        cv2.imshow("Recognize Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
