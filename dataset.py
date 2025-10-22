import os
import cv2
import psycopg2
import hashlib
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

def connect_db():
    """Create a single reusable DB connection."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def image_hash(path):
    """Generate SHA256 hash of the image for traceability."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_or_create_person(cur, first_name, last_name):
    """Return person_id if exists, otherwise insert."""
    cur.execute("SELECT id FROM persons WHERE first_name = %s AND last_name = %s;", (first_name, last_name,))
    result = cur.fetchone()
    if result:
        return result[0]
    cur.execute("INSERT INTO persons (first_name, last_name) VALUES (%s, %s) RETURNING id;", (first_name, last_name,))
    return cur.fetchone()[0]

def embedding_exists(cur, source_image, image_hash_val):
    """Check if image already processed."""
    cur.execute("""
        SELECT id FROM face_embeddings
        WHERE source_image = %s OR image_hash = %s;
    """, (source_image, image_hash_val))
    return cur.fetchone() is not None

def insert_embedding(cur, pid, embedding, source_image, image_hash_val, model='insightface_buffalo_l_v1'):
    """Insert a new embedding row."""
    cur.execute("""
        INSERT INTO face_embeddings (person_id, embedding, source_image, image_hash, model)
        VALUES (%s, %s, %s, %s, %s);
    """, (pid, embedding.tolist(), source_image, image_hash_val, model))

def process_image(path, app):
    """Extract embedding from an image — returns None if 0 or >1 faces."""
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Failed to read image: {path}")
        return None
    faces = app.get(img)
    if not faces:
        print(f"⚠️ No face detected in {path}")
        return None
    if len(faces) > 1:
        print(f"⚠️ Multiple ({len(faces)}) faces detected in {path}. Skipping.")
        return None
    return faces[0].embedding

def main(dataset_dir="dataset/train"):
    # Prepare model and database
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)
    conn = connect_db()
    cur = conn.cursor()

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"\n👤 Processing person: {person_name}")
        first_name, last_name = person_name.split("_")
        person_id = get_or_create_person(cur, first_name, last_name)
        conn.commit()  # commit early for stability

        for img_file in os.listdir(person_path):
            # Ignore .keep files (aka A_Example Person)
            if img_file.startswith(".") or img_file.endswith(".keep"):
              continue
            # Images only
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
              continue
            img_path = os.path.join(person_path, img_file)
            img_hash = image_hash(img_path)
            
            normalized_path = img_path.replace("\\", "/")

            # Skip duplicates
            if embedding_exists(cur, normalized_path, img_hash):
                # print(f"⏩ Skipping already processed: {img_file}")
                continue

            emb = process_image(normalized_path, app)
            if emb is not None:
                insert_embedding(cur, person_id, emb, normalized_path, img_hash)
                print(f"✅ Added embedding for {person_name}: {img_file}")
                conn.commit()

    cur.close()
    conn.close()
    print("\n🎉 Import completed successfully!")

if __name__ == "__main__":
    main()
