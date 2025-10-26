from enum import Enum
import os
import cv2
import numpy as np
import psycopg2
from dotenv import load_dotenv
from flask import Flask, json, render_template, request, redirect, url_for
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename

load_dotenv()

# --- Flask setup ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Face analysis model ---
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# --- Database connection ---
def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )

# --- Core logic ---
def find_closest_face(cur, embedding):
    cur.execute(
      """
      SELECT p.first_name, p.last_name, (f.embedding <=> %s::vector) AS distance
      FROM face_embeddings f
      JOIN persons p ON f.person_id = p.id
      ORDER BY distance ASC
      LIMIT 1;
      """,
      (embedding.tolist(),),
    )
    result = cur.fetchone()
    if result:
      return result[0], result[1], result[2]
    return "Unknown", "Unknown", None


def compute_faces(img, app, cur, threshold=0.70):
    faces = app.get(img)
    faces_info = []
    for face in faces:
      bbox = face.bbox.astype(int)
      embedding = face.embedding
      first_name, last_name, distance = find_closest_face(cur, embedding)
      name = f"{first_name} {last_name}"
      dist_str = f"{distance:.4f}" if distance is not None else "N/A"
      color = (0, 255, 0) if distance is not None and distance <= threshold else (0, 0, 255)
      faces_info.append({
        "bbox": bbox,
        "name": name,
        "first_name": first_name,
        "last_name": last_name,
        "distance": distance,
        "distance_str": dist_str,
        "color": color
      })
    return faces_info

# --- Flask routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if not file:
            return redirect(url_for("index"))
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        return redirect(url_for("result", filename=filename))

    # ✅ List all uploaded images
    upload_folder = app.config["UPLOAD_FOLDER"]
    images = [
        f for f in os.listdir(upload_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]

    return render_template("index.html", images=images)

# def index():
#     if request.method == "POST":
#         file = request.files["image"]
#         visual_style = request.form.get("visual_style", "BLOCK").upper()
#         if not file:
#             return redirect(url_for("index"))
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(filepath)
#         return redirect(url_for("result", filename=filename, style=visual_style))
#     return render_template("index.html")


@app.route("/result/<filename>")
def result(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return redirect(url_for("index"))
    
    json_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename}_faces.json")
    if os.path.exists(json_path):
        return render_template(
            "result.html",
            image=url_for("static", filename=f"uploads/{filename}"),
            faces_json=url_for("static", filename=f"uploads/{filename}_faces.json"),
        )
    
    
    img = cv2.imread(filepath)

    conn = connect_db()
    cur = conn.cursor()

    faces_info = compute_faces(img, face_app, cur)    
    with open(json_path, "w") as f:
        json.dump(faces_info, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

    cur.close()
    conn.close()

    return render_template(
        "result.html",
        image=url_for("static", filename=f"uploads/{filename}"),
        faces_json=url_for("static", filename=f"uploads/{filename}_faces.json"),  # Pass JSON file URL
    )


if __name__ == "__main__":
    app.run(debug=True)
