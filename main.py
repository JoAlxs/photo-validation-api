from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
import mediapipe as mp
from PIL import Image

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODELO_PATH = "modelo.jpg"  # Imagen de referencia

app = FastAPI()

def detectar_y_recortar_rostro(imagen_path, modelo_dims=(480, 640)):
    img = cv2.imread(imagen_path)
    alto, ancho = img.shape[:2]

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        result = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not result.detections:
            return None, "❌ No se detectó rostro"

        face = result.detections[0].location_data.relative_bounding_box
        x = int(face.xmin * ancho)
        y = int(face.ymin * alto)
        w = int(face.width * ancho)
        h = int(face.height * alto)

        # 🧠 NUEVO: calcula región extendida alrededor del rostro
        factor_alto = modelo_dims[1] / modelo_dims[0]
        proporción_modelo = 1.5  # ancho-alto aproximado deseado

        # Tamaño deseado basado en el rostro
        new_w = int(w * 2)  # más ancho que la cara
        new_h = int(new_w * proporción_modelo)

        centro_x = x + w // 2
        centro_y = y + h // 2

        x1 = max(centro_x - new_w // 2, 0)
        y1 = max(centro_y - new_h // 2, 0)
        x2 = min(centro_x + new_w // 2, ancho)
        y2 = min(centro_y + new_h // 2, alto)

        recorte = img[y1:y2, x1:x2]
        return recorte, "✅ Recorte extendido proporcional al modelo"

def redimensionar(imagen, dimensiones=(480, 640)):
    return cv2.resize(imagen, dimensiones)

def comparar_histogramas(img1, img2):
    # Convertir a HSV
    h1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    h2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Histogramas de color
    hist1 = cv2.calcHist([h1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([h2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    # Normalizar
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Comparación (cuanto más cerca de 1, más similar)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

@app.post("/validar-foto")
async def validar(file: UploadFile = File(...)):
    original_path = os.path.join(UPLOAD_DIR, "original.jpg")
    procesada_path = os.path.join(UPLOAD_DIR, "procesada.jpg")

    with open(original_path, "wb") as f:
        f.write(await file.read())

    # Paso 1: Detectar y recortar rostro
    recorte, mensaje = detectar_y_recortar_rostro(original_path)
    if recorte is None:
        return JSONResponse(content={"valida": False, "motivo": mensaje})

    # Paso 2: Redimensionar
    referencia = cv2.imread(MODELO_PATH)
    dimensiones = (referencia.shape[1], referencia.shape[0])
    final = redimensionar(recorte, dimensiones)
    cv2.imwrite(procesada_path, final)

    # Paso 3: Comparar con imagen modelo
    similarity = comparar_histogramas(final, referencia)

    # Paso 4: Validación
    valido = similarity > 0.85
    resultado = {
        "valida": valido,
        "similitud": round(similarity, 3),
        "motivo": "✅ Muy similar al modelo" if valido else "❌ No se parece suficiente al modelo",
        "dimensiones_modelo": dimensiones
    }
    return JSONResponse(content=resultado)


@app.get("/descargar-procesada")
async def descargar_procesada():
    procesada_path = os.path.join(UPLOAD_DIR, "procesada.jpg")
    if not os.path.exists(procesada_path):
        return JSONResponse(status_code=404, content={"error": "Imagen procesada no encontrada. Primero sube una foto."})
    
    return FileResponse(path=procesada_path, filename="foto_procesada.jpg", media_type="image/jpeg")