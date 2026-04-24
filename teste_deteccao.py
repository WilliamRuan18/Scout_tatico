# teste_deteccao.py
import cv2
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("runs/detect/futebol2/weights/best.pt")
print("Classes do modelo:", model.names)

cap = cv2.VideoCapture("jogo_teste.mp4")

# Pula para o meio do vídeo
cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)

ret, frame = cap.read()
if ret:
    results = model.predict(frame, conf=0.1, verbose=True, imgsz=960)
    print(f"\nTotal detectado: {len(results[0].boxes)}")
    for box in results[0].boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  Classe: {model.names[cls]} | Confiança: {conf:.2%}")

    # Salva imagem com detecções
    img = results[0].plot()
    cv2.imwrite("teste_resultado.jpg", img)
    print("\nImagem salva como teste_resultado.jpg")

cap.release()