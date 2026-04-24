import cv2
import os

ARQUIVO_ENTRADA  = "jogo3.mp4"
ARQUIVO_SAIDA    = "jogo4_teste.mp4"
INICIO_SEGUNDOS  = 1757   # minuto 29:17
FIM_SEGUNDOS     = 1817   # minuto 30:17 (1 minuto)

cap = cv2.VideoCapture(ARQUIVO_ENTRADA)

if not cap.isOpened():
    print(f"[ERRO] Não foi possível abrir: {ARQUIVO_ENTRADA}")
    exit(1)

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0

total_frames = int((FIM_SEGUNDOS - INICIO_SEGUNDOS) * fps)
inicio_frame = int(INICIO_SEGUNDOS * fps)

print(f"[Info] FPS: {fps:.1f} | Resolução: {width}x{height}")
print(f"[Info] Cortando do minuto {INICIO_SEGUNDOS//60} ao {FIM_SEGUNDOS//60}")
print(f"[Info] Total de frames a copiar: {total_frames}")

# Pula para o frame de início
cap.set(cv2.CAP_PROP_POS_FRAMES, inicio_frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(ARQUIVO_SAIDA, fourcc, fps, (width, height))

for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        print(f"[Aviso] Vídeo terminou antes do esperado no frame {i}")
        break
    out.write(frame)
    if i % 300 == 0:
        segundos = INICIO_SEGUNDOS + int(i / fps)
        print(f"  Processando... {segundos//60:02d}:{segundos%60:02d} "
              f"({i}/{total_frames} frames)")

cap.release()
out.release()
print(f"\n✅ Salvo como: {ARQUIVO_SAIDA}")
print(f"   Duração: 5 minutos ({total_frames} frames)")