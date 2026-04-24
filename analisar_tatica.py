"""
ANÁLISE TÁTICA DE FUTEBOL
==========================
Detecta jogadores, identifica formação tática e gera relatório.

Uso:
  python analisar_tatica.py --video jogo.mp4
  python analisar_tatica.py --video 0              (webcam)
  python analisar_tatica.py --video "URL_STREAM"   (YouTube ao vivo)
"""

import os, sys, cv2, json, argparse, threading, time
import numpy as np
from collections import Counter
from datetime import datetime
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                              QFrame, QVBoxLayout, QHBoxLayout, QPushButton,
                              QFileDialog, QProgressBar)
from PyQt5.QtCore    import Qt, QThread, pyqtSignal
from PyQt5.QtGui     import QImage, QPixmap

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/detect/futebol2/weights/best.pt"
CONF        = 0.30
IMGSZ       = 640    # 640 = mais rápido | 960 = mais preciso
SKIP_FRAMES = 1      # 1 = todos | 2 = metade | 3 = um terço

CLASSES_JOGADOR = ("player", "goalkeeper")
CLASSE_BOLA     = "ball"
CLASSE_ARBITRO  = "referee"

COR_TIME_A  = (220, 100,  40)
COR_TIME_B  = ( 40, 190,  40)
COR_BOLA    = (  0, 210, 255)
COR_ARBITRO = (200, 200, 200)

FORMACOES = {
    (4,3,3):   "4-3-3",
    (4,4,2):   "4-4-2",
    (4,2,3,1): "4-2-3-1",
    (3,5,2):   "3-5-2",
    (3,4,3):   "3-4-3",
    (5,3,2):   "5-3-2",
    (5,4,1):   "5-4-1",
    (4,5,1):   "4-5-1",
    (4,1,4,1): "4-1-4-1",
}


# ── LEITOR DE STREAM COM BAIXO DELAY ──────────────────────────────────────────
class StreamReader:
    """
    Lê frames em thread separada e mantém sempre o frame mais recente.
    Elimina acúmulo de buffer que causa delay em streams ao vivo.
    """
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buffer mínimo
        self.frame   = None
        self.ret     = False
        self.running = True
        self.lock    = threading.Lock()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret   = ret
                self.frame = frame if ret else self.frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.running = False
        self.cap.release()


# ── CLASSIFICADOR DE TIME POR COR DE CAMISA ────────────────────────────────────
class ClassificadorTimes:
    def __init__(self):
        # Calibração manual — Man City (azul claro) x Tottenham (branco)
        self.centro_a  = np.array([160.0, 140.0,  90.0], dtype=np.float32)
        self.centro_b  = np.array([210.0, 210.0, 210.0], dtype=np.float32)
        self.calibrado = True
        print("[Calibração] Man City=Azul | Tottenham=Verde")

    def _roi_camisa(self, frame, x1, y1, x2, y2):
        h, w = y2 - y1, x2 - x1
        ry1 = y1 + int(h * 0.20)
        ry2 = y1 + int(h * 0.65)
        rx1 = x1 + int(w * 0.15)
        rx2 = x1 + int(w * 0.85)
        ry1, ry2 = max(ry1, 0), min(ry2, frame.shape[0] - 1)
        rx1, rx2 = max(rx1, 0), min(rx2, frame.shape[1] - 1)
        roi = frame[ry1:ry2, rx1:rx2]
        return roi if roi.size > 0 else None

    def _cor_media(self, frame, x1, y1, x2, y2):
        roi = self._roi_camisa(frame, x1, y1, x2, y2)
        if roi is None:
            return None
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        grama  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        mask   = cv2.bitwise_not(grama)
        pixels = roi[mask > 0]
        if len(pixels) < 10:
            return roi.mean(axis=(0, 1))
        return pixels.mean(axis=0)

    def classificar(self, frame, x1, y1, x2, y2):
        cor = self._cor_media(frame, x1, y1, x2, y2)
        if cor is None:
            return 'A'
        b, g, r = int(cor[0]), int(cor[1]), int(cor[2])
        pixel   = np.uint8([[[b, g, r]]])
        hsv     = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h_val, s_val, v_val = int(hsv[0]), int(hsv[1]), int(hsv[2])
        # Man City: azul claro
        if 95 <= h_val <= 130 and 40 <= s_val <= 200:
            return 'A'
        # Tottenham: branco ou preto
        if s_val < 50 or v_val < 60:
            return 'B'
        # Fallback euclidiano
        da = np.linalg.norm(cor - self.centro_a)
        db = np.linalg.norm(cor - self.centro_b)
        return 'A' if da <= db else 'B'


# ── FORMAÇÃO ───────────────────────────────────────────────────────────────────
def detectar_formacao(posicoes_y, altura):
    if len(posicoes_y) < 5:
        return "?", []
    yn = sorted(y / max(altura, 1) for y in posicoes_y)
    outfield = yn[1:] if len(yn) > 1 else yn
    z1 = sum(1 for y in outfield if y < 0.33)
    z2 = sum(1 for y in outfield if 0.33 <= y < 0.66)
    z3 = sum(1 for y in outfield if y >= 0.66)
    tupla  = tuple(z for z in [z1, z2, z3] if z > 0)
    melhor, menor = None, 999
    for ft, fn in FORMACOES.items():
        ft3 = ft[-3:]
        d   = sum(abs(a - b) for a, b in zip(ft3, tupla[:len(ft3)]))
        if d < menor:
            menor, melhor = d, fn
    return melhor or f"{z1}-{z2}-{z3}", [z1, z2, z3]


def sugerir_contra_ataque(form):
    dicas = {
        "4-3-3":   "Adversário ataca pelos lados — explore o espaço central no contra-ataque.",
        "4-4-2":   "Dois atacantes fixos — explore as costas dos laterais com velocidade.",
        "4-2-3-1": "Dupla pivô protege a defesa — jogue pelas laterais.",
        "3-5-2":   "Frágil pelas pontas — explore os corredores com profundidade.",
        "3-4-3":   "Linha de 3 — 2 pontas rápidos criam superioridade.",
        "5-3-2":   "Time defensivo — seja paciente, explore espaços entre as linhas.",
        "5-4-1":   "Muito fechado — bolas paradas e chutes de fora da área.",
        "4-5-1":   "Forte no meio — passes diretos ao 9 + entradas de médios.",
    }
    return dicas.get(form, "Analise as zonas de pressão e explore os espaços descobertos.")


# ── THREAD DE ANÁLISE ──────────────────────────────────────────────────────────
class AnaliseThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(dict)
    progresso     = pyqtSignal(int)
    finalizado    = pyqtSignal(dict)

    def __init__(self, source):
        super().__init__()
        self.source        = source
        self.running       = False
        self.model         = YOLO(MODEL_PATH)
        self.clf           = ClassificadorTimes()
        self.posses_a      = 0
        self.posses_b      = 0
        self.hist_form_a   = []
        self.hist_form_b   = []
        self.frame_count   = 0
        self.total_frames  = 1
        self._ultimo_frame = None
        print(f"[Modelo] Classes: {self.model.names}")
        print(f"[Config] IMGSZ={IMGSZ} | SKIP={SKIP_FRAMES} | CONF={CONF}")

    def run(self):
        src = 0 if self.source == "0" else self.source

        # StreamReader para baixo delay em streams ao vivo
        cap = StreamReader(src)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
        self.running = True
        t_prev = time.time()

        # Aguarda o primeiro frame chegar (até 5 segundos)
        print(f"[Stream] Abrindo: {src}")
        for _ in range(100):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[Stream] Primeiro frame recebido! Resolução: {frame.shape[1]}x{frame.shape[0]}")
                break
            time.sleep(0.05)
        else:
            print(f"[ERRO] Não foi possível abrir o vídeo: {src}")
            print("[ERRO] Verifique se o arquivo existe e o caminho está correto.")
            return

        # FPS do vídeo para controlar velocidade de reprodução
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if fps_video <= 0 or fps_video > 120:
            fps_video = 30.0
        delay_frame = 1.0 / fps_video
        print(f"[Stream] FPS do vídeo: {fps_video:.1f}")

        while self.running:
            t_inicio = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1

            # FPS real de processamento
            now    = time.time()
            fps    = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            if self.total_frames > 1:
                self.progresso.emit(
                    int(self.frame_count / self.total_frames * 100))

            # ── Pula frames para reduzir delay ────────────────────────────────
            if SKIP_FRAMES > 1 and self.frame_count % SKIP_FRAMES != 0:
                if self._ultimo_frame is not None:
                    self.frame_ready.emit(self._ultimo_frame)
                continue

            h, w = frame.shape[:2]

            # ── Detecção ──────────────────────────────────────────────────────
            try:
                results = self.model.track(
                    frame, persist=True, conf=CONF,
                    iou=0.40, verbose=False, imgsz=IMGSZ)
            except Exception as e:
                print(f"[ERRO track] {e}")
                self.frame_ready.emit(frame)
                continue

            jog_a, jog_b = [], []
            pos_bola     = None

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()
                ids_t   = results[0].boxes.id
                ids     = (ids_t.cpu().numpy().astype(int)
                           if ids_t is not None else np.arange(len(boxes)))

                if self.frame_count % 60 == 0:
                    print(f"[Frame {self.frame_count}] "
                          f"{len(boxes)} det | FPS={fps:.1f}")

                for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                    x1,y1,x2,y2 = map(int, box)
                    x1,y1 = max(x1,0), max(y1,0)
                    x2,y2 = min(x2,w-1), min(y2,h-1)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    nome   = self.model.names[cls].lower()

                    # ── Bola ──────────────────────────────────────────────────
                    if nome == CLASSE_BOLA:
                        pos_bola = (cx, cy)
                        cv2.circle(frame, (cx,cy), 10, COR_BOLA, -1)
                        cv2.circle(frame, (cx,cy), 12, (255,255,255), 2)
                        cv2.putText(frame, "Bola", (cx+13, cy+5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_BOLA, 1)

                    # ── Árbitro ───────────────────────────────────────────────
                    elif nome == CLASSE_ARBITRO:
                        cv2.rectangle(frame, (x1,y1),(x2,y2), COR_ARBITRO, 2)
                        cv2.putText(frame, "Arbitro", (x1, y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_ARBITRO, 1)

                    # ── Jogador / Goleiro ─────────────────────────────────────
                    elif nome in CLASSES_JOGADOR:
                        time_clf = self.clf.classificar(frame, x1, y1, x2, y2)
                        cor  = COR_TIME_A if time_clf == 'A' else COR_TIME_B
                        if time_clf == 'A':
                            jog_a.append((cx, cy))
                        else:
                            jog_b.append((cx, cy))

                        cv2.rectangle(frame, (x1,y1),(x2,y2), cor, 2)

                        prefixo = "G" if nome == "goalkeeper" else "J"
                        lbl = f"{prefixo}{tid} {conf:.0%}"
                        (tw, th), _ = cv2.getTextSize(
                            lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                        cv2.rectangle(frame,
                                      (x1, y1 - th - 8),
                                      (x1 + tw + 6, y1), cor, -1)
                        cv2.putText(frame, lbl, (x1+3, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                    (255,255,255), 1)

                        bw = int(conf * (x2 - x1))
                        cv2.rectangle(frame,(x1,y2+3),(x2,y2+8),(30,30,30),-1)
                        cv2.rectangle(frame,(x1,y2+3),(x1+bw,y2+8), cor, -1)

            # ── Overlay de status + FPS ───────────────────────────────────────
            total_det = (len(results[0].boxes)
                         if results and results[0].boxes is not None else 0)
            cv2.rectangle(frame, (6,6),(370,75),(0,0,0),-1)
            cv2.putText(frame,
                        f"Frame:{self.frame_count} | Det:{total_det} | FPS:{fps:.1f}",
                        (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(255,165,0),1)
            cv2.putText(frame,
                        f"A(azul):{len(jog_a)}   B(verde):{len(jog_b)}",
                        (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.50,(50,255,50),1)

            # ── Posse de bola ─────────────────────────────────────────────────
            if pos_bola and (jog_a or jog_b):
                bx, by = pos_bola
                da = min((((bx-px)**2+(by-py)**2)**.5 for px,py in jog_a),
                         default=9999)
                db = min((((bx-px)**2+(by-py)**2)**.5 for px,py in jog_b),
                         default=9999)
                if da < db:
                    self.posses_a += 1
                else:
                    self.posses_b += 1

            # ── Formação ──────────────────────────────────────────────────────
            if self.frame_count % 10 == 0:
                if jog_a:
                    fa, _ = detectar_formacao([p[1] for p in jog_a], h)
                    self.hist_form_a.append(fa)
                if jog_b:
                    fb, _ = detectar_formacao([p[1] for p in jog_b], h)
                    self.hist_form_b.append(fb)

            # ── Stats ─────────────────────────────────────────────────────────
            tp = max(self.posses_a + self.posses_b, 1)
            fa = (Counter(self.hist_form_a[-30:]).most_common(1)[0][0]
                  if self.hist_form_a else "?")
            fb = (Counter(self.hist_form_b[-30:]).most_common(1)[0][0]
                  if self.hist_form_b else "?")

            self.stats_updated.emit({
                "posse_a": round(self.posses_a / tp * 100),
                "posse_b": round(self.posses_b / tp * 100),
                "form_a":  fa,
                "form_b":  fb,
                "jog_a":   len(jog_a),
                "jog_b":   len(jog_b),
                "dica":    sugerir_contra_ataque(fb),
                "frame":   self.frame_count,
            })

            self._ultimo_frame = frame.copy()
            self.frame_ready.emit(frame.copy())

            # Controle de velocidade — respeita o FPS original do vídeo
            elapsed = time.time() - t_inicio
            espera  = delay_frame - elapsed
            if espera > 0:
                time.sleep(espera)

        cap.release()

        tp   = max(self.posses_a + self.posses_b, 1)
        fa_f = (Counter(self.hist_form_a).most_common(1)[0][0]
                if self.hist_form_a else "?")
        fb_f = (Counter(self.hist_form_b).most_common(1)[0][0]
                if self.hist_form_b else "?")

        rel = {
            "data":        datetime.now().strftime("%d/%m/%Y %H:%M"),
            "frames":      self.frame_count,
            "posse_a":     round(self.posses_a / tp * 100),
            "posse_b":     round(self.posses_b / tp * 100),
            "formacao_a":  fa_f,
            "formacao_b":  fb_f,
            "dica_tatica": sugerir_contra_ataque(fb_f),
        }
        with open("relatorio_tatico.json", "w", encoding="utf-8") as f:
            json.dump(rel, f, indent=4, ensure_ascii=False)

        self.finalizado.emit(rel)
        self.progresso.emit(100)

    def stop(self):
        self.running = False
        self.wait()


# ── PAINEL LATERAL ─────────────────────────────────────────────────────────────
class PainelTatico(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setStyleSheet("background:#0D1B2A;")
        self._build()

    def _sep(self, lay):
        s = QFrame()
        s.setFrameShape(QFrame.HLine)
        s.setStyleSheet("background:#1E3040; border:none; max-height:1px;")
        lay.addWidget(s)

    def _lbl(self, lay, txt, cor, size=11, bold=True):
        l = QLabel(txt)
        w = "bold" if bold else "normal"
        l.setStyleSheet(
            f"color:{cor}; font-size:{size}px; font-weight:{w};"
            f" background:transparent;")
        lay.addWidget(l)
        return l

    def _bar(self, lay, cor):
        b = QProgressBar()
        b.setRange(0, 100); b.setValue(0)
        b.setFixedHeight(14); b.setFormat("%p% posse")
        b.setStyleSheet(f"""
            QProgressBar {{ background:#1E3040; border-radius:6px;
                            border:none; color:#ECEFF1; font-size:10px; }}
            QProgressBar::chunk {{ background:{cor}; border-radius:6px; }}""")
        lay.addWidget(b)
        return b

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16,16,16,16)
        lay.setSpacing(10)

        self._lbl(lay, "Análise Tática", "#4FC3F7", 16)
        self._sep(lay)

        self._lbl(lay, "TIME A — Man City (Azul)", "#4FC3F7")
        self.lbl_fa = self._lbl(lay, "Formação: ?",           "#4FC3F7", 14)
        self.lbl_ja = self._lbl(lay, "Jogadores em campo: 0", "#78909C", 12, False)
        self.bar_a  = self._bar(lay, "#378ADD")
        self._sep(lay)

        self._lbl(lay, "TIME B — Tottenham (Verde)", "#66BB6A")
        self.lbl_fb = self._lbl(lay, "Formação: ?",           "#66BB6A", 14)
        self.lbl_jb = self._lbl(lay, "Jogadores em campo: 0", "#78909C", 12, False)
        self.bar_b  = self._bar(lay, "#66BB6A")
        self._sep(lay)

        self._lbl(lay, "Sugestão de Contra-Ataque", "#FFA726")
        self.lbl_dica = QLabel("Aguardando análise...")
        self.lbl_dica.setWordWrap(True)
        self.lbl_dica.setStyleSheet(
            "color:#ECEFF1; font-size:12px; background:transparent;")
        lay.addWidget(self.lbl_dica)
        self._sep(lay)

        self._lbl(lay, "Progresso da Análise", "#78909C")
        self.prog = QProgressBar()
        self.prog.setRange(0,100); self.prog.setValue(0)
        self.prog.setFixedHeight(8); self.prog.setTextVisible(False)
        self.prog.setStyleSheet("""
            QProgressBar { background:#1E3040; border-radius:4px; border:none; }
            QProgressBar::chunk { background:#4FC3F7; border-radius:4px; }""")
        lay.addWidget(self.prog)

        self.lbl_fr = QLabel("Frame: 0")
        self.lbl_fr.setStyleSheet(
            "color:#37474F; font-size:11px; background:transparent;")
        lay.addWidget(self.lbl_fr)
        lay.addStretch()

        self.btn_sal = QPushButton("Salvar Relatório JSON")
        self.btn_sal.setEnabled(False)
        self.btn_sal.setStyleSheet("""
            QPushButton { background:rgba(79,195,247,0.15); color:#4FC3F7;
                          border:1px solid rgba(79,195,247,0.4);
                          border-radius:8px; padding:8px; font-size:13px; }
            QPushButton:hover { background:rgba(79,195,247,0.3); }
            QPushButton:disabled { color:#37474F; border-color:#1E3040; }""")
        lay.addWidget(self.btn_sal)

    def atualizar(self, s):
        self.lbl_fa.setText(f"Formação: {s['form_a']}")
        self.lbl_fb.setText(f"Formação: {s['form_b']}")
        self.lbl_ja.setText(f"Jogadores em campo: {s['jog_a']}")
        self.lbl_jb.setText(f"Jogadores em campo: {s['jog_b']}")
        self.bar_a.setValue(s['posse_a'])
        self.bar_b.setValue(s['posse_b'])
        self.lbl_dica.setText(s['dica'])
        self.lbl_fr.setText(f"Frame: {s['frame']}")

    def set_prog(self, v):
        self.prog.setValue(v)


# ── JANELA PRINCIPAL ───────────────────────────────────────────────────────────
class JanelaTatica(QMainWindow):
    def __init__(self, source=None):
        super().__init__()
        self.setWindowTitle("Análise Tática de Futebol — IA")
        self.showFullScreen()
        self.setStyleSheet("background:#0D1B2A;")
        self.analise = None
        self._build()
        if source:
            self._iniciar(source)

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background:#0D1B2A;")

        lay = QHBoxLayout(central)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(0)

        cam = QWidget(); cam.setStyleSheet("background:#000;")
        cl  = QVBoxLayout(cam); cl.setContentsMargins(0,0,0,0)

        self.btn_vid = QPushButton("Carregar Vídeo do Jogo", cam)
        self.btn_vid.setStyleSheet("""
            QPushButton { background:rgba(79,195,247,0.2); color:#4FC3F7;
                          border:2px solid rgba(79,195,247,0.5);
                          border-radius:10px; padding:12px 24px; font-size:16px; }
            QPushButton:hover { background:rgba(79,195,247,0.4); }""")
        self.btn_vid.clicked.connect(self._carregar)
        self.btn_vid.move(20, 20)

        self.vid_lbl = QLabel()
        self.vid_lbl.setAlignment(Qt.AlignCenter)
        self.vid_lbl.setScaledContents(True)
        self.vid_lbl.setStyleSheet("background:#000;")
        cl.addWidget(self.vid_lbl)
        lay.addWidget(cam, 70)

        div = QFrame(); div.setFrameShape(QFrame.VLine)
        div.setStyleSheet("background:#1E3040; border:none; max-width:1px;")
        lay.addWidget(div)

        self.painel = PainelTatico()
        self.painel.btn_sal.clicked.connect(self._salvar)
        lay.addWidget(self.painel)

        self.btn_sair = QPushButton("✕ Sair", central)
        self.btn_sair.setFixedSize(80, 32)
        self.btn_sair.setStyleSheet("""
            QPushButton { background:rgba(18,26,32,200); color:#EF5350;
                          border:1px solid rgba(239,83,80,0.4);
                          border-radius:8px; font-size:12px; }
            QPushButton:hover { background:rgba(239,83,80,0.25); }""")
        self.btn_sair.clicked.connect(self.close)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'btn_sair'):
            self.btn_sair.move(self.width() - 340, 10)
        if hasattr(self, 'btn_vid'):
            self.btn_vid.move(20, 20)

    def _carregar(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecione o vídeo", "",
            "Vídeos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._iniciar(path)

    def _iniciar(self, src):
        if self.analise:
            self.analise.stop()
        self.btn_vid.hide()
        self.analise = AnaliseThread(src)
        self.analise.frame_ready.connect(self._show)
        self.analise.stats_updated.connect(self.painel.atualizar)
        self.analise.progresso.connect(self.painel.set_prog)
        self.analise.finalizado.connect(self._concluido)
        self.analise.start()

    def _show(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.vid_lbl.setPixmap(QPixmap.fromImage(img))

    def _concluido(self, rel):
        self.relatorio = rel
        self.painel.btn_sal.setEnabled(True)
        print("\n✅ Análise concluída!")
        print(json.dumps(rel, indent=2, ensure_ascii=False))

    def _salvar(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Relatório", "relatorio_tatico.json", "JSON (*.json)")
        if path and hasattr(self, 'relatorio'):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.relatorio, f, indent=4, ensure_ascii=False)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, e):
        if self.analise:
            self.analise.stop()
        super().closeEvent(e)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None,
                        help="Caminho do vídeo, 0 para webcam, ou URL do stream")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = JanelaTatica(source=args.video)
    sys.exit(app.exec_())