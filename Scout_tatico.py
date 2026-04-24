"""
SCOUT TÁTICO — Análise do Time Adversário
==========================================
Analisa vídeos do adversário e gera relatório completo para o treinador.

Uso:
  python scout_tatico.py --video jogo_adversario.mp4
  python scout_tatico.py --video jogo1.mp4 jogo2.mp4 jogo3.mp4  (múltiplos jogos)
"""

import os, sys, cv2, json, argparse, threading, time
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFrame,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QProgressBar, QScrollArea, QGridLayout, QTabWidget
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal
from PyQt5.QtGui   import QImage, QPixmap, QFont, QColor

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/detect/runs/detect/futebol_v22/weights/best.pt"
CONF        = 0.30
CONF_BOLA   = 0.55   # confiança mínima para aceitar uma bola (mais alto = menos falsos positivos)
BOLA_MAX_AREA = 0.002  # área máxima da bola em relação ao frame (filtra objetos grandes)
IMGSZ       = 416    # otimo equilibrio velocidade/precisao
SKIP_FRAMES = 3      # processa 1 a cada 3 frames — quase tempo real

CLASSES_JOGADOR = ("player", "goalkeeper")
CLASSE_BOLA     = "ball"
CLASSE_ARBITRO  = "referee"

# Cores para o time adversário (vermelho) e nosso time (azul)
COR_ADV     = ( 50,  50, 220)   # vermelho (BGR)
COR_NOSSO   = (220, 100,  40)   # azul (BGR)
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

# Zonas do campo (3x3 grid)
ZONAS = {
    (0,0): "Defesa Esquerda",   (1,0): "Defesa Central",   (2,0): "Defesa Direita",
    (0,1): "Meio Esquerdo",     (1,1): "Centro",            (2,1): "Meio Direito",
    (0,2): "Ataque Esquerdo",   (1,2): "Ataque Central",   (2,2): "Ataque Direito",
}


# ── STREAM READER ──────────────────────────────────────────────────────────────
class StreamReader:
    def __init__(self, src):
        self.cap     = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame   = None
        self.ret     = False
        self.running = True
        self.lock    = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

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


# ── CLASSIFICADOR DE TIME ──────────────────────────────────────────────────────
class ClassificadorTimes:
    """
    Separa adversário (Time A) do nosso time (Time B).
    Suporta calibração por clique na interface ou automática por K-Means.
    """
    def __init__(self):
        self.centro_a  = None
        self.centro_b  = None
        self.calibrado = False
        self.amostras  = []
        self.N_CAL     = 40
        print("[Classificador] Aguardando calibracao — clique nos jogadores na tela")

    def _roi(self, frame, x1, y1, x2, y2):
        h, w = y2-y1, x2-x1
        ry1 = y1 + int(h*0.20); ry2 = y1 + int(h*0.65)
        rx1 = x1 + int(w*0.15); rx2 = x1 + int(w*0.85)
        ry1,ry2 = max(ry1,0), min(ry2, frame.shape[0]-1)
        rx1,rx2 = max(rx1,0), min(rx2, frame.shape[1]-1)
        roi = frame[ry1:ry2, rx1:rx2]
        return roi if roi.size > 0 else None

    def _cor(self, frame, x1, y1, x2, y2):
        roi = self._roi(frame, x1, y1, x2, y2)
        if roi is None: return None
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        grama = cv2.inRange(hsv, (35,40,40), (85,255,255))
        mask  = cv2.bitwise_not(grama)
        pix   = roi[mask > 0]
        return pix.mean(axis=0) if len(pix) >= 10 else roi.mean(axis=(0,1))

    def adicionar(self, frame, x1, y1, x2, y2):
        # Coleta amostras para calibração automática por K-Means
        if not self.calibrado:
            c = self._cor(frame, x1, y1, x2, y2)
            if c is not None:
                self.amostras.append(c.astype(np.float32))
            if len(self.amostras) >= self.N_CAL:
                self._calibrar_kmeans()

    def _calibrar_kmeans(self):
        dados = np.array(self.amostras, dtype=np.float32)
        crit  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, _, centros = cv2.kmeans(dados, 2, None, crit, 15, cv2.KMEANS_PP_CENTERS)
        # Time A = cluster mais saturado
        hsv_a = cv2.cvtColor(np.uint8([[centros[0].astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0][0]
        hsv_b = cv2.cvtColor(np.uint8([[centros[1].astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0][0]
        if hsv_a[1] >= hsv_b[1]:
            self.centro_a, self.centro_b = centros[0], centros[1]
        else:
            self.centro_a, self.centro_b = centros[1], centros[0]
        self.calibrado = True
        print(f"[Classificador] Calibrado automaticamente!")
        print(f"  Adv BGR: {self.centro_a.astype(int)}")
        print(f"  Nosso BGR: {self.centro_b.astype(int)}")

    def definir_cor_manual(self, cor_adv, cor_nosso):
        self.centro_a  = np.array(cor_adv,   dtype=np.float32)
        self.centro_b  = np.array(cor_nosso,  dtype=np.float32)
        self.calibrado = True
        print(f"[Classificador] Calibrado manualmente!")
        print(f"  Adv BGR: {self.centro_a.astype(int)}")
        print(f"  Nosso BGR: {self.centro_b.astype(int)}")

    def classificar(self, frame, x1, y1, x2, y2):
        c = self._cor(frame, x1, y1, x2, y2)
        if c is None: return 'A'
        if not self.calibrado:
            return 'A'  # antes de calibrar, tudo vai para A
        da = np.linalg.norm(c - self.centro_a)
        db = np.linalg.norm(c - self.centro_b)
        return 'A' if da <= db else 'B'


# ── ENGINE DE ANÁLISE TÁTICA ───────────────────────────────────────────────────
class AnaliseTatica:
    """
    Coleta e processa todos os dados táticos do adversário ao longo do vídeo.
    """
    def __init__(self):
        # Posições dos jogadores por frame (para mapa de calor e formação)
        self.posicoes_adv   = []   # lista de listas de (x,y) normalizados
        self.posicoes_nosso = []

        # Posse de bola
        self.posse_adv   = 0
        self.posse_nosso = 0

        # Formações detectadas
        self.hist_form_adv   = []
        self.hist_form_nosso = []

        # Pressão por zona (grid 3x3)
        self.pressao_adv   = np.zeros((3,3), dtype=np.float32)
        self.pressao_nosso = np.zeros((3,3), dtype=np.float32)

        # Mapa de calor (onde o adversário mais atua)
        self.mapa_calor_adv = np.zeros((68, 105), dtype=np.float32)

        # Transições (contra-ataques detectados)
        self.transicoes = 0
        self._tinha_posse_adv = False

        # Distância média entre linhas (compacidade)
        self.compacidade_frames = []

        # Velocidade média de transição
        self.vel_transicao = []

    def atualizar(self, frame_h, frame_w, jog_adv, jog_nosso, pos_bola):
        """Recebe posições do frame atual e atualiza todas as métricas."""
        if not jog_adv and not jog_nosso:
            return

        # Normaliza posições (0-1)
        norm_adv   = [(x/frame_w, y/frame_h) for x,y in jog_adv]
        norm_nosso = [(x/frame_w, y/frame_h) for x,y in jog_nosso]

        self.posicoes_adv.append(norm_adv)
        self.posicoes_nosso.append(norm_nosso)

        # Mapa de calor do adversário
        for x,y in norm_adv:
            mx = min(int(x * 105), 104)
            my = min(int(y * 68),  67)
            self.mapa_calor_adv[my, mx] += 1

        # Pressão por zona (grid 3x3)
        for x,y in norm_adv:
            zx = min(int(x * 3), 2)
            zy = min(int(y * 3), 2)
            self.pressao_adv[zy, zx] += 1
        for x,y in norm_nosso:
            zx = min(int(x * 3), 2)
            zy = min(int(y * 3), 2)
            self.pressao_nosso[zy, zx] += 1

        # Posse de bola
        if pos_bola and (jog_adv or jog_nosso):
            bx, by = pos_bola[0]/frame_w, pos_bola[1]/frame_h
            da = min(((bx-x)**2+(by-y)**2)**.5 for x,y in norm_adv) if norm_adv else 9999
            db = min(((bx-x)**2+(by-y)**2)**.5 for x,y in norm_nosso) if norm_nosso else 9999
            tem_posse_adv = da < db
            if tem_posse_adv:
                self.posse_adv += 1
            else:
                self.posse_nosso += 1

            # Detecta transições (mudança de posse)
            if self._tinha_posse_adv != tem_posse_adv:
                self.transicoes += 1
            self._tinha_posse_adv = tem_posse_adv

        # Compacidade do adversário (distância entre jogadores mais avançado e mais recuado)
        if len(jog_adv) >= 4:
            ys = sorted([y for _,y in norm_adv])
            compac = ys[-1] - ys[0]
            self.compacidade_frames.append(compac)

    def detectar_formacao(self, posicoes_y_norm):
        if len(posicoes_y_norm) < 5:
            return "?"
        yn = sorted(posicoes_y_norm)
        outfield = yn[1:] if len(yn) > 1 else yn
        z1 = sum(1 for y in outfield if y < 0.33)
        z2 = sum(1 for y in outfield if 0.33 <= y < 0.66)
        z3 = sum(1 for y in outfield if y >= 0.66)
        tupla  = tuple(z for z in [z1,z2,z3] if z > 0)
        melhor, menor = None, 999
        for ft, fn in FORMACOES.items():
            ft3 = ft[-3:]
            d   = sum(abs(a-b) for a,b in zip(ft3, tupla[:len(ft3)]))
            if d < menor:
                menor, melhor = d, fn
        return melhor or f"{z1}-{z2}-{z3}"

    def gerar_relatorio(self):
        """Compila todas as métricas em um relatório final."""
        total_posse = max(self.posse_adv + self.posse_nosso, 1)

        # Formação mais usada
        form_adv = Counter(self.hist_form_adv).most_common(1)
        form_adv = form_adv[0][0] if form_adv else "?"

        # Zona de maior pressão do adversário
        zona_max = np.unravel_index(self.pressao_adv.argmax(), self.pressao_adv.shape)
        zona_nome = ZONAS.get((zona_max[1], zona_max[0]), "Centro")

        # Zona mais fraca do adversário (menor pressão)
        pressao_temp = self.pressao_adv.copy()
        pressao_temp[pressao_temp == 0] = 9999
        zona_min = np.unravel_index(pressao_temp.argmin(), pressao_temp.shape)
        zona_fraca = ZONAS.get((zona_min[1], zona_min[0]), "Defesa")

        # Compacidade média
        compac_media = np.mean(self.compacidade_frames) if self.compacidade_frames else 0
        estilo = "Defensivo (bloco baixo)" if compac_media < 0.4 else \
                 "Equilibrado"             if compac_media < 0.6 else \
                 "Ofensivo (linha alta)"

        # Pontos fracos baseados na análise
        pontos_fracos = self._analisar_pontos_fracos(
            form_adv, zona_fraca, compac_media,
            round(self.posse_adv / total_posse * 100)
        )

        # Sugestões táticas
        sugestoes = self._gerar_sugestoes(form_adv, zona_fraca, compac_media)

        return {
            "data_analise":   datetime.now().strftime("%d/%m/%Y %H:%M"),
            "frames_analisados": len(self.posicoes_adv),
            "adversario": {
                "formacao_principal": form_adv,
                "posse_bola_pct":     round(self.posse_adv / total_posse * 100),
                "zona_pressao":       zona_nome,
                "zona_fraca":         zona_fraca,
                "estilo_jogo":        estilo,
                "compacidade":        round(float(compac_media), 2),
                "transicoes_sofridas": self.transicoes // 2,
            },
            "pontos_fracos":  pontos_fracos,
            "sugestoes_taticas": sugestoes,
            "pressao_por_zona": {
                ZONAS[(j,i)]: round(float(self.pressao_adv[i,j]))
                for i in range(3) for j in range(3)
            }
        }

    def _analisar_pontos_fracos(self, form, zona_fraca, compac, posse_pct):
        fracos = []

        # Baseado na formação
        form_fracos = {
            "4-3-3":   "Espaço nas costas dos laterais quando atacam",
            "4-4-2":   "Meio-campo pode ficar exposto em transições rápidas",
            "4-2-3-1": "Flancos abertos quando o meia avançado sobe",
            "3-5-2":   "Pontas vulneráveis a bolas em profundidade",
            "3-4-3":   "Defesa de 3 pode ser explorada com 2 atacantes",
            "5-3-2":   "Bolas paradas e chutes de fora da área",
            "5-4-1":   "Time reativo — vulnerável quando pressão alta falha",
            "4-5-1":   "Ataque isolado, fácil de neutralizar o centroavante",
        }
        if form in form_fracos:
            fracos.append(f"Formação {form}: {form_fracos[form]}")

        # Baseado na zona fraca
        fracos.append(f"Zona menos coberta: {zona_fraca} — exploit com bolas nessa região")

        # Baseado na compacidade
        if compac < 0.35:
            fracos.append("Bloco muito compacto — difícil de penetrar diretamente, use velocidade nas pontas")
        elif compac > 0.65:
            fracos.append("Linha defensiva alta — vulnerável a bolas em profundidade nas costas da defesa")

        # Baseado na posse
        if posse_pct > 60:
            fracos.append("Time com muita posse — pressão alta pode forçar erros de passe")
        elif posse_pct < 40:
            fracos.append("Time reativo — pode ser surpreendido com posse organizada e paciência")

        return fracos

    def _gerar_sugestoes(self, form, zona_fraca, compac):
        sugestoes = []

        # Sugestão de formação para enfrentar
        contra_form = {
            "4-3-3":   "Use 4-5-1 para fechar o meio e explorar contra-ataques",
            "4-4-2":   "Use 4-3-3 para ter superioridade no meio-campo",
            "4-2-3-1": "Use 4-4-2 para marcar o meia avançado e os pivôs",
            "3-5-2":   "Use 4-3-3 com pontas rápidos para explorar os flancos",
            "3-4-3":   "Use 4-4-2 para neutralizar as pontas e criar superioridade",
            "5-3-2":   "Use 3-4-3 para pressionar alto e criar superioridade",
            "5-4-1":   "Use 4-3-3 para ter amplitude e forçar erros",
            "4-5-1":   "Use 4-3-3 para criar superioridade numérica no ataque",
        }
        if form in contra_form:
            sugestoes.append(f"Formação recomendada: {contra_form[form]}")

        # Sugestão baseada na zona fraca
        sugestoes.append(f"Direcione as jogadas para: {zona_fraca}")

        # Sugestão baseada na compacidade
        if compac > 0.55:
            sugestoes.append("Use bolas longas nas costas da defesa — linha alta é vulnerável")
            sugestoes.append("Pressione alto para forçar erros do goleiro")
        else:
            sugestoes.append("Tenha paciência na construção — evite passes longos sem apoio")
            sugestoes.append("Use laterais para abrir espaços e criar cruzamentos")

        sugestoes.append("Treine bolas paradas — podem ser decisivas contra esse adversário")

        return sugestoes


# ── THREAD DE ANÁLISE ──────────────────────────────────────────────────────────

# ── KALMAN FILTER PARA RASTREAMENTO DA BOLA ────────────────────────────────────
class KalmanBola:
    """
    Prediz a posição da bola usando Filtro de Kalman.
    Quando a bola some por motion blur, o sistema prevê
    onde ela vai estar baseado na velocidade e direção.
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        # Estado: [x, y, vx, vy] — posição e velocidade
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix  = np.array(
            [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.inicializado = False
        self.frames_sem_deteccao = 0
        self.MAX_SEM_DETECCAO    = 15  # frames máximos sem detecção real

    def atualizar(self, pos):
        """Atualiza com posição real detectada."""
        meas = np.array([[np.float32(pos[0])],[np.float32(pos[1])]])
        if not self.inicializado:
            self.kf.statePre = np.array(
                [[pos[0]],[pos[1]],[0],[0]], np.float32)
            self.inicializado = True
        self.kf.correct(meas)
        self.frames_sem_deteccao = 0
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def prever(self):
        """Prevê posição quando bola não foi detectada."""
        if not self.inicializado:
            return None
        self.frames_sem_deteccao += 1
        if self.frames_sem_deteccao > self.MAX_SEM_DETECCAO:
            return None
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def resetar(self):
        self.inicializado = False
        self.frames_sem_deteccao = 0


class ScoutThread(QThread):
    frame_ready   = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(dict)
    progresso     = pyqtSignal(int)
    finalizado    = pyqtSignal(dict)

    def __init__(self, source):
        super().__init__()
        self.source      = source
        self.running     = False
        self.model       = YOLO(MODEL_PATH)
        self.clf         = ClassificadorTimes()
        self.analise     = AnaliseTatica()
        self.frame_count = 0
        self.total_frames = 1
        self._ultimo_frame = None
        self._frame_atual  = None
        # Memória de posições — mantém última posição conhecida
        # quando jogador ou bola não é detectado (motion blur)
        self._ultima_bola     = None   # última posição conhecida da bola
        self._bola_ttl        = 0      # frames que a bola pode ficar sem detecção
        self._ultimos_jog_adv  = []    # últimas posições do time adversário
        self._ultimos_jog_nos  = []    # últimas posições do nosso time
        self._jog_ttl          = 0     # frames que jogadores podem ficar sem detecção
        self._kalman_bola      = KalmanBola()  # Filtro de Kalman para rastreamento da bola
        print(f"[Modelo] Classes: {self.model.names}")

    def run(self):
        src = 0 if self.source == "0" else self.source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
        self.running = True

        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir: {src}")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        if fps_video <= 0 or fps_video > 120:
            fps_video = 30.0
        delay_frame = 1.0 / fps_video
        print(f"[Stream] Aberto com sucesso!")
        print(f"[FPS] Video: {fps_video:.1f} | Delay por frame: {delay_frame*1000:.1f}ms")
        t_prev = time.time()

        while self.running:
            t_inicio = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            self.frame_count += 1
            now   = time.time()
            fps   = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            if self.total_frames > 1:
                self.progresso.emit(
                    int(self.frame_count / self.total_frames * 100))

            # Pula frames
            if SKIP_FRAMES > 1 and self.frame_count % SKIP_FRAMES != 0:
                if self._ultimo_frame is not None:
                    self.frame_ready.emit(self._ultimo_frame)
                elapsed = time.time() - t_inicio
                espera  = delay_frame - elapsed
                if espera > 0:
                    time.sleep(espera)
                else:
                    time.sleep(0.001)
                continue

            h, w = frame.shape[:2]

            # Detecção
            try:
                results = self.model.track(
                    frame, persist=True, conf=CONF,
                    iou=0.35, verbose=False, imgsz=IMGSZ)
            except Exception as e:
                print(f"[ERRO] {e}")
                self.frame_ready.emit(frame)
                continue

            jog_adv, jog_nosso = [], []
            pos_bola = None

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()
                ids_t   = results[0].boxes.id
                ids     = (ids_t.cpu().numpy().astype(int)
                           if ids_t is not None else np.arange(len(boxes)))

                # 1ª passagem: calibra classificador
                for box, cls in zip(boxes, classes):
                    if self.model.names[cls].lower() in CLASSES_JOGADOR:
                        x1,y1,x2,y2 = map(int, box)
                        x1,y1 = max(x1,0), max(y1,0)
                        x2,y2 = min(x2,w-1), min(y2,h-1)
                        self.clf.adicionar(frame, x1, y1, x2, y2)

                # 2ª passagem: desenha e classifica
                for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                    x1,y1,x2,y2 = map(int, box)
                    x1,y1 = max(x1,0), max(y1,0)
                    x2,y2 = min(x2,w-1), min(y2,h-1)
                    cx,cy  = (x1+x2)//2, (y1+y2)//2
                    nome   = self.model.names[cls].lower()

                    if nome == CLASSE_BOLA:
                        # Filtro 1: confiança mínima maior para bola
                        if conf < CONF_BOLA:
                            continue
                        # Filtro 2: tamanho máximo (bola é pequena)
                        area_bola = (x2-x1) * (y2-y1)
                        area_frame = h * w
                        if area_bola > BOLA_MAX_AREA * area_frame:
                            continue
                        # Filtro 3: só aceita a bola com maior confiança do frame
                        if pos_bola is None or conf > pos_bola[2]:
                            pos_bola = (cx, cy, float(conf))
                        continue  # não desenha ainda

                    elif nome == CLASSE_ARBITRO:
                        cv2.rectangle(frame, (x1,y1),(x2,y2), COR_ARBITRO, 1)

                    elif nome in CLASSES_JOGADOR:
                        time_clf = self.clf.classificar(frame, x1, y1, x2, y2)

                        if time_clf == 'A':
                            cor = COR_ADV
                            jog_adv.append((cx, cy))
                        else:
                            cor = COR_NOSSO
                            jog_nosso.append((cx, cy))

                        cv2.rectangle(frame, (x1,y1),(x2,y2), cor, 2)
                        prefixo = "G" if nome == "goalkeeper" else "J"
                        lbl = f"{prefixo}{tid} {conf:.0%}"
                        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                        cv2.rectangle(frame, (x1,y1-th-8),(x1+tw+6,y1), cor, -1)
                        cv2.putText(frame, lbl, (x1+3,y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
                        bw = int(conf*(x2-x1))
                        cv2.rectangle(frame,(x1,y2+3),(x2,y2+8),(30,30,30),-1)
                        cv2.rectangle(frame,(x1,y2+3),(x1+bw,y2+8), cor,-1)

            TTL_JOG = 5

            # Desenha bola detectada pelo YOLO
            if pos_bola is not None:
                bx, by = pos_bola[0], pos_bola[1]
                bc     = pos_bola[2] if len(pos_bola) > 2 else 1.0
                pos_bola = (bx, by)
                cv2.circle(frame, (bx, by), 10, COR_BOLA, -1)
                cv2.circle(frame, (bx, by), 12, (255,255,255), 2)
                cv2.putText(frame, f"Bola {bc:.0%}", (bx+13, by+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_BOLA, 1)

            # Jogadores — usa últimas posições se frame atual tem poucos
            if len(jog_adv) >= 3:
                self._ultimos_jog_adv = jog_adv
                self._jog_ttl = TTL_JOG
            elif self._jog_ttl > 0 and self._ultimos_jog_adv:
                self._jog_ttl -= 1
                # Complementa com últimas posições conhecidas
                jog_adv = jog_adv if len(jog_adv) >= 3 else self._ultimos_jog_adv

            if len(jog_nosso) >= 3:
                self._ultimos_jog_nos = jog_nosso
            elif self._ultimos_jog_nos:
                jog_nosso = jog_nosso if len(jog_nosso) >= 3 else self._ultimos_jog_nos

            # Atualiza análise tática
            self.analise.atualizar(h, w, jog_adv, jog_nosso, pos_bola)

            # Formação a cada 10 frames
            if self.frame_count % 10 == 0:
                if jog_adv:
                    fa = self.analise.detectar_formacao(
                        [y/h for _,y in jog_adv])
                    self.analise.hist_form_adv.append(fa)
                if jog_nosso:
                    fn = self.analise.detectar_formacao(
                        [y/h for _,y in jog_nosso])
                    self.analise.hist_form_nosso.append(fn)

            # Overlay
            total_det = len(results[0].boxes) if results and results[0].boxes is not None else 0
            cv2.rectangle(frame, (6,6),(380,75),(0,0,0),-1)
            cv2.putText(frame,
                f"Frame:{self.frame_count} | Det:{total_det} | FPS:{fps:.1f}",
                (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(255,165,0),1)
            adv_form = Counter(self.analise.hist_form_adv[-20:]).most_common(1)
            adv_form = adv_form[0][0] if adv_form else "?"
            tp = max(self.analise.posse_adv + self.analise.posse_nosso, 1)
            posse_adv = round(self.analise.posse_adv / tp * 100)
            cv2.putText(frame,
                f"ADV(vm):{len(jog_adv)} form:{adv_form} posse:{posse_adv}%",
                (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.48,(50,50,255),1)

            # Stats para o painel
            self.stats_updated.emit({
                "jog_adv":    len(jog_adv),
                "jog_nosso":  len(jog_nosso),
                "form_adv":   adv_form,
                "posse_adv":  posse_adv,
                "posse_nosso": 100 - posse_adv,
                "frame":      self.frame_count,
                "total":      self.total_frames,
            })

            self._ultimo_frame = frame.copy()
            self._frame_atual  = frame.copy()
            self.frame_ready.emit(frame.copy())

            elapsed = time.time() - t_inicio
            espera  = delay_frame - elapsed
            if espera > 0:
                time.sleep(espera)
            else:
                time.sleep(0.001)

        cap.release()

        # Gera relatório final
        rel = self.analise.gerar_relatorio()
        with open("scout_relatorio.json", "w", encoding="utf-8") as f:
            json.dump(rel, f, indent=4, ensure_ascii=False)
        print("\n✅ Scout concluído!")
        print(json.dumps(rel, indent=2, ensure_ascii=False))
        self.finalizado.emit(rel)
        self.progresso.emit(100)

    def stop(self):
        self.running = False
        self.wait()


# ── PAINEL DE SCOUT ────────────────────────────────────────────────────────────
class PainelScout(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(380)
        self.setStyleSheet("background:#0D1B2A;")
        self._build()

    def _sep(self, lay):
        s = QFrame(); s.setFrameShape(QFrame.HLine)
        s.setStyleSheet("background:#1E3040; border:none; max-height:1px;")
        lay.addWidget(s)

    def _lbl(self, lay, txt, cor, size=11, bold=True):
        l = QLabel(txt)
        l.setStyleSheet(f"color:{cor}; font-size:{size}px;"
                        f" font-weight:{'bold' if bold else 'normal'};"
                        f" background:transparent;")
        l.setWordWrap(True)
        lay.addWidget(l)
        return l

    def _bar(self, lay, cor):
        b = QProgressBar()
        b.setRange(0,100); b.setValue(0)
        b.setFixedHeight(14); b.setFormat("%p% posse")
        b.setStyleSheet(f"""
            QProgressBar {{ background:#1E3040; border-radius:6px;
                            border:none; color:#ECEFF1; font-size:10px; }}
            QProgressBar::chunk {{ background:{cor}; border-radius:6px; }}""")
        lay.addWidget(b)
        return b

    def _build(self):
        # Scroll area para o painel
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; background:#0D1B2A; }")
        scroll.setFixedWidth(380)

        content = QWidget()
        content.setStyleSheet("background:#0D1B2A;")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(16,16,16,16)
        lay.setSpacing(10)

        # Título
        self._lbl(lay, "Scout Tático", "#FF6B6B", 16)
        self._lbl(lay, "Análise do Time Adversário", "#78909C", 10, False)
        self._sep(lay)

        # Adversário — ao vivo
        self._lbl(lay, "ADVERSÁRIO (Vermelho)", "#FF6B6B")
        self.lbl_form_adv = self._lbl(lay, "Formação: ?", "#FF6B6B", 14)
        self.lbl_jog_adv  = self._lbl(lay, "Jogadores: 0", "#78909C", 12, False)
        self.bar_adv      = self._bar(lay, "#E74C3C")
        self._sep(lay)

        # Nosso time
        self._lbl(lay, "NOSSO TIME (Azul)", "#4FC3F7")
        self.lbl_form_nosso = self._lbl(lay, "Formação: ?", "#4FC3F7", 14)
        self.lbl_jog_nosso  = self._lbl(lay, "Jogadores: 0", "#78909C", 12, False)
        self.bar_nosso      = self._bar(lay, "#378ADD")
        self._sep(lay)

        # Progresso
        self._lbl(lay, "Progresso da Análise", "#78909C")
        self.prog = QProgressBar()
        self.prog.setRange(0,100); self.prog.setValue(0)
        self.prog.setFixedHeight(8); self.prog.setTextVisible(False)
        self.prog.setStyleSheet("""
            QProgressBar { background:#1E3040; border-radius:4px; border:none; }
            QProgressBar::chunk { background:#FF6B6B; border-radius:4px; }""")
        lay.addWidget(self.prog)
        self.lbl_frame = self._lbl(lay, "Frame: 0 / 0", "#37474F", 10, False)
        self._sep(lay)

        # Relatório final (aparece quando termina)
        self._lbl(lay, "Pontos Fracos do Adversário", "#FFA726")
        self.lbl_fracos = self._lbl(lay,
            "Aguardando análise completa...", "#ECEFF1", 11, False)
        self._sep(lay)

        self._lbl(lay, "Sugestões Táticas", "#66BB6A")
        self.lbl_sugestoes = self._lbl(lay,
            "Aguardando análise completa...", "#ECEFF1", 11, False)
        self._sep(lay)

        self._lbl(lay, "Zona de Maior Pressão", "#FF6B6B")
        self.lbl_zona_pressao = self._lbl(lay, "—", "#ECEFF1", 12)

        self._lbl(lay, "Zona Mais Fraca", "#66BB6A")
        self.lbl_zona_fraca = self._lbl(lay, "—", "#ECEFF1", 12)

        self._lbl(lay, "Estilo de Jogo", "#4FC3F7")
        self.lbl_estilo = self._lbl(lay, "—", "#ECEFF1", 12)
        self._sep(lay)

        self.btn_sal = QPushButton("Salvar Relatório Scout (JSON)")
        self.btn_sal.setEnabled(False)
        self.btn_sal.setStyleSheet("""
            QPushButton { background:rgba(255,107,107,0.15); color:#FF6B6B;
                          border:1px solid rgba(255,107,107,0.4);
                          border-radius:8px; padding:8px; font-size:13px; }
            QPushButton:hover { background:rgba(255,107,107,0.3); }
            QPushButton:disabled { color:#37474F; border-color:#1E3040; }""")
        lay.addWidget(self.btn_sal)
        lay.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        outer.addWidget(scroll)

    def atualizar(self, s):
        self.lbl_form_adv.setText(f"Formação: {s['form_adv']}")
        self.lbl_jog_adv.setText(f"Jogadores em campo: {s['jog_adv']}")
        self.lbl_form_nosso.setText(f"Formação: ?")
        self.lbl_jog_nosso.setText(f"Jogadores em campo: {s['jog_nosso']}")
        self.bar_adv.setValue(s['posse_adv'])
        self.bar_nosso.setValue(s['posse_nosso'])
        self.lbl_frame.setText(
            f"Frame: {s['frame']} / {s['total']}")

    def mostrar_relatorio(self, rel):
        adv = rel.get("adversario", {})

        # Pontos fracos
        fracos = rel.get("pontos_fracos", [])
        self.lbl_fracos.setText(
            "\n\n".join(f"⚠ {f}" for f in fracos) if fracos
            else "Nenhum ponto fraco identificado")

        # Sugestões
        sug = rel.get("sugestoes_taticas", [])
        self.lbl_sugestoes.setText(
            "\n\n".join(f"✅ {s}" for s in sug) if sug
            else "Sem sugestões")

        self.lbl_zona_pressao.setText(adv.get("zona_pressao", "—"))
        self.lbl_zona_fraca.setText(adv.get("zona_fraca", "—"))
        self.lbl_estilo.setText(adv.get("estilo_jogo", "—"))
        self.btn_sal.setEnabled(True)

    def set_prog(self, v):
        self.prog.setValue(v)


# ── JANELA PRINCIPAL ───────────────────────────────────────────────────────────
class JanelaScout(QMainWindow):
    def __init__(self, source=None):
        super().__init__()
        self.setWindowTitle("Scout Tático — Análise do Adversário")
        self.showFullScreen()
        self.setStyleSheet("background:#0D1B2A;")
        self.scout       = None
        self.relatorio   = None
        self._calibrando   = False
        self._passo_calib  = 0
        self._cor_adv_temp = None
        self._frame_bgr    = None
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

        # Área do vídeo
        cam = QWidget(); cam.setStyleSheet("background:#000;")
        cl  = QVBoxLayout(cam); cl.setContentsMargins(0,0,0,0)

        self.btn_vid = QPushButton("Carregar Vídeo do Adversário", cam)
        self.btn_vid.setStyleSheet("""
            QPushButton { background:rgba(255,107,107,0.2); color:#FF6B6B;
                          border:2px solid rgba(255,107,107,0.5);
                          border-radius:10px; padding:12px 24px; font-size:16px; }
            QPushButton:hover { background:rgba(255,107,107,0.4); }""")
        self.btn_vid.clicked.connect(self._carregar)
        self.btn_vid.move(20, 20)

        self.vid_lbl = QLabel()
        self.vid_lbl.setAlignment(Qt.AlignCenter)
        self.vid_lbl.setScaledContents(True)
        self.vid_lbl.setStyleSheet("background:#000;")
        cl.addWidget(self.vid_lbl)
        lay.addWidget(cam, 65)

        div = QFrame(); div.setFrameShape(QFrame.VLine)
        div.setStyleSheet("background:#1E3040; border:none; max-width:1px;")
        lay.addWidget(div)

        self.painel = PainelScout()
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

        # Botao calibrar times por clique
        self.btn_calibrar = QPushButton('🎯 Calibrar Times', central)
        self.btn_calibrar.setFixedSize(150, 32)
        self.btn_calibrar.setCheckable(True)
        self.btn_calibrar.setStyleSheet("""
            QPushButton { background:rgba(255,165,0,0.2); color:#FFA726;
                          border:1px solid rgba(255,165,0,0.5);
                          border-radius:8px; font-size:12px; }
            QPushButton:hover { background:rgba(255,165,0,0.4); }
            QPushButton:checked { background:rgba(255,165,0,0.6); }""")
        self.btn_calibrar.clicked.connect(self._modo_calibracao)
        self.btn_calibrar.hide()

        # Label de instrucao
        self.lbl_instrucao = QLabel('', central)
        self.lbl_instrucao.setStyleSheet(
            'color:#FFA726; font-size:13px; background:rgba(0,0,0,180);'
            ' padding:6px 12px; border-radius:6px;')
        self.lbl_instrucao.setFixedWidth(500)
        self.lbl_instrucao.hide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'btn_sair'):
            self.btn_sair.move(self.width() - 400, 10)
        if hasattr(self, 'btn_vid'):
            self.btn_vid.move(20, 20)
        if hasattr(self, 'btn_calibrar'):
            self.btn_calibrar.move(self.width() - 570, 10)
        if hasattr(self, 'lbl_instrucao'):
            self.lbl_instrucao.move(20, self.height() - 50)

    def _carregar(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecione o vídeo do adversário", "",
            "Vídeos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._iniciar(path)

    def _iniciar(self, src):
        if self.scout:
            self.scout.stop()
        self.btn_vid.hide()
        self.btn_calibrar.show()
        self.scout = ScoutThread(src)
        self.scout.frame_ready.connect(self._show)
        self.scout.stats_updated.connect(self.painel.atualizar)
        self.scout.progresso.connect(self.painel.set_prog)
        self.scout.finalizado.connect(self._concluido)
        self.scout.start()

    def _modo_calibracao(self, checked):
        self._calibrando = checked
        self._passo_calib = 0
        self._cor_adv_temp = None
        if checked:
            self.lbl_instrucao.setText(
                "🔴 Passo 1/2: Clique em um jogador do TIME ADVERSÁRIO")
            self.lbl_instrucao.show()
            self.vid_lbl.setCursor(Qt.CrossCursor)
        else:
            self.lbl_instrucao.hide()
            self.vid_lbl.setCursor(Qt.ArrowCursor)

    def _show(self, frame):
        self._frame_bgr = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.vid_lbl.setPixmap(QPixmap.fromImage(img))

    def mousePressEvent(self, e):
        if not self._calibrando or not self.scout:
            return
        if not self.vid_lbl.geometry().contains(e.pos()):
            return
        if not hasattr(self, '_frame_bgr') or self._frame_bgr is None:
            return

        # Converte clique da tela para coordenadas do frame
        vx = self.vid_lbl.x()
        vy = self.vid_lbl.y()
        vw = self.vid_lbl.width()
        vh = self.vid_lbl.height()
        fh, fw = self._frame_bgr.shape[:2]

        px = int((e.pos().x() - vx) / vw * fw)
        py = int((e.pos().y() - vy) / vh * fh)
        px = max(0, min(px, fw-1))
        py = max(0, min(py, fh-1))

        # Pega cor média numa região 20x20 ao redor do clique
        x1 = max(0, px-10); x2 = min(fw, px+10)
        y1 = max(0, py-10); y2 = min(fh, py+10)
        roi = self._frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return
        cor = roi.mean(axis=(0,1))

        if self._passo_calib == 0:
            # Primeiro clique — time adversário
            self._cor_adv_temp = cor
            self._passo_calib  = 1
            b,g,r = int(cor[0]),int(cor[1]),int(cor[2])
            self.lbl_instrucao.setText(
                f"✅ Adversário salvo (BGR:{b},{g},{r}) | "
                f"🔵 Passo 2/2: Clique em um jogador do SEU TIME")
        else:
            # Segundo clique — nosso time
            cor_nosso = cor
            self.scout.clf.definir_cor_manual(
                self._cor_adv_temp.tolist(),
                cor_nosso.tolist()
            )
            b,g,r = int(cor_nosso[0]),int(cor_nosso[1]),int(cor_nosso[2])
            self.lbl_instrucao.setText(
                f"✅ Times calibrados! Adversário=Vermelho | Nosso=Azul")
            # Desativa modo calibração após 3 segundos
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(3000, lambda: self._modo_calibracao(False))
            self.btn_calibrar.setChecked(False)
            self._calibrando = False
            self.vid_lbl.setCursor(Qt.ArrowCursor)

    def _concluido(self, rel):
        self.relatorio = rel
        self.painel.mostrar_relatorio(rel)

    def _salvar(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar Scout", "scout_relatorio.json", "JSON (*.json)")
        if path and self.relatorio:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.relatorio, f, indent=4, ensure_ascii=False)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, e):
        if self.scout:
            self.scout.stop()
        super().closeEvent(e)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None,
                        help="Vídeo do adversário para analisar")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = JanelaScout(source=args.video)
    sys.exit(app.exec_())