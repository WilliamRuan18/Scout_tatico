"""
PROCESSADOR DE VÍDEO — Scout Tático (Modo Background)
Uso:
  python processar_video.py --video jogo.mp4 --adversario "Real Madrid"
  python painel_web.py  →  http://localhost:5000
"""

import os, cv2, json, argparse, time, sqlite3
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = "runs/detect/runs/detect/futebol_v22/weights/best.pt"
CONF          = 0.20
CONF_BOLA     = 0.15
BOLA_MAX_AREA = 0.002
IMGSZ         = 1280
SKIP_FRAMES   = 1
CONF_FORM     = 15   # frames consecutivos para confirmar formacao

CLASSES_JOGADOR = ("player", "goalkeeper")
CLASSE_BOLA     = "ball"

FORMACOES = {
    (4,3,3):   "4-3-3",   (4,4,2):   "4-4-2",
    (4,2,3,1): "4-2-3-1", (3,5,2):   "3-5-2",
    (3,4,3):   "3-4-3",   (5,3,2):   "5-3-2",
    (5,4,1):   "5-4-1",   (4,5,1):   "4-5-1",
    (4,1,4,1): "4-1-4-1",
}

ZONAS = {
    (0,0): "Defesa Esquerda",  (1,0): "Defesa Central",  (2,0): "Defesa Direita",
    (0,1): "Meio Esquerdo",    (1,1): "Centro",           (2,1): "Meio Direito",
    (0,2): "Ataque Esquerdo",  (1,2): "Ataque Central",   (2,2): "Ataque Direito",
}

DB_PATH = "scout_historico.db"


# ── BANCO DE DADOS ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jogos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT, adversario TEXT, video TEXT,
            formacao TEXT, posse_adv INTEGER,
            zona_pressao TEXT, zona_fraca TEXT,
            estilo TEXT, compacidade REAL,
            frames INTEGER, relatorio TEXT
        )
    """)
    conn.commit(); conn.close()


def salvar_no_banco(relatorio, adversario, video):
    conn = sqlite3.connect(DB_PATH)
    adv  = relatorio.get("adversario", {})
    conn.execute("""
        INSERT INTO jogos (data,adversario,video,formacao,posse_adv,
        zona_pressao,zona_fraca,estilo,compacidade,frames,relatorio)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        relatorio.get("data_analise", datetime.now().strftime("%d/%m/%Y %H:%M")),
        adversario, video,
        adv.get("formacao_principal", "?"),
        adv.get("posse_bola_pct", 0),
        adv.get("zona_pressao", "?"),
        adv.get("zona_fraca", "?"),
        adv.get("estilo_jogo", "?"),
        adv.get("compacidade", 0),
        relatorio.get("frames_analisados", 0),
        json.dumps(relatorio, ensure_ascii=False),
    ))
    conn.commit(); conn.close()
    print("[DB] Salvo no banco!")


# ── CLASSIFICADOR DE TIMES ─────────────────────────────────────────────────────
# Cores de camisa disponiveis para --cor_adv
CORES_CAMISA = {
    "branco":   ([0,   0,  180], [180,  40, 255]),   # HSV: baixa saturacao, alto brilho
    "preto":    ([0,   0,    0], [180,  60,  80]),
    "vermelho": ([0,  120,   70], [10,  255, 255]),   # vermelho HSV (0-10 e 170-180)
    "vermelho2":([170,120,   70], [180, 255, 255]),
    "azul":     ([100, 80,   50], [130, 255, 255]),
    "azul_claro":([85, 60,  100], [105, 255, 255]),
    "amarelo":  ([20, 100,  100], [35,  255, 255]),
    "verde":    ([36,  80,   50], [85,  255, 255]),
    "laranja":  ([10, 120,  100], [20,  255, 255]),
    "roxo":     ([130, 60,   50], [160, 255, 255]),
    "cinza":    ([0,   0,   90], [180,  40, 180]),
}

class ClassificadorTimes:
    """
    Classifica jogadores pelo HSV da camisa.
    Se cor_adv for informada, usa mascara de cor direta (confiavel).
    Se nao, usa kmeans automatico (pode errar).
    """
    def __init__(self, cor_adv=None):
        # cor_adv: string como "branco", "azul", "vermelho", etc.
        self.cor_adv   = cor_adv
        self.amostras  = []
        self.centro_a  = None   # usado so no modo automatico
        self.centro_b  = None
        self.calibrado = False
        self.N_CAL     = 60

        if cor_adv:
            print(f"[Classificador] Modo cor fixa: adversario usa camisa '{cor_adv}'")
        else:
            print("[Classificador] Modo automatico (kmeans) — se errar, use --cor_adv")

    def _roi(self, frame, x1, y1, x2, y2):
        """Recorta so o tronco do jogador (evita cabeca e pernas)."""
        h, w = y2-y1, x2-x1
        ry1 = y1 + int(h*0.20)
        ry2 = y1 + int(h*0.65)
        rx1 = x1 + int(w*0.15)
        rx2 = x1 + int(w*0.85)
        ry1,ry2 = max(ry1,0), min(ry2, frame.shape[0]-1)
        rx1,rx2 = max(rx1,0), min(rx2, frame.shape[1]-1)
        roi = frame[ry1:ry2, rx1:rx2]
        return roi if roi.size > 0 else None

    def _cor_media(self, frame, x1, y1, x2, y2):
        """Retorna cor media BGR do tronco, excluindo grama."""
        roi = self._roi(frame, x1, y1, x2, y2)
        if roi is None: return None
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        grama = cv2.inRange(hsv, (35,40,40), (85,255,255))
        pix   = roi[cv2.bitwise_not(grama) > 0]
        return pix.mean(axis=0) if len(pix) >= 10 else roi.mean(axis=(0,1))

    def _pct_cor(self, frame, x1, y1, x2, y2, nome_cor):
        """
        Retorna % de pixels do tronco que batem com a cor informada.
        Lida com vermelho que aparece em duas faixas HSV.
        """
        roi = self._roi(frame, x1, y1, x2, y2)
        if roi is None: return 0.0
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        if total == 0: return 0.0

        lower = np.array(CORES_CAMISA[nome_cor][0], dtype=np.uint8)
        upper = np.array(CORES_CAMISA[nome_cor][1], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)

        # Vermelho tem duas faixas — combina as duas
        if nome_cor == "vermelho":
            lower2 = np.array(CORES_CAMISA["vermelho2"][0], dtype=np.uint8)
            upper2 = np.array(CORES_CAMISA["vermelho2"][1], dtype=np.uint8)
            mask   = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

        return cv2.countNonZero(mask) / total

    # ── Modo automatico (kmeans) ────────────────────────────────────────────
    def adicionar(self, frame, x1, y1, x2, y2):
        if self.cor_adv: return   # nao precisa no modo cor fixa
        c = self._cor_media(frame, x1, y1, x2, y2)
        if c is not None: self.amostras.append(c.astype(np.float32))
        if len(self.amostras) >= self.N_CAL and not self.calibrado:
            self._calibrar()

    def _calibrar(self):
        dados = np.array(self.amostras, dtype=np.float32)
        crit  = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centros = cv2.kmeans(dados, 2, None, crit, 15, cv2.KMEANS_PP_CENTERS)
        # O grupo com MAIS jogadores = time da casa (geralmente mais visivel)
        # O grupo com MENOS = adversario  →  heuristica simples
        c0 = np.sum(labels == 0)
        c1 = np.sum(labels == 1)
        # Adversario = grupo menor (time visitante tende a ter menos jogadores no campo visivel)
        if c0 <= c1:
            self.centro_a, self.centro_b = centros[0], centros[1]
        else:
            self.centro_a, self.centro_b = centros[1], centros[0]
        self.calibrado = True
        print(f"[Classificador] kmeans calibrado | grupo_adv={min(c0,c1)} | grupo_nosso={max(c0,c1)}")
        print(f"  Adv BGR aprox: {self.centro_a.astype(int)}")
        print(f"  Nosso BGR aprox: {self.centro_b.astype(int)}")
        print(f"  Se estiver trocado, rode com --cor_adv branco/azul/vermelho/etc")

    # ── Classificacao ───────────────────────────────────────────────────────
    def classificar(self, frame, x1, y1, x2, y2):
        if self.cor_adv:
            # Modo cor fixa: adversario se >= 15% dos pixels batem com a cor
            pct = self._pct_cor(frame, x1, y1, x2, y2, self.cor_adv)
            return 'A' if pct >= 0.15 else 'B'
        else:
            # Modo automatico: compara distancia aos centros kmeans
            c = self._cor_media(frame, x1, y1, x2, y2)
            if c is None or not self.calibrado: return 'B'
            return 'A' if np.linalg.norm(c-self.centro_a) <= np.linalg.norm(c-self.centro_b) else 'B'


# ── ANALISE TATICA ─────────────────────────────────────────────────────────────
class AnaliseTatica:
    def __init__(self):
        self.posse_adv = 0; self.posse_nosso = 0
        self.hist_form_adv = []; self.hist_form_nosso = []
        self.pressao_adv = np.zeros((3,3), dtype=np.float32)
        self.compacidade_frames = []; self.transicoes = 0
        self._tinha_posse = False

    def atualizar(self, h, w, jog_adv, jog_nosso, pos_bola):
        if not jog_adv and not jog_nosso: return
        norm_adv   = [(x/w, y/h) for x,y in jog_adv]
        norm_nosso = [(x/w, y/h) for x,y in jog_nosso]
        for x,y in norm_adv:
            self.pressao_adv[min(int(y*3),2), min(int(x*3),2)] += 1
        if pos_bola and (jog_adv or jog_nosso):
            bx,by = pos_bola[0]/w, pos_bola[1]/h
            da = min(((bx-x)**2+(by-y)**2)**.5 for x,y in norm_adv) if norm_adv else 9999
            db = min(((bx-x)**2+(by-y)**2)**.5 for x,y in norm_nosso) if norm_nosso else 9999
            tem = da < db
            if tem: self.posse_adv += 1
            else:   self.posse_nosso += 1
            if self._tinha_posse != tem: self.transicoes += 1
            self._tinha_posse = tem
        if len(jog_adv) >= 4:
            ys = sorted([y/h for _,y in jog_adv])
            self.compacidade_frames.append(ys[-1]-ys[0])

    def detectar_formacao(self, posicoes_y_norm):
        if len(posicoes_y_norm) < 5: return "?"
        yn = sorted(posicoes_y_norm)
        outfield = yn[1:] if len(yn) > 1 else yn
        z1 = sum(1 for y in outfield if y < 0.33)
        z2 = sum(1 for y in outfield if 0.33 <= y < 0.66)
        z3 = sum(1 for y in outfield if y >= 0.66)
        tupla = tuple(z for z in [z1,z2,z3] if z > 0)
        melhor, menor = None, 999
        for ft,fn in FORMACOES.items():
            d = sum(abs(a-b) for a,b in zip(ft[-3:], tupla[:3]))
            if d < menor: menor,melhor = d,fn
        return melhor or f"{z1}-{z2}-{z3}"

    def gerar_relatorio(self):
        total      = max(self.posse_adv+self.posse_nosso, 1)
        form       = (Counter(self.hist_form_adv).most_common(1) or [("?",)])[0][0]
        zona_max   = np.unravel_index(self.pressao_adv.argmax(), self.pressao_adv.shape)
        zona_nome  = ZONAS.get((zona_max[1],zona_max[0]), "Centro")
        pt = self.pressao_adv.copy(); pt[pt==0] = 9999
        zona_min   = np.unravel_index(pt.argmin(), pt.shape)
        zona_fraca = ZONAS.get((zona_min[1],zona_min[0]), "Defesa")
        compac     = float(np.mean(self.compacidade_frames)) if self.compacidade_frames else 0
        estilo     = ("Defensivo (bloco baixo)" if compac < 0.4 else
                      "Equilibrado"             if compac < 0.6 else
                      "Ofensivo (linha alta)")
        posse_pct  = round(self.posse_adv/total*100)

        form_fracos = {
            "4-3-3":   "Espaco nas costas dos laterais quando atacam",
            "4-4-2":   "Meio-campo exposto em transicoes rapidas",
            "4-2-3-1": "Flancos abertos quando meia avancado sobe",
            "3-5-2":   "Pontas vulneraveis a bolas em profundidade",
            "3-4-3":   "Defesa de 3 explorada com 2 atacantes",
            "5-3-2":   "Bolas paradas e chutes de fora da area",
            "5-4-1":   "Time reativo - vulneravel quando pressao alta falha",
            "4-5-1":   "Ataque isolado, facil neutralizar o centroavante",
        }
        fracos = []
        if form in form_fracos: fracos.append(f"Formacao {form}: {form_fracos[form]}")
        fracos.append(f"Zona menos coberta: {zona_fraca}")
        if compac < 0.35:    fracos.append("Bloco compacto - use velocidade nas pontas")
        elif compac > 0.65:  fracos.append("Linha alta - vulneravel a bolas em profundidade")
        if posse_pct > 60:   fracos.append("Muita posse - pressao alta pode forcar erros")
        elif posse_pct < 40: fracos.append("Time reativo - surpreenda com posse organizada")

        contra = {
            "4-3-3":   "Use 4-5-1 para fechar o meio e explorar contra-ataques",
            "4-4-2":   "Use 4-3-3 para superioridade no meio-campo",
            "4-2-3-1": "Use 4-4-2 para marcar o meia avancado",
            "3-5-2":   "Use 4-3-3 com pontas rapidos pelos flancos",
            "3-4-3":   "Use 4-4-2 para neutralizar as pontas",
            "5-3-2":   "Use 3-4-3 para pressionar alto",
            "5-4-1":   "Use 4-3-3 para amplitude e forcar erros",
            "4-5-1":   "Use 4-3-3 com superioridade no ataque",
        }
        sugestoes = []
        if form in contra: sugestoes.append(f"Formacao recomendada: {contra[form]}")
        sugestoes.append(f"Direcione jogadas para: {zona_fraca}")
        if compac > 0.55:
            sugestoes.append("Bolas longas nas costas da defesa")
            sugestoes.append("Pressione alto para forcar erros do goleiro")
        else:
            sugestoes.append("Paciencia na construcao - evite passes longos")
            sugestoes.append("Use laterais para abrir espacos e cruzamentos")
        sugestoes.append("Treine bolas paradas - podem ser decisivas")

        return {
            "data_analise":      datetime.now().strftime("%d/%m/%Y %H:%M"),
            "frames_analisados": len(self.hist_form_adv),
            "adversario": {
                "formacao_principal":  form,
                "posse_bola_pct":      posse_pct,
                "zona_pressao":        zona_nome,
                "zona_fraca":          zona_fraca,
                "estilo_jogo":         estilo,
                "compacidade":         round(compac, 2),
                "transicoes_sofridas": self.transicoes // 2,
            },
            "pontos_fracos":     fracos,
            "sugestoes_taticas": sugestoes,
            "pressao_por_zona":  {
                ZONAS[(j,i)]: round(float(self.pressao_adv[i,j]))
                for i in range(3) for j in range(3)
            }
        }


# ── CAPTURA SIMPLES E FUNCIONAL ────────────────────────────────────────────────
def salvar_captura(frame, jog_adv, formacao, adversario, minuto, segundo, caminho):
    """
    Captura minimalista:
    - Filtra jogadores fora do campo
    - Agrupa em linhas por Y
    - Linha verde conectando jogadores da mesma linha
    - Circulo vermelho em cada jogador
    - Tarja preta no topo com formacao e infos
    """
    img = frame.copy()
    h, w = img.shape[:2]

    # Filtra deteccoes claramente fora do campo
    # (remove jogadores acima da tarja do placar e abaixo do gramado)
    margem_top = int(h * 0.10)
    margem_bot = int(h * 0.90)
    jogs = [(x, y) for x, y in jog_adv
            if margem_top < y < margem_bot and 10 < x < w - 10]

    if not jogs:
        jogs = jog_adv  # fallback

    # Agrupa em linhas por proximidade vertical
    linhas = []
    if jogs:
        jogs_sort = sorted(jogs, key=lambda p: p[1])
        grupo = [jogs_sort[0]]
        TOL = h * 0.08  # tolerancia: 8% da altura

        for i in range(1, len(jogs_sort)):
            if abs(jogs_sort[i][1] - jogs_sort[i-1][1]) < TOL:
                grupo.append(jogs_sort[i])
            else:
                linhas.append(sorted(grupo, key=lambda p: p[0]))
                grupo = [jogs_sort[i]]
        linhas.append(sorted(grupo, key=lambda p: p[0]))

    # Desenha linhas verdes conectando jogadores da mesma linha
    for linha in linhas:
        for i in range(len(linha) - 1):
            p1 = (int(linha[i][0]),   int(linha[i][1]))
            p2 = (int(linha[i+1][0]), int(linha[i+1][1]))
            cv2.line(img, p1, p2, (0, 200, 60), 2, cv2.LINE_AA)

    # Circulo vermelho em cada jogador
    for x, y in jogs:
        cx, cy = int(x), int(y)
        cv2.circle(img, (cx, cy), 12, (0, 0, 200), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 12, (255, 255, 255), 2, cv2.LINE_AA)

    # Tarja preta no topo
    cv2.rectangle(img, (0, 0), (w, 55), (0, 0, 0), -1)

    # Formacao em verde
    cv2.putText(img, f"FORMACAO CONFIRMADA: {formacao}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 220, 80), 2, cv2.LINE_AA)

    # Segunda linha de info
    cv2.rectangle(img, (0, 55), (w, 85), (30, 30, 30), -1)
    info = f"Adversario: {adversario}   Minuto: {minuto:02d}:{segundo:02d}   Jogadores detectados: {len(jogs)}"
    cv2.putText(img, info,
                (10, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(caminho, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [FOTO] Salva: {caminho}")


# ── PROCESSADOR PRINCIPAL ──────────────────────────────────────────────────────
def processar(video_path, adversario, cor_adv=None):
    print("\n" + "="*55)
    print(f"  PROCESSANDO: {Path(video_path).name}")
    print(f"  Adversario : {adversario}")
    if cor_adv:
        print(f"  Cor camisa adversario: {cor_adv}")
    else:
        print("  Cor camisa: automatica (se errar use --cor_adv)")
    print("="*55 + "\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Nao foi possivel abrir: {video_path}"); return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30
    h_frame      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_frame      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"[Video] {total_frames} frames | {fps_video:.0f} FPS | {total_frames/fps_video/60:.1f} min | {w_frame}x{h_frame}\n")

    model   = YOLO(MODEL_PATH)
    clf     = ClassificadorTimes(cor_adv=cor_adv)
    analise = AnaliseTatica()

    frame_count     = 0
    t_inicio        = time.time()
    t_ultimo        = t_inicio
    form_contagem   = {}
    capturas_salvas = set()
    pasta = Path(f"capturas_{Path(video_path).stem}")
    pasta.mkdir(exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        h, w = frame.shape[:2]

        # Log de progresso a cada 5s
        agora = time.time()
        if agora - t_ultimo >= 5:
            pct = frame_count / max(total_frames,1) * 100
            el  = agora - t_inicio
            re  = (el / max(frame_count,1)) * (total_frames - frame_count)
            print(f"  [{pct:5.1f}%] {frame_count}/{total_frames} | "
                  f"{el/60:.1f}min decorrido | {re/60:.1f}min restante")
            t_ultimo = agora

        if SKIP_FRAMES > 1 and frame_count % SKIP_FRAMES != 0:
            continue

        # Deteccao YOLO
        try:
            results = model.track(frame, persist=True, conf=CONF,
                                  iou=0.35, verbose=False, imgsz=IMGSZ)
        except:
            continue

        jog_adv, jog_nosso, pos_bola = [], [], None

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()
            ids_t   = results[0].boxes.id
            ids     = ids_t.cpu().numpy().astype(int) if ids_t is not None else np.arange(len(boxes))

            for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                x1,y1,x2,y2 = map(int, box)
                x1,y1 = max(x1,0), max(y1,0)
                x2,y2 = min(x2,w-1), min(y2,h-1)
                cx,cy = (x1+x2)//2, (y1+y2)//2
                nome  = model.names[cls].lower()

                if nome == CLASSE_BOLA:
                    area = (x2-x1)*(y2-y1)
                    if conf >= CONF_BOLA and area <= BOLA_MAX_AREA*h*w:
                        if pos_bola is None or conf > pos_bola[2]:
                            pos_bola = (cx, cy, float(conf))
                elif nome in CLASSES_JOGADOR:
                    clf.adicionar(frame, x1, y1, x2, y2)
                    if clf.classificar(frame, x1, y1, x2, y2) == 'A':
                        jog_adv.append((cx, cy))
                    else:
                        jog_nosso.append((cx, cy))

        if pos_bola: pos_bola = (pos_bola[0], pos_bola[1])
        analise.atualizar(h, w, jog_adv, jog_nosso, pos_bola)

        # Detecta formacao a cada 10 frames
        if frame_count % 10 == 0:
            if jog_adv:
                fa = analise.detectar_formacao([y/h for _,y in jog_adv])
                analise.hist_form_adv.append(fa)

                if fa != "?":
                    form_contagem[fa] = form_contagem.get(fa, 0) + 1

                    # Salva captura quando formacao for confirmada
                    if form_contagem[fa] >= CONF_FORM and fa not in capturas_salvas:
                        minuto  = int(frame_count / fps_video / 60)
                        segundo = int(frame_count / fps_video % 60)
                        nome    = pasta / f"formacao_{fa.replace('-','_')}_min{minuto:02d}{segundo:02d}.jpg"
                        salvar_captura(frame, jog_adv, fa, adversario,
                                       minuto, segundo, str(nome))
                        capturas_salvas.add(fa)

            if jog_nosso:
                fn = analise.detectar_formacao([y/h for _,y in jog_nosso])
                analise.hist_form_nosso.append(fn)

    cap.release()
    el = time.time() - t_inicio
    print(f"\n[OK] {frame_count} frames em {el/60:.1f} min | {len(capturas_salvas)} captura(s) salva(s)")

    # Relatorio
    relatorio = analise.gerar_relatorio()
    nome_json = f"scout_{Path(video_path).stem}.json"
    with open(nome_json, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=4, ensure_ascii=False)
    print(f"[OK] Relatorio: {nome_json}")

    init_db()
    salvar_no_banco(relatorio, adversario, video_path)

    adv = relatorio["adversario"]
    print(f"\n{'='*55}")
    print(f"  {adversario}  —  Formacao: {adv['formacao_principal']}")
    print(f"  Posse: {adv['posse_bola_pct']}%  |  Estilo: {adv['estilo_jogo']}")
    print(f"  Zona de pressao: {adv['zona_pressao']}")
    print(f"  Zona fraca:      {adv['zona_fraca']}")
    print(f"\n  Pontos Fracos:")
    for p in relatorio["pontos_fracos"]:    print(f"    - {p}")
    print(f"\n  Sugestoes:")
    for s in relatorio["sugestoes_taticas"]: print(f"    > {s}")
    print(f"\n  Painel: python painel_web.py -> http://localhost:5000")
    print(f"{'='*55}\n")
    return relatorio


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True)
    parser.add_argument("--adversario", default="Adversario")
    parser.add_argument("--cor_adv", default=None,
        help="Cor da camisa do adversario: branco, preto, azul, azul_claro, vermelho, amarelo, verde, laranja, roxo, cinza")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"[ERRO] Video nao encontrado: {args.video}"); exit(1)

    if args.cor_adv and args.cor_adv not in CORES_CAMISA:
        print(f"[ERRO] Cor invalida: {args.cor_adv}")
        print(f"  Opcoes: {', '.join(CORES_CAMISA.keys())}")
        exit(1)

    init_db()
    processar(args.video, args.adversario, cor_adv=args.cor_adv)