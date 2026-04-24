"""
TREINO MELHORADO — YOLOv8m com múltiplos datasets
===================================================
Resolve problemas de:
- Bola sendo confundida com jogador
- Times trocados na classificação

Uso:
  python treinar_futebol2.py
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATASETS = [
    "dataset_futebol/football-player.v5i.yolov8",
    "dataset_futebol/Football Players Detection.v1i.yolov8",
    "dataset_futebol/Football Players Detection.v1i.yolov8 (1)",
    "dataset_futebol/football-players-detection.v20-rf-detr-m...",  # ajuste o nome se necessário
]

MODELO_BASE = "yolov8s.pt"   # small — 3x mais rapido, qualidade proxima do medium
EPOCAS      = 50
IMGSZ       = 640
BATCH       = 16             # maior batch = mais rapido
PROJETO     = "runs/detect"
NOME_TREINO = "futebol_v2"


# ── MERGE DOS DATASETS ─────────────────────────────────────────────────────────
def merge_datasets(datasets, dest="dataset_merged"):
    dest = Path(dest)
    for split in ("train", "valid", "test"):
        (dest / split / "images").mkdir(parents=True, exist_ok=True)
        (dest / split / "labels").mkdir(parents=True, exist_ok=True)

    # Lê classes do primeiro dataset como referência
    classes_ref = None
    for ds in datasets:
        yaml_path = Path(ds) / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if classes_ref is None:
                classes_ref = data.get("names", [])
            break

    if classes_ref is None:
        classes_ref = ["ball", "goalkeeper", "player", "referee"]

    print(f"\n[Merge] Classes de referência: {classes_ref}")

    total_imgs = {"train": 0, "valid": 0, "test": 0}

    for ds_idx, ds_path in enumerate(datasets):
        ds_path = Path(ds_path)
        if not ds_path.exists():
            print(f"[AVISO] Dataset não encontrado: {ds_path} — pulando")
            continue

        yaml_path = ds_path / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            classes_ds = data.get("names", classes_ref)
        else:
            classes_ds = classes_ref

        # Mapeia classes do dataset para as classes de referência
        mapa = {}
        for i, nome in enumerate(classes_ds):
            nome_lower = nome.lower()
            for j, ref in enumerate(classes_ref):
                if nome_lower == ref.lower() or nome_lower in ref.lower():
                    mapa[i] = j
                    break

        print(f"[Merge] Dataset {ds_idx+1}: {ds_path.name} | Classes: {classes_ds}")

        for split in ("train", "valid", "test"):
            img_dir = ds_path / split / "images"
            lbl_dir = ds_path / split / "labels"

            if not img_dir.exists():
                continue

            imgs = list(img_dir.glob("*.jpg")) + \
                   list(img_dir.glob("*.jpeg")) + \
                   list(img_dir.glob("*.png"))

            for img_path in imgs:
                novo_nome = f"ds{ds_idx}_{img_path.name}"
                dest_img  = dest / split / "images" / novo_nome
                shutil.copy2(img_path, dest_img)

                lbl_path = lbl_dir / (img_path.stem + ".txt")
                dest_lbl = dest / split / "labels" / (Path(novo_nome).stem + ".txt")

                if lbl_path.exists():
                    with open(lbl_path) as f:
                        linhas = f.readlines()
                    novas = []
                    for linha in linhas:
                        partes = linha.strip().split()
                        if partes:
                            cls_orig = int(partes[0])
                            cls_novo = mapa.get(cls_orig, cls_orig)
                            novas.append(f"{cls_novo} " + " ".join(partes[1:]))
                    with open(dest_lbl, "w") as f:
                        f.write("\n".join(novas))
                else:
                    open(dest_lbl, "w").close()

                total_imgs[split] += 1

    print(f"\n[Merge] Total de imagens:")
    for split, n in total_imgs.items():
        print(f"  {split}: {n} imagens")

    data_yaml = {
        "path":  str(dest.absolute()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(classes_ref),
        "names": classes_ref,
    }
    with open(dest / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)

    print(f"\n[Merge] Dataset combinado salvo em: {dest}/")
    return str(dest / "data.yaml")


# ── TREINO ─────────────────────────────────────────────────────────────────────
def treinar(data_yaml):
    print(f"\n[Treino] Iniciando com modelo: {MODELO_BASE}")
    print(f"[Treino] Epocas: {EPOCAS} | IMGSZ: {IMGSZ} | Batch: {BATCH}")
    print(f"[Treino] Isso vai levar 60-90 minutos...\n")

    model = YOLO(MODELO_BASE)

    results = model.train(
        data        = data_yaml,
        epochs      = EPOCAS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        project     = PROJETO,
        name        = NOME_TREINO,
        patience    = 20,
        save_period = 10,
        lr0         = 0.01,
        lrf         = 0.001,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 10,
        scale       = 0.5,
        flipud      = 0.0,
        fliplr      = 0.5,
        mosaic      = 1.0,
        verbose     = True,
    )

    melhor = f"{PROJETO}/{NOME_TREINO}/weights/best.pt"
    print(f"\n Treino concluido!")
    print(f"   Modelo salvo em: {melhor}")
    print(f"\n   Atualize o scout_tatico.py:")
    print(f'   MODEL_PATH = "{melhor}"')

    return melhor


# ── VALIDAÇÃO ──────────────────────────────────────────────────────────────────
def validar(modelo_path, data_yaml):
    print(f"\n[Validacao] Testando modelo: {modelo_path}")
    model   = YOLO(modelo_path)
    metrics = model.val(data=data_yaml, imgsz=IMGSZ, verbose=True)

    print(f"\n Metricas finais:")
    print(f"   mAP@50:    {metrics.box.map50:.3f}  (ideal > 0.85)")
    print(f"   mAP@50-95: {metrics.box.map:.3f}   (ideal > 0.60)")
    print(f"   Precisao:  {metrics.box.mp:.3f}")
    print(f"   Recall:    {metrics.box.mr:.3f}")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  TREINO MELHORADO — YOLOv8m + Multiplos Datasets")
    print("=" * 55)

    # Lista os datasets disponíveis na pasta para ajudar
    print("\n[Info] Datasets encontrados em dataset_futebol/:")
    pasta = Path("dataset_futebol")
    if pasta.exists():
        for p in sorted(pasta.iterdir()):
            if p.is_dir():
                imgs = list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
                print(f"  {p.name} — {len(imgs)} imagens")
    else:
        print("  [AVISO] Pasta dataset_futebol nao encontrada!")

    # Verifica quais datasets existem
    encontrados = [d for d in DATASETS if Path(d).exists()]
    if not encontrados:
        print("\n[ERRO] Nenhum dataset encontrado!")
        print("Os caminhos configurados sao:")
        for d in DATASETS:
            existe = "OK" if Path(d).exists() else "NAO ENCONTRADO"
            print(f"  [{existe}] {d}")
        exit(1)

    print(f"\n[Info] {len(encontrados)} dataset(s) valido(s) para treino:")
    for d in encontrados:
        imgs = list(Path(d).rglob("*.jpg")) + list(Path(d).rglob("*.png"))
        print(f"  {d} — {len(imgs)} imagens")

    # Merge + Treino + Validação
    data_yaml = merge_datasets(encontrados)
    modelo    = treinar(data_yaml)
    validar(modelo, data_yaml)

    print("\n Tudo pronto! Agora atualize o MODEL_PATH no scout_tatico.py e rode:")
    print("   python scout_tatico.py --video jogo_teste.mp4")