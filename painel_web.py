"""
PAINEL WEB — Scout Tático
==========================
Interface web para visualizar histórico de jogos analisados.

Uso:
  1. Rode o scout_tatico.py normalmente para analisar jogos
  2. Em outro terminal, rode: python painel_web.py
  3. Abra no navegador: http://localhost:5000

Instalar dependências:
  pip install flask
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)
DB_PATH = "scout_historico.db"


# ── BANCO DE DADOS ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS jogos (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            data          TEXT,
            adversario    TEXT,
            video         TEXT,
            formacao      TEXT,
            posse_adv     INTEGER,
            zona_pressao  TEXT,
            zona_fraca    TEXT,
            estilo        TEXT,
            compacidade   REAL,
            frames        INTEGER,
            relatorio     TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS jogadas (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            jogo_id    INTEGER,
            tipo       TEXT,
            minuto     INTEGER,
            descricao  TEXT,
            zona       TEXT,
            FOREIGN KEY (jogo_id) REFERENCES jogos(id)
        )
    """)

    conn.commit()
    conn.close()


def salvar_jogo(relatorio: dict, nome_adversario: str = "Adversário", video: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    adv  = relatorio.get("adversario", {})

    c.execute("""
        INSERT INTO jogos
        (data, adversario, video, formacao, posse_adv, zona_pressao,
         zona_fraca, estilo, compacidade, frames, relatorio)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        relatorio.get("data_analise", datetime.now().strftime("%d/%m/%Y %H:%M")),
        nome_adversario,
        video,
        adv.get("formacao_principal", "?"),
        adv.get("posse_bola_pct", 0),
        adv.get("zona_pressao", "?"),
        adv.get("zona_fraca", "?"),
        adv.get("estilo_jogo", "?"),
        adv.get("compacidade", 0),
        relatorio.get("frames_analisados", 0),
        json.dumps(relatorio, ensure_ascii=False),
    ))

    jogo_id = c.lastrowid
    conn.commit()
    conn.close()
    return jogo_id


def carregar_jogos():
    conn   = sqlite3.connect(DB_PATH)
    c      = conn.cursor()
    c.execute("SELECT * FROM jogos ORDER BY id DESC")
    cols   = [d[0] for d in c.description]
    jogos  = [dict(zip(cols, row)) for row in c.fetchall()]
    conn.close()
    return jogos


def carregar_jogo(jogo_id):
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    c.execute("SELECT * FROM jogos WHERE id = ?", (jogo_id,))
    cols  = [d[0] for d in c.description]
    row   = c.fetchone()
    conn.close()
    if row:
        jogo = dict(zip(cols, row))
        jogo["relatorio"] = json.loads(jogo["relatorio"])
        return jogo
    return None


def importar_jsons():
    """Importa automaticamente arquivos scout_relatorio*.json da pasta."""
    importados = 0
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    for f in Path(".").glob("scout_relatorio*.json"):
        # Verifica se já foi importado
        c.execute("SELECT id FROM jogos WHERE video = ?", (str(f),))
        if c.fetchone():
            continue
        try:
            with open(f, encoding="utf-8") as fp:
                rel = json.load(fp)
            conn.close()
            salvar_jogo(rel, video=str(f))
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            importados += 1
            print(f"[DB] Importado: {f}")
        except Exception as e:
            print(f"[AVISO] Erro ao importar {f}: {e}")

    conn.close()
    return importados


# ── HTML DO PAINEL ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scout Tático — Painel Web</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0D1B2A; color: #ECEFF1; font-family: 'Segoe UI', sans-serif; }

  header {
    background: #0A1520;
    padding: 16px 32px;
    border-bottom: 1px solid #1E3040;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  header h1 { color: #FF6B6B; font-size: 22px; }
  header span { color: #78909C; font-size: 14px; }

  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  .stat-card {
    background: #0A1520;
    border: 1px solid #1E3040;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
  }
  .stat-card .valor { font-size: 36px; font-weight: bold; color: #FF6B6B; }
  .stat-card .label { color: #78909C; font-size: 13px; margin-top: 4px; }

  h2 { color: #4FC3F7; font-size: 18px; margin-bottom: 16px; }

  .jogos-lista { display: flex; flex-direction: column; gap: 12px; margin-bottom: 32px; }

  .jogo-card {
    background: #0A1520;
    border: 1px solid #1E3040;
    border-radius: 12px;
    padding: 20px;
    cursor: pointer;
    transition: border-color 0.2s;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    align-items: center;
  }
  .jogo-card:hover { border-color: #FF6B6B; }
  .jogo-card .adv { font-size: 18px; font-weight: bold; color: #FF6B6B; }
  .jogo-card .info { color: #78909C; font-size: 13px; margin-top: 4px; }
  .jogo-card .badge {
    background: rgba(255,107,107,0.15);
    color: #FF6B6B;
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    text-align: center;
  }

  /* Modal */
  .modal-overlay {
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.8);
    z-index: 100;
    justify-content: center;
    align-items: flex-start;
    padding: 40px 20px;
    overflow-y: auto;
  }
  .modal-overlay.open { display: flex; }
  .modal {
    background: #0D1B2A;
    border: 1px solid #1E3040;
    border-radius: 16px;
    padding: 32px;
    max-width: 800px;
    width: 100%;
    position: relative;
  }
  .modal h2 { color: #FF6B6B; font-size: 22px; margin-bottom: 24px; }
  .modal .fechar {
    position: absolute; top: 16px; right: 16px;
    background: none; border: none; color: #EF5350;
    font-size: 22px; cursor: pointer;
  }

  .detalhe-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }
  .detalhe-item { background: #0A1520; border-radius: 10px; padding: 16px; }
  .detalhe-item .titulo { color: #78909C; font-size: 12px; margin-bottom: 6px; }
  .detalhe-item .valor  { color: #ECEFF1; font-size: 16px; font-weight: bold; }

  .lista-itens { list-style: none; }
  .lista-itens li {
    padding: 10px 0;
    border-bottom: 1px solid #1E3040;
    color: #ECEFF1;
    font-size: 14px;
  }
  .lista-itens li:last-child { border-bottom: none; }
  .lista-itens li::before { content: '• '; color: #FF6B6B; }

  .sug li::before { color: #66BB6A; }

  .pressao-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 12px;
  }
  .zona-cell {
    background: #1E3040;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    font-size: 12px;
  }
  .zona-cell .znome { color: #78909C; font-size: 11px; }
  .zona-cell .zval  { color: #FF6B6B; font-size: 18px; font-weight: bold; }

  .tag-form {
    display: inline-block;
    background: rgba(255,107,107,0.2);
    color: #FF6B6B;
    border: 1px solid rgba(255,107,107,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 14px;
    font-weight: bold;
    margin-right: 8px;
  }

  .vazio {
    text-align: center;
    color: #37474F;
    padding: 48px;
    font-size: 16px;
  }

  .btn-importar {
    background: rgba(79,195,247,0.15);
    color: #4FC3F7;
    border: 1px solid rgba(79,195,247,0.4);
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    cursor: pointer;
    margin-bottom: 24px;
  }
  .btn-importar:hover { background: rgba(79,195,247,0.3); }
</style>
</head>
<body>

<header>
  <div>
    <h1>⚽ Scout Tático</h1>
    <span>Painel de Análise do Adversário</span>
  </div>
</header>

<div class="container">

  <div class="stats-grid" id="statsGrid">
    <div class="stat-card">
      <div class="valor" id="totalJogos">—</div>
      <div class="label">Jogos Analisados</div>
    </div>
    <div class="stat-card">
      <div class="valor" id="formMaisUsada">—</div>
      <div class="label">Formação Mais Usada</div>
    </div>
    <div class="stat-card">
      <div class="valor" id="mediaPosse">—%</div>
      <div class="label">Posse Média do Adv.</div>
    </div>
    <div class="stat-card">
      <div class="valor" id="zonaFraca">—</div>
      <div class="label">Zona Mais Fraca</div>
    </div>
  </div>

  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
    <h2>📋 Histórico de Jogos</h2>
    <button class="btn-importar" onclick="importarJsons()">
      ⬆ Importar JSONs da Pasta
    </button>
  </div>

  <div class="jogos-lista" id="jogosLista">
    <div class="vazio">Nenhum jogo analisado ainda.<br>Rode o scout_tatico.py e clique em "Importar JSONs".</div>
  </div>

</div>

<!-- Modal de detalhes -->
<div class="modal-overlay" id="modalOverlay">
  <div class="modal">
    <button class="fechar" onclick="fecharModal()">✕</button>
    <h2 id="modalTitulo">Detalhes do Jogo</h2>

    <div class="detalhe-grid" id="modalGrid"></div>

    <h2 style="color:#FFA726; margin-bottom:12px;">⚠ Pontos Fracos</h2>
    <ul class="lista-itens" id="modalFracos"></ul>

    <h2 style="color:#66BB6A; margin-top:24px; margin-bottom:12px;">✅ Sugestões Táticas</h2>
    <ul class="lista-itens sug" id="modalSugestoes"></ul>

    <h2 style="color:#FF6B6B; margin-top:24px; margin-bottom:12px;">📍 Pressão por Zona</h2>
    <div class="pressao-grid" id="modalPressao"></div>
  </div>
</div>

<script>
async function carregarDados() {
  const res   = await fetch('/api/jogos');
  const dados = await res.json();

  // Stats
  document.getElementById('totalJogos').textContent = dados.length;
  if (dados.length > 0) {
    const mediaPosse = Math.round(dados.reduce((s,j) => s + j.posse_adv, 0) / dados.length);
    document.getElementById('mediaPosse').textContent = mediaPosse + '%';

    const formas = dados.map(j => j.formacao).filter(f => f !== '?');
    if (formas.length > 0) {
      const contagem = {};
      formas.forEach(f => contagem[f] = (contagem[f] || 0) + 1);
      const maisUsada = Object.entries(contagem).sort((a,b) => b[1]-a[1])[0][0];
      document.getElementById('formMaisUsada').textContent = maisUsada;
    }

    const zonas = dados.map(j => j.zona_fraca).filter(z => z && z !== '?');
    if (zonas.length > 0) {
      const cont = {};
      zonas.forEach(z => cont[z] = (cont[z] || 0) + 1);
      const maisFreq = Object.entries(cont).sort((a,b) => b[1]-a[1])[0][0];
      // Abreviação para o card
      const partes = maisFreq.split(' ');
      document.getElementById('zonaFraca').textContent =
        partes.length > 1 ? partes[0][0] + '.' + partes.slice(1).join(' ') : maisFreq;
    }
  }

  // Lista de jogos
  const lista = document.getElementById('jogosLista');
  if (dados.length === 0) {
    lista.innerHTML = '<div class="vazio">Nenhum jogo analisado ainda.<br>Rode o scout_tatico.py e clique em "Importar JSONs".</div>';
    return;
  }

  lista.innerHTML = dados.map(j => `
    <div class="jogo-card" onclick="abrirJogo(${j.id})">
      <div>
        <div class="adv">${j.adversario}</div>
        <div class="info">
          📅 ${j.data} &nbsp;|&nbsp;
          🎯 ${j.formacao} &nbsp;|&nbsp;
          📍 Zona Fraca: ${j.zona_fraca} &nbsp;|&nbsp;
          ⚡ ${j.estilo}
        </div>
      </div>
      <div>
        <div class="badge">Posse ${j.posse_adv}%</div>
      </div>
    </div>
  `).join('');
}

async function abrirJogo(id) {
  const res  = await fetch('/api/jogo/' + id);
  const jogo = await res.json();
  const rel  = jogo.relatorio;
  const adv  = rel.adversario || {};

  document.getElementById('modalTitulo').textContent =
    '⚽ ' + jogo.adversario + ' — ' + jogo.data;

  // Grid de detalhes
  document.getElementById('modalGrid').innerHTML = `
    <div class="detalhe-item">
      <div class="titulo">Formação Principal</div>
      <div class="valor"><span class="tag-form">${adv.formacao_principal || '?'}</span></div>
    </div>
    <div class="detalhe-item">
      <div class="titulo">Posse de Bola</div>
      <div class="valor">${adv.posse_bola_pct || 0}%</div>
    </div>
    <div class="detalhe-item">
      <div class="titulo">Zona de Maior Pressão</div>
      <div class="valor">${adv.zona_pressao || '—'}</div>
    </div>
    <div class="detalhe-item">
      <div class="titulo">Zona Mais Fraca</div>
      <div class="valor">${adv.zona_fraca || '—'}</div>
    </div>
    <div class="detalhe-item">
      <div class="titulo">Estilo de Jogo</div>
      <div class="valor">${adv.estilo_jogo || '—'}</div>
    </div>
    <div class="detalhe-item">
      <div class="titulo">Frames Analisados</div>
      <div class="valor">${rel.frames_analisados || 0}</div>
    </div>
  `;

  // Pontos fracos
  const fracos = rel.pontos_fracos || [];
  document.getElementById('modalFracos').innerHTML =
    fracos.length > 0
      ? fracos.map(f => `<li>${f}</li>`).join('')
      : '<li>Nenhum ponto fraco identificado</li>';

  // Sugestões
  const sugs = rel.sugestoes_taticas || [];
  document.getElementById('modalSugestoes').innerHTML =
    sugs.length > 0
      ? sugs.map(s => `<li>${s}</li>`).join('')
      : '<li>Sem sugestões</li>';

  // Pressão por zona
  const pressao = rel.pressao_por_zona || {};
  const ordem = [
    'Ataque Esquerdo','Ataque Central','Ataque Direito',
    'Meio Esquerdo','Centro','Meio Direito',
    'Defesa Esquerda','Defesa Central','Defesa Direita',
  ];
  const maxVal = Math.max(...Object.values(pressao), 1);
  document.getElementById('modalPressao').innerHTML = ordem.map(zona => {
    const val   = pressao[zona] || 0;
    const intens = Math.round((val / maxVal) * 255);
    const cor   = `rgb(${intens}, ${Math.round(intens*0.2)}, ${Math.round(intens*0.2)})`;
    return `
      <div class="zona-cell" style="background:${cor}20; border:1px solid ${cor}40;">
        <div class="znome">${zona}</div>
        <div class="zval">${val}</div>
      </div>`;
  }).join('');

  document.getElementById('modalOverlay').classList.add('open');
}

function fecharModal() {
  document.getElementById('modalOverlay').classList.remove('open');
}

async function importarJsons() {
  const res  = await fetch('/api/importar', { method: 'POST' });
  const data = await res.json();
  alert(data.mensagem);
  carregarDados();
}

// Fecha modal clicando fora
document.getElementById('modalOverlay').addEventListener('click', function(e) {
  if (e.target === this) fecharModal();
});

carregarDados();
setInterval(carregarDados, 10000); // atualiza a cada 10 segundos
</script>
</body>
</html>
"""


# ── ROTAS ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/jogos")
def api_jogos():
    return jsonify(carregar_jogos())


@app.route("/api/jogo/<int:jogo_id>")
def api_jogo(jogo_id):
    jogo = carregar_jogo(jogo_id)
    if jogo:
        return jsonify(jogo)
    return jsonify({"erro": "Jogo não encontrado"}), 404


@app.route("/api/importar", methods=["POST"])
def api_importar():
    n = importar_jsons()
    return jsonify({"mensagem": f"{n} jogo(s) importado(s) com sucesso!"})


@app.route("/api/salvar", methods=["POST"])
def api_salvar():
    """Recebe relatório do scout_tatico.py e salva no banco."""
    dados = request.json
    rel   = dados.get("relatorio", {})
    nome  = dados.get("adversario", "Adversário")
    video = dados.get("video", "")
    jogo_id = salvar_jogo(rel, nome, video)
    return jsonify({"id": jogo_id, "mensagem": "Jogo salvo com sucesso!"})


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    n = importar_jsons()
    if n > 0:
        print(f"[DB] {n} jogo(s) importado(s) automaticamente!")

    print("\n" + "="*50)
    print("  PAINEL WEB — Scout Tático")
    print("="*50)
    print("\n  Acesse no navegador:")
    print("  http://localhost:5000")
    print("\n  Pressione Ctrl+C para parar\n")

    app.run(debug=False, port=5000, host="0.0.0.0")