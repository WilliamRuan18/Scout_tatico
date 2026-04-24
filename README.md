# ⚽ Scout Tático — Análise de Futebol por IA

> Sistema de análise tática de futebol por visão computacional que detecta jogadores, identifica formações e gera relatórios automáticos para treinadores estudarem o time adversário.

---

## 🎯 Problema que resolve

Ferramentas profissionais de análise tática como **Wyscout** e **InStat** custam até **R$150.000 por temporada**, tornando-as inacessíveis para a maioria dos clubes brasileiros. O Scout Tático oferece funcionalidades similares de forma **gratuita**, rodando em qualquer computador convencional.

---

## 🚀 Como usar

### 1. Instale as dependências
```bash
pip install ultralytics opencv-python numpy flask pyqt5
```

### 2. Baixe o vídeo do adversário
Acesse [cobalt.tools](https://cobalt.tools), cole o link do YouTube e baixe em **1080p**.

### 3. Processe o vídeo
```bash
python processar_video.py --video jogo.mp4 --adversario "Nome do Time"
```

### 4. Abra o painel web
```bash
python painel_web.py
```
Acesse **http://localhost:5000** no navegador ou celular.

### 5. Estude o adversário e prepare o time! ✅

---

## 📊 O que o sistema entrega

| Funcionalidade | Descrição |
|---|---|
| 🔍 Detecção em tempo real | Detecta jogadores, goleiros, árbitros e bola |
| 👕 Separação de times | Classifica por cor de camisa com calibração por clique |
| 📐 Formação tática | Detecta automaticamente 4-3-3, 4-4-2, 3-5-2... |
| 📍 Mapa de pressão | Divide o campo em 9 zonas e mapeia onde o adversário atua |
| ⚽ Posse de bola | Calcula posse em tempo real |
| 📸 Evidência visual | Salva foto com a formação confirmada e linhas conectando jogadores |
| 📋 Relatório completo | Pontos fracos e sugestões táticas geradas automaticamente |
| 🌐 Painel web | Histórico de jogos acessível pelo celular |

---

## ⚠️ Pontos Fracos e Sugestões

Exemplo de relatório gerado automaticamente:

```
Formação detectada: 5-4-1
Posse de bola: 75%
Estilo: Ofensivo (linha alta)

Pontos Fracos:
  ⚠ Time reativo — vulnerável quando pressão alta falha
  ⚠ Ataque Esquerdo descoberto — menor cobertura do campo
  ⚠ Muita posse — pressão alta pode forçar erros

Sugestões Táticas:
  ✅ Use 4-3-3 para amplitude e forçar erros
  ✅ Direcione jogadas para o Ataque Esquerdo
  ✅ Bolas longas nas costas da defesa
  ✅ Pressione alto para forçar erros do goleiro
  ✅ Treine bolas paradas — podem ser decisivas
```

---

## 🗂️ Estrutura do Projeto

```
A3 (PETROS)/
│
├── processar_video.py    # Processa vídeo em background com máxima qualidade
├── painel_web.py         # Painel web com histórico de jogos
├── cortar_video.py       # Recorta partes do vídeo
├── treinar_futebol2.py   # Retreina o modelo com novos dados
│
├── dataset_futebol/      # Datasets de treino
├── runs/                 # Modelos treinados
└── capturas/             # Fotos das formações confirmadas
```

---

## 🛠️ Tecnologias utilizadas

| Tecnologia | Função |
|---|---|
| **YOLOv8** | Detecção de objetos em tempo real |
| **OpenCV** | Processamento de vídeo e visão computacional |
| **PyQt5** | Interface gráfica |
| **Flask** | Painel web |
| **SQLite** | Banco de dados do histórico |
| **NumPy** | Cálculos matemáticos |
| **K-Means** | Classificação de times por cor de camisa |

---

## 📈 Resultados do modelo treinado

| Métrica | Valor |
|---|---|
| Precisão de jogadores | **98.9%** |
| Precisão de goleiros | **98.1%** |
| Precisão de árbitros | **96.1%** |
| mAP@50 geral | **84.5%** |
| Dataset de treino | **2.899 imagens** |

---

## 📱 Acessar painel pelo celular

1. Descubra o IP do computador: `ipconfig` (Windows)
2. Conecte o celular no mesmo Wi-Fi
3. Acesse: `http://SEU_IP:5000`

---

## 🔮 Trabalhos futuros

- [ ] Histórico de jogadas de gol
- [ ] Análise de transmissão ao vivo
- [ ] Exportar relatório em PDF

---

## 👨‍💻 Desenvolvido por

**William Ruan** — Projeto A3 — Análise Tática de Futebol por IA

> *"Times profissionais pagam R$150.000 por temporada para ter isso. O meu roda de graça em qualquer computador com 95% de precisão."*
