# 🧠 Thinking Models Analyzer

Aplicação **Streamlit** para explorar, comparar e avaliar diferentes **abordagens de prompts** (Chain of Thought, Skeleton of Thought, Tree of Thought, Self-consistency, Zero-shot e Few-shot).  
Inclui um módulo de **Experimentos & Benchmark** que permite testar casos reais, salvar tentativas em JSONL local e gerar ranking das abordagens.

---

## 🚀 Funcionalidades

- **Visão geral das abordagens**: explicação, quando usar, exemplos e template de prompt.
- **Recomendação com Azure OpenAI (opcional)**: IA sugere a melhor abordagem para um contexto específico.
- **Experimentos & Benchmark**:
  - CRUD de casos (com pesos de métricas customizáveis).
  - Execução de testes com várias abordagens e repetições.
  - Execução via **Azure OpenAI** ou **heurística local simulada**.
  - Registro de resultados em arquivos **JSONL locais** (`data/cases.jsonl`, `runs.jsonl`, `attempts.jsonl`).
  - Avaliação manual (nota 1–5, observações).
  - Ranking automático com pesos normalizados (Qualidade, Tempo, Tokens).
  - Exportação em **CSV** ou **JSON**.

---

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/thinking-models-analyzer.git
cd thinking-models-analyzer
```

### 2. Crie e ative um ambiente virtual
Linux/macOS:
```bash
ppython -m venv .thinkvenv
source .thinkvenv/bin/activate # On Windows: .thinkvenv\Scripts\activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

---

## 📑 Arquivo `requirements.txt`

```txt
streamlit>=1.34.0
openai>=1.30.0
jsonlines>=4.0.0
python-dotenv>=1.0.1
plotly>=5.22.0
pandas>=2.2.0
```

---

## ⚙️ Configuração Azure OpenAI (opcional)

Crie um arquivo `.env` na raiz do projeto com:

```ini
AZURE_OPENAI_ENDPOINT=https://<seu-recurso>.openai.azure.com/
AZURE_OPENAI_API_KEY=<sua_chave_api>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

> Se não configurar, a aplicação funcionará no **modo heurístico local**.

---

## ▶️ Executando a aplicação

```bash
streamlit run streamlit_app_full.py
```

Acesse em: [http://localhost:8501](http://localhost:8501)

---

## 📂 Estrutura do projeto

```
thinking-models-analyzer/
│
├── streamlit_app_full.py     # Código principal da aplicação
├── requirements.txt          # Dependências do projeto
├── README.md                 # Este arquivo
├── .env.example              # Exemplo de configuração do Azure OpenAI
└── data/                     # Diretório com JSONL persistidos (criado em runtime)
    ├── cases.jsonl
    ├── runs.jsonl
    └── attempts.jsonl
```

---

## 🛠️ Roadmap futuro

- Adicionar suporte a **exemplos Few-shot customizáveis** por caso.
- Persistir métricas adicionais (custo estimado por execução).
- Dashboard de comparações entre múltiplos casos.

---

## 📜 Licença

MIT — fique à vontade para usar, modificar e compartilhar.

---
