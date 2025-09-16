
# streamlit_app_full.py
# -------------------------------------------------------------
# Prompting Approaches Explorer + Experiments/Benchmark (JSONL)
# Layout vibe: Copilot Studio (dark gradient, cards, expanders)
# -------------------------------------------------------------
# How to run:
#   pip install streamlit openai jsonlines python-dotenv plotly
#   streamlit run streamlit_app_full.py
#
# Azure (optional):
#   AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
#   AZURE_OPENAI_API_KEY="<your_key>"
#   AZURE_OPENAI_DEPLOYMENT_NAME="<deployment>"   # e.g. gpt-4o-mini
#   AZURE_OPENAI_API_VERSION="2024-02-15-preview"
#
import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import jsonlines
from dotenv import load_dotenv

# Optional deps (only needed if using Azure)
try:
    import openai  # openai>=1.*
    has_openai = True
except Exception:
    has_openai = False

# Optional charts
try:
    import plotly.express as px
except Exception:
    px = None

load_dotenv()

# ----------------------------------------
# Page config + global CSS (Copilot vibe)
# ----------------------------------------
st.set_page_config(
    page_title="Prompting Approaches + Benchmark",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #ffffff;
    }
    .main-header {
        background: linear-gradient(90deg, #6264A7 0%, #8B5CF6 100%);
        padding: 18px 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(98, 100, 167, 0.3);
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0 16px;
        backdrop-filter: blur(8px);
    }
    .example-box {
        background: rgba(139, 92, 246, 0.08);
        border-left: 4px solid #8B5CF6;
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 13px;
        white-space: pre-wrap;
    }
    .analysis-result {
        background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(59,130,246,0.12) 100%);
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }
    .stTextArea textarea {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6264A7 0%, #8B5CF6 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        transition: all .2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(139,92,246,.25);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Approaches data (six types)
# -----------------------------
THINKING_MODELS: Dict[str, Dict[str, str]] = {
    "Chain of Thought (CoT)": {
        "descricao": "Quebra o problema em etapas sequenciais e lógicas (passo-a-passo).",
        "quando_usar": "Cálculos, lógica, explicações lineares; troubleshooting com etapas claras.",
        "prompt_template": "Resolva passo a passo:\n\n{context}\n\nExplique cada etapa e finalize com **Resposta final:**",
        "exemplo": "Pergunta: A equipe tem 12 devs e contratou +5. Quantos agora?\nRaciocínio: 12 + 5 = 17.\nResposta final: 17."
    },
    "Skeleton of Thought (SoT)": {
        "descricao": "Cria primeiro um esqueleto (tópicos), depois expande cada um.",
        "quando_usar": "Relatórios, planejamentos, textos longos que exigem estrutura.",
        "prompt_template": "Fase A (Esqueleto 3–6 tópicos) para:\n\n{context}\n\nFase B: expanda 2–4 frases por tópico. Resuma em 2 linhas ao final.",
        "exemplo": "Tema: Estratégia de IA no varejo\n1) Objetivos\n2) Dados/Integrações\n3) Casos prioritários\n4) Riscos/Governança\n(Depois expandir cada item)."
    },
    "Tree of Thought (ToT)": {
        "descricao": "Explora múltiplos caminhos/ramificações, compara e seleciona o melhor.",
        "quando_usar": "Tomada de decisão, brainstorm, priorização com trade-offs.",
        "prompt_template": "Modele como árvore de opções para:\n\n{context}\n\nListe 2–4 caminhos (prós/contras), defina critérios (custo/risco/prazo/impacto), pontue 1–5 e escolha o vencedor com justificativa.",
        "exemplo": "Decisão: Expandir operação\nA) Crescimento orgânico (±)\nB) Investimento externo (±)\nC) Parcerias (±)\n→ Escolha justificada."
    },
    "Self-consistency": {
        "descricao": "Gera N cadeias de raciocínio e elege a resposta mais consistente.",
        "quando_usar": "Maior confiabilidade; cálculos/decisões críticas.",
        "prompt_template": "Gere N=3 respostas independentes para:\n\n{context}\n\nExtraia as respostas finais e selecione a mais frequente. Mostre um quadro com as 3 respostas e a escolhida.",
        "exemplo": "Pergunta: √144?\nAmostras: [12, 12, 14] → Resposta: 12."
    },
    "Zero-shot": {
        "descricao": "Resposta direta, sem exemplos prévios (rápido e simples).",
        "quando_usar": "Perguntas claras/simples ou prototipagem muito rápida.",
        "prompt_template": "Responda objetivamente (2–5 frases):\n\n{context}\n\nSe houver ambiguidade, declare suposições em 1 linha.",
        "exemplo": "Pergunta: O que é RAG?\nResposta: (definição direta)."
    },
    "Few-shot": {
        "descricao": "Ensina pelo exemplo (2–5 exemplos), depois pede a saída-alvo.",
        "quando_usar": "Padronizar formato/estilo; classificações e extrações.",
        "prompt_template": "Siga o padrão dos exemplos:\n[Ex 1: entrada → saída curta]\n[Ex 2: entrada → saída curta]\nAgora, gere a saída para:\n{context}",
        "exemplo": "Ex1: Resumo 1 (padrão X)\nEx2: Resumo 2 (padrão X)\nAgora, faça no mesmo padrão."
    },
}

# -------------------------------------------------
# JSONL persistence helpers (./data/*.jsonl files)
# -------------------------------------------------
def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def _path(filename: str) -> str:
    ensure_data_dir()
    return os.path.join("data", filename)

def load_jsonl(filename: str) -> List[Dict[str, Any]]:
    fp = _path(filename)
    if not os.path.exists(fp):
        return []
    try:
        with jsonlines.open(fp, "r") as reader:
            return list(reader)
    except Exception as e:
        st.error(f"Erro ao carregar {filename}: {e}")
        return []

def save_to_jsonl(filename: str, data: List[Dict[str, Any]]) -> None:
    fp = _path(filename)
    try:
        with jsonlines.open(fp, "w") as writer:
            writer.write_all(data)
    except Exception as e:
        st.error(f"Erro ao salvar {filename}: {e}")

def add_to_jsonl(filename: str, item: Dict[str, Any]) -> None:
    data = load_jsonl(filename)
    data.append(item)
    save_to_jsonl(filename, data)

def update_jsonl_item(filename: str, item_id: str, new_item: Dict[str, Any]) -> None:
    data = load_jsonl(filename)
    for i, x in enumerate(data):
        if x.get("id") == item_id:
            data[i] = new_item
            break
    save_to_jsonl(filename, data)

def delete_from_jsonl(filename: str, item_id: str) -> None:
    data = load_jsonl(filename)
    data = [x for x in data if x.get("id") != item_id]
    save_to_jsonl(filename, data)

# ---------------------------------------
# Azure OpenAI helpers (optional use)
# ---------------------------------------
def setup_azure_openai() -> Tuple[str, str, str, str]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    with st.sidebar:
        st.subheader("🔧 Azure OpenAI")
        endpoint = st.text_input("Endpoint", value=endpoint, placeholder="https://<resource>.openai.azure.com/")
        api_key = st.text_input("API Key", value=api_key, type="password", placeholder="********")
        deployment = st.text_input("Deployment", value=deployment, placeholder="gpt-4o-mini")
        api_version = st.text_input("API Version", value=api_version)

        st.caption("Preencha acima ou use variáveis de ambiente.")

    return endpoint.strip(), api_key.strip(), deployment.strip(), api_version.strip()

def execute_with_azure(prompt: str, endpoint: str, api_key: str, deployment: str,
                      api_version: str, temperature: float = 0.2) -> Dict[str, Any]:
    start = time.time()
    if not (has_openai and endpoint and api_key and deployment and api_version):
        return {"response_text": "Azure OpenAI não configurado.", "latency_ms": 0, "input_tokens": None, "output_tokens": None, "error": "missing_config"}
    try:
        client = openai.AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1200,
        )
        latency_ms = int((time.time() - start) * 1000)
        usage = getattr(resp, "usage", None)
        return {
            "response_text": resp.choices[0].message.content if resp and resp.choices else "",
            "latency_ms": latency_ms,
            "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "error": None,
        }
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        return {"response_text": f"Erro: {e}", "latency_ms": latency_ms, "input_tokens": None, "output_tokens": None, "error": str(e)}

def execute_with_heuristic(approach: str, context: str) -> Dict[str, Any]:
    import random
    # Simulate processing delay
    time.sleep(random.uniform(0.4, 1.2))
    snippets = {
        "Chain of Thought (CoT)": f"[CoT] {context[:120]}...\n1) Entenda o problema\n2) Analise causas\n3) Proponha ações\n4) Conclua com 'Resposta final:'",
        "Skeleton of Thought (SoT)": f"[SoT] {context[:120]}...\nI) Situação\nII) Desafios\nIII) Oportunidades\nIV) Recomendações",
        "Tree of Thought (ToT)": f"[ToT] {context[:120]}...\nA) Caminho 1 (±)\nB) Caminho 2 (±)\nC) Caminho 3 (±)\n→ Escolha com critérios.",
        "Self-consistency": f"[Self-consistency] {context[:120]}...\nGere 3 raciocínios e selecione o mais frequente.",
        "Zero-shot": f"[Zero-shot] {context[:120]}...\nResposta direta objetiva, com 2–5 frases.",
        "Few-shot": f"[Few-shot] {context[:120]}...\nUse 2–3 exemplos e replique o padrão na saída."
    }
    return {
        "response_text": snippets.get(approach, f"[{approach}] {context[:140]}..."),
        "latency_ms": random.randint(600, 1800),
        "input_tokens": None,
        "output_tokens": None,
        "error": None,
    }

# ---------------------------------------
# Ranking helpers
# ---------------------------------------
def calculate_ranking(attempts: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    if not attempts:
        return []

    # group by approach
    by_approach: Dict[str, List[Dict[str, Any]]] = {}
    for a in attempts:
        by_approach.setdefault(a["approach"], []).append(a)

    # max for normalization
    all_lat = [a["latency_ms"] for a in attempts if a.get("latency_ms") is not None]
    max_latency = max(all_lat) if all_lat else 1

    all_tokens = []
    for a in attempts:
        it = a.get("input_tokens") or 0
        ot = a.get("output_tokens") or 0
        if it or ot:
            all_tokens.append(it + ot)
    max_tokens = max(all_tokens) if all_tokens else 1

    results = []
    for approach, arr in by_approach.items():
        valids = [x for x in arr if (x.get("quality_score") or 0) > 0]
        if not valids:
            continue
        q_avg = sum([x.get("quality_score", 0) for x in valids]) / len(valids)
        lats = [x.get("latency_ms", 0) for x in valids if x.get("latency_ms") is not None]
        lat_avg = sum(lats) / len(lats) if lats else 0
        toks = []
        for x in valids:
            it = x.get("input_tokens") or 0
            ot = x.get("output_tokens") or 0
            if it or ot:
                toks.append(it + ot)
        tok_avg = sum(toks) / len(toks) if toks else 0

        quality_norm = (q_avg - 1) / 4 if q_avg > 0 else 0
        time_norm = 1 - (lat_avg / max_latency) if max_latency else 0
        tokens_norm = 1 - (tok_avg / max_tokens) if max_tokens and tok_avg else 0

        score = (
            weights.get("quality", 0.7) * quality_norm
            + weights.get("time", 0.2) * time_norm
            + weights.get("tokens", 0.1) * tokens_norm
        )
        results.append({
            "approach": approach,
            "final_score": score,
            "quality_avg": q_avg,
            "latency_avg": lat_avg,
            "tokens_avg": tok_avg,
            "attempts_count": len(valids),
        })

    results.sort(key=lambda x: (-x["final_score"], -x["quality_avg"], x["latency_avg"]))
    return results

# ---------------------------------------
# Experiments UI (three sub-tabs)
# ---------------------------------------
def render_cases_management():
    st.markdown("### 📋 Gerenciar Casos de Teste")
    cases = load_jsonl("cases.jsonl")

    with st.expander("➕ Criar novo caso", expanded=(len(cases) == 0)):
        with st.form("case_form"):
            c1, c2 = st.columns(2)
            with c1:
                title = st.text_input("Título*", placeholder="Ex.: Otimização de E-commerce")
                industry = st.text_input("Indústria", placeholder="Varejo, Saúde, Educação...")
                domain = st.text_input("Domínio", placeholder="Marketing, Operações...")
            with c2:
                goal = st.text_area("Objetivo", height=80, placeholder="O que você quer alcançar?")
                success_metric = st.text_input("Métrica de Sucesso", placeholder="Ex.: +20% conversão")

            context = st.text_area("Contexto Detalhado*", height=140,
                                   placeholder="Descreva a situação (país, mercado, restrições, riscos etc.)")

            st.markdown("**Pesos para ranking (devem somar 1.0 — normalizamos para você):**")
            w1, w2, w3 = st.columns(3)
            with w1:
                w_quality = st.slider("Qualidade", 0.0, 1.0, 0.7, 0.1)
            with w2:
                w_time = st.slider("Tempo", 0.0, 1.0, 0.2, 0.1)
            with w3:
                w_tokens = st.slider("Tokens", 0.0, 1.0, 0.1, 0.1)

            total = max(w_quality + w_time + w_tokens, 1e-9)
            w_quality, w_time, w_tokens = w_quality/total, w_time/total, w_tokens/total
            st.info(f"Pesos normalizados → Qualidade={w_quality:.2f} • Tempo={w_time:.2f} • Tokens={w_tokens:.2f}")

            submit = st.form_submit_button("💾 Salvar Caso")
            if submit:
                if not title or not context:
                    st.error("Preencha pelo menos Título e Contexto.")
                else:
                    item = {
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "industry": industry,
                        "domain": domain,
                        "goal": goal,
                        "context": context,
                        "success_metric": success_metric,
                        "weights": {"quality": w_quality, "time": w_time, "tokens": w_tokens},
                        "created_at": datetime.utcnow().isoformat()
                    }
                    add_to_jsonl("cases.jsonl", item)
                    st.success("Caso criado!")
                    st.rerun()

    if cases:
        st.markdown("### 📚 Casos existentes")
        for c in cases:
            with st.expander(f"📄 {c['title']} — {c.get('industry','N/A')}"):
                left, right = st.columns([4, 1])
                with left:
                    st.write(f"**Domínio:** {c.get('domain','—')}")
                    st.write(f"**Objetivo:** {c.get('goal','—')}")
                    st.write(f"**Métrica:** {c.get('success_metric','—')}")
                    with st.expander("📎 Contexto completo"):
                        st.write(c.get("context",""))
                    w = c.get("weights", {})
                    st.caption(f"Pesos → Qualidade={w.get('quality',0.7):.2f} • Tempo={w.get('time',0.2):.2f} • Tokens={w.get('tokens',0.1):.2f}")
                with right:
                    if st.button("📋 Duplicar", key=f"dup_{c['id']}"):
                        dup = dict(c)
                        dup["id"] = str(uuid.uuid4())
                        dup["title"] = c["title"] + " (cópia)"
                        dup["created_at"] = datetime.utcnow().isoformat()
                        add_to_jsonl("cases.jsonl", dup)
                        st.success("Cópia criada.")
                        st.rerun()
                    if st.button("🗑️ Excluir", key=f"del_{c['id']}"):
                        delete_from_jsonl("cases.jsonl", c["id"])
                        st.success("Excluído.")
                        st.rerun()
    else:
        st.info("Nenhum caso ainda — crie o primeiro acima.")

def render_test_execution(endpoint: str, api_key: str, deployment: str, api_version: str):
    st.markdown("### 🚀 Executar Testes")
    cases = load_jsonl("cases.jsonl")
    if not cases:
        st.warning("Crie um caso primeiro.")
        return

    case_map = {c["id"]: f"{c['title']} — {c.get('industry','N/A')}" for c in cases}
    case_id = st.selectbox("Caso", options=list(case_map.keys()), format_func=lambda x: case_map[x])
    selected = next(c for c in cases if c["id"] == case_id)

    with st.expander("📋 Preview do caso"):
        st.write(selected["context"][:1000] + ("..." if len(selected["context"]) > 1000 else ""))

    st.write("**Escolha abordagens:**")
    chosen: List[str] = []
    cols = st.columns(3)
    keys = list(THINKING_MODELS.keys())
    for i, k in enumerate(keys):
        with cols[i % 3]:
            if st.checkbox(k, value=False, key=f"ch_{i}"):
                chosen.append(k)

    c1, c2, c3 = st.columns(3)
    with c1:
        repetitions = st.number_input("Repetições por abordagem", min_value=1, max_value=10, value=2, step=1)
    with c2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    with c3:
        use_azure = st.toggle("Usar Azure OpenAI", value=False)

    if st.button("▶️ Executar", type="primary"):
        if not chosen:
            st.error("Selecione ao menos uma abordagem.")
            return
        run_id = str(uuid.uuid4())
        run = {
            "id": run_id,
            "case_id": case_id,
            "approaches": chosen,
            "repetitions": int(repetitions),
            "use_azure": bool(use_azure),
            "temperature": float(temperature),
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None
        }
        add_to_jsonl("runs.jsonl", run)

        progress = st.progress(0.0, text="Rodando...")
        total = len(chosen) * int(repetitions)
        done = 0

        for approach in chosen:
            tmpl = THINKING_MODELS[approach]["prompt_template"]
            prompt = tmpl.replace("{context}", selected["context"])
            for rep in range(1, int(repetitions) + 1):
                if use_azure:
                    res = execute_with_azure(prompt, endpoint, api_key, deployment, api_version, temperature)
                else:
                    res = execute_with_heuristic(approach, selected["context"])

                attempt = {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "case_id": case_id,
                    "approach": approach,
                    "iteration": rep,
                    "prompt_used": prompt,
                    "response_text": res.get("response_text", ""),
                    "latency_ms": res.get("latency_ms"),
                    "input_tokens": res.get("input_tokens"),
                    "output_tokens": res.get("output_tokens"),
                    "quality_score": 0,   # user will rate later
                    "notes": "",
                    "created_at": datetime.utcnow().isoformat()
                }
                add_to_jsonl("attempts.jsonl", attempt)
                done += 1
                progress.progress(done/total, text=f"{done}/{total} concluído(s)")
        # close run
        runs = load_jsonl("runs.jsonl")
        for r in runs:
            if r["id"] == run_id:
                r["ended_at"] = datetime.utcnow().isoformat()
        save_to_jsonl("runs.jsonl", runs)
        progress.empty()
        st.success("Execução concluída. Vá para 'Resultados & Ranking' para avaliar e ver o ranking.")

def render_results_and_ranking():
    st.markdown("### 📊 Resultados & Ranking")
    cases = load_jsonl("cases.jsonl")
    attempts = load_jsonl("attempts.jsonl")
    if not cases:
        st.info("Crie um caso primeiro.")
        return
    case_map = {c["id"]: f"{c['title']} — {c.get('industry','N/A')}" for c in cases}
    cid = st.selectbox("Caso", list(case_map.keys()), format_func=lambda x: case_map[x], key="case_results")
    current = next(c for c in cases if c["id"] == cid)
    weights = current.get("weights", {"quality":0.7,"time":0.2,"tokens":0.1})

    # Filter attempts
    items = [a for a in attempts if a["case_id"] == cid]
    st.caption(f"{len(items)} tentativa(s) registradas para este caso.")

    if items:
        # Editable table for quality_score and notes
        editable_rows = []
        for a in items:
            editable_rows.append({
                "id": a["id"],
                "approach": a["approach"],
                "iteration": a["iteration"],
                "latency_ms": a.get("latency_ms"),
                "input_tokens": a.get("input_tokens"),
                "output_tokens": a.get("output_tokens"),
                "quality_score": a.get("quality_score", 0),
                "notes": a.get("notes",""),
                "response_preview": (a.get("response_text","")[:140] + "...") if a.get("response_text") and len(a["response_text"])>140 else a.get("response_text","")
            })

        st.write("Edite **quality_score (1–5)** e **notes** conforme avaliação:")
        edited = st.data_editor(
            editable_rows,
            hide_index=True,
            column_config={
                "quality_score": st.column_config.NumberColumn(min_value=0, max_value=5, step=1),
                "notes": st.column_config.TextColumn(max_chars=400),
            },
            use_container_width=True,
            num_rows="fixed"
        )

        if st.button("💾 Salvar avaliações"):
            # Persist back to attempts.jsonl
            by_id = {e["id"]: e for e in edited}
            all_attempts = load_jsonl("attempts.jsonl")
            for i, a in enumerate(all_attempts):
                if a["id"] in by_id:
                    a["quality_score"] = by_id[a["id"]].get("quality_score", 0)
                    a["notes"] = by_id[a["id"]].get("notes", "")
            save_to_jsonl("attempts.jsonl", all_attempts)
            st.success("Avaliações salvas.")
            st.rerun()

        st.markdown("---")
        st.subheader("🏆 Ranking por abordagem")
        ranked = calculate_ranking(items, weights)
        if ranked:
            st.table([{
                "Abordagem": r["approach"],
                "Score": round(r["final_score"], 3),
                "Qualidade média": round(r["quality_avg"], 2),
                "Latência média (ms)": int(r["latency_avg"]) if r["latency_avg"] else 0,
                "Tokens médios": int(r["tokens_avg"]) if r["tokens_avg"] else 0,
                "Tentativas válidas": r["attempts_count"]
            } for r in ranked])

            if px is not None:
                import pandas as pd
                df = pd.DataFrame(ranked)
                fig = px.bar(df, x="approach", y="final_score", title="Score por abordagem")
                st.plotly_chart(fig, use_container_width=True)

        # Export
        st.markdown("#### Exportar")
        colx, coly = st.columns(2)
        st.download_button(
            "⬇️ Baixar tentativas (JSON)",
            data=json.dumps(items, ensure_ascii=False, indent=2),
            file_name="attempts_filtered.json",
            mime="application/json",
            use_container_width=True
        )
        if coly:
            import pandas as pd
            df = pd.DataFrame(items)
            st.download_button(
                "⬇️ Baixar tentativas (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="attempts_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Ainda não há tentativas para este caso. Vá em 'Executar Testes'.")

def render_experiments_tab():
    st.markdown("## 🧪 Experimentos & Benchmark")
    st.caption("Compare abordagens de prompt em casos reais e gere rankings baseados em métricas.")
    endpoint, api_key, deployment, api_version = setup_azure_openai()
    tab1, tab2, tab3 = st.tabs(["📋 Casos", "🚀 Executar Testes", "📊 Resultados & Ranking"])
    with tab1:
        render_cases_management()
    with tab2:
        render_test_execution(endpoint, api_key, deployment, api_version)
    with tab3:
        render_results_and_ranking()

# ---------------------------------------
# Analysis (recommendation) with Azure
# ---------------------------------------
def analyze_with_azure_openai(text: str, endpoint: str, api_key: str, deployment: str, api_version: str) -> Dict[str, Any]:
    if not (has_openai and endpoint and api_key and deployment):
        return {"erro": "Azure OpenAI não configurado."}

    sys_prompt = """Você é especialista em engenharia de prompt. Escolha a melhor abordagem entre:
- Chain of Thought (CoT)
- Skeleton of Thought (SoT)
- Tree of Thought (ToT)
- Self-consistency
- Zero-shot
- Few-shot
Responda APENAS JSON válido com:
{
  "modelo_recomendado": "...",
  "confianca": "alta|média|baixa",
  "justificativa": "... (3-6 linhas)",
  "aplicacao_pratica": "...",
  "modelos_alternativos": ["..."],
  "consideracoes_especiais": "...",
  "template_sugerido": "prompt curto pronto para usar"
}
"""

    prompt = f"Contexto:\n{text}\n\nRetorne somente o JSON pedido."
    try:
        client = openai.AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700
        )
        content = resp.choices[0].message.content if resp and resp.choices else "{}"
        data = json.loads(content)
        return data
    except Exception as e:
        return {"erro": f"Erro Azure: {e}"}

# ---------------------------------------
# Models overview & recommendation
# ---------------------------------------
def render_models_overview():
    st.markdown("## 📚 Abordagens de Prompt")
    exp_cols = st.columns(2)
    items = list(THINKING_MODELS.items())
    for i, (name, meta) in enumerate(items):
        with exp_cols[i % 2]:
            with st.expander(name, expanded=False):
                st.markdown(f"**O que é:** {meta['descricao']}")
                st.markdown(f"**Quando usar:** {meta['quando_usar']}")
                st.markdown("**Exemplo:**")
                st.markdown(f"<div class='example-box'>{meta['exemplo']}</div>", unsafe_allow_html=True)
                st.markdown("**Template curto:**")
                st.code(meta["prompt_template"], language="markdown")

def render_recommendation_block():
    st.markdown("## 🎯 Recomendação de abordagem (Azure opcional)")
    endpoint, api_key, deployment, api_version = setup_azure_openai()

    default_ctx = "Banco quer priorizar casos de IA para fraude; precisa balancear custo, risco e prazo."
    ctx = st.text_area("Contexto do cliente/tarefa", value=default_ctx, height=160)

    c1, c2 = st.columns([1,3])
    with c1:
        use_azure = st.toggle("Usar Azure OpenAI", value=False)
    with c2:
        st.caption("Sem Azure, use a aba de **Experimentos** para testar heurísticas.")

    if st.button("🔍 Sugerir abordagem"):
        if use_azure:
            data = analyze_with_azure_openai(ctx, endpoint, api_key, deployment, api_version)
            if "erro" in data:
                st.error(data["erro"])
            else:
                st.markdown("<div class='analysis-result'>", unsafe_allow_html=True)
                st.write(f"**Modelo recomendado:** {data.get('modelo_recomendado','—')}")
                st.write(f"**Confiança:** {data.get('confianca','—')}")
                st.write(f"**Justificativa:** {data.get('justificativa','—')}")
                st.write(f"**Como aplicar:** {data.get('aplicacao_pratica','—')}")
                st.write(f"**Alternativas:** {', '.join(data.get('modelos_alternativos', []))}")
                if data.get("template_sugerido"):
                    st.markdown("**Template sugerido:**")
                    st.code(data["template_sugerido"], language="markdown")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Azure desativado. Use a aba **Experimentos** para rodar heurístico e comparar abordagens.")

# ---------------------------------------
# Main
# ---------------------------------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Prompting Approaches Explorer</h1>
        <p>Compare, experimente e escolha a melhor abordagem de prompt para cada caso.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_experiments = st.tabs(["📚 Abordagens & Recomendação", "🧪 Experimentos"])
    with tab_overview:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            render_models_overview()
            st.markdown("</div>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            render_recommendation_block()
            st.markdown("</div>", unsafe_allow_html=True)
    with tab_experiments:
        render_experiments_tab()

if __name__ == "__main__":
    main()
