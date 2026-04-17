"""
AI Dev Studio — Multi-Agent Project Development System
Streamlit app using OpenRouter free models for full-stack enterprise project generation.
Includes live API key validation and dynamic model discovery from OpenRouter.
"""

import streamlit as st
import requests
import time
from datetime import datetime

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
OPENROUTER_API_URL    = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_AUTH_URL   = "https://openrouter.ai/api/v1/auth/key"
ANTHROPIC_BASE_URL    = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY    = st.secrets["or_api_key"]
# Default preferred models (overridden by live fetch + user selection)
DEFAULT_AGENT_MODELS = {
    "BA Agent":        "qwen/qwen3-235b-a22b:free",
    "Architect Agent": "qwen/qwen3-235b-a22b:free",
    "LLD Agent":       "qwen/qwen3-235b-a22b:free",
    "Dev Agent":       "mistralai/devstral-small:free",
    "Reviewer Agent":  "meta-llama/llama-3.3-70b-instruct:free",
}

AGENT_META = {
    "BA Agent":        {"icon": "📋", "role": "Senior Business Analyst",     "tag": "Requirements Engineering"},
    "Architect Agent": {"icon": "🏛️", "role": "Enterprise Solution Architect","tag": "System Design"},
    "LLD Agent":       {"icon": "🔩", "role": "Principal Designer",           "tag": "Low-Level Design"},
    "Dev Agent":       {"icon": "💻", "role": "Senior Java Developer",        "tag": "Code Generation"},
    "Reviewer Agent":  {"icon": "🔍", "role": "Principal Code Reviewer",      "tag": "Quality Assurance"},
}

AGENT_SYSTEM_PROMPTS = {
    "BA Agent": """You are a Senior Business Analyst. Your role is REQUIREMENTS EXTRACTION.
Analyze the user's prompt and produce a comprehensive, structured document covering:

## 1. Business Objectives
## 2. Stakeholders / Actors
## 3. Functional Requirements (FRs) — numbered list
## 4. Non-Functional Requirements (NFRs) — performance, scalability, security
## 5. Business Rules — critical logic rules
## 6. Inputs / Outputs — data contracts
## 7. Edge Cases & Exceptions
## 8. Assumptions
## 9. Gaps / Open Questions — explicitly flag missing information with ⚠️

Rules:
- Be precise and implementation-ready
- Number every requirement
- Flag gaps explicitly with ⚠️
- Use markdown tables where helpful
- Prioritize enterprise-grade completeness""",

    "Architect Agent": """You are an Enterprise Solution Architect. Your role is SOLUTION ARCHITECTURE.
Given the requirements, design a production-grade architecture covering:

## 1. Architecture Overview (pattern: layered/microservices/event-driven — justify choice)
## 2. Component Design
   - API Layer
   - Service Layer
   - Rule Engine
   - Data Layer
   - Integration Layer
## 3. Technology Stack
   - Java / Spring Boot (version + justification)
   - Workflow system (if applicable)
   - Database (with justification)
   - Messaging (Kafka/RabbitMQ if needed)
## 4. Data Flow (step-by-step numbered)
## 5. Rule Engine Design (structure, config approach, execution flow)
## 6. Mermaid Diagram (MANDATORY — wrap in ```mermaid block)
## 7. Scalability & Performance strategy
## 8. Error Handling strategy
## 9. Security Considerations
## 10. Deployment Architecture (Docker/K8s/cloud)

Rules:
- Prefer configuration-based, rule-driven design
- Modular, loosely coupled components
- Always include the Mermaid diagram
- Be implementation-ready""",

    "LLD Agent": """You are a Principal Designer. Your role is LOW-LEVEL DESIGN (LLD).
Generate developer-ready design artifacts:

## 1. Project Structure (Maven standard layered)
## 2. Class Design Table
   | Class Name | Layer | Responsibility | Key Methods |
## 3. API Contracts
   For each endpoint: Method, Path, Request Body, Response Body, HTTP Status Codes
## 4. Rule Engine Detailed Design (interfaces, strategies, config)
## 5. Database Schema — full DDL SQL
## 6. Sequence Diagrams (Mermaid sequenceDiagram blocks)
## 7. Integration Points (external systems, adapters, error handling)
## 8. Exception Hierarchy

Rules:
- Single Responsibility per class
- Include validation logic
- Proper REST naming conventions
- Pagination for list endpoints
- Include audit fields in all entities""",

    "Dev Agent": """You are a Senior Java Developer. Your role is CODE GENERATION.
Generate production-quality Spring Boot code:

## 1. pom.xml — complete with all dependencies
## 2. Controller Layer — REST controllers with @Valid, @RestController
## 3. Service Layer — interfaces + implementations
## 4. Rule Engine Implementation — Strategy or Chain-of-Responsibility pattern
## 5. Model / Entity Classes — with full JPA annotations
## 6. Repository Layer — Spring Data JPA with custom queries
## 7. DTOs — Request/Response classes with validation annotations
## 8. application.yml — complete configuration (externalize all rules/config)
## 9. GlobalExceptionHandler — @ControllerAdvice
## 10. Unit Test skeletons — JUnit 5 + Mockito

Rules:
- Clean code: meaningful names, small methods, no magic numbers
- All business rules must be externalized to application.yml or database
- Use @Valid everywhere
- Javadoc on all public APIs
- Each file in its own ```java block with // filename: as first comment
- No hardcoded credentials — use environment variables""",

    "Reviewer Agent": """You are a Principal Code Reviewer and QA Lead. Your role is REVIEW & VALIDATION.
Critically review ALL previous agent outputs and produce:

## 1. Executive Summary
   - Overall quality score: X/10
   - Key strengths
   - Critical issues count

## 2. Requirements Review (BA Agent)
   - Missing requirements ⚠️
   - Ambiguous statements ❓
   - Completeness score: X/10

## 3. Architecture Review (Architect Agent)
   - Design flaws 🔴
   - Scalability risks ⚠️
   - Security gaps 🔐
   - Architecture score: X/10

## 4. LLD Review
   - Design gaps
   - Missing classes or contracts
   - LLD score: X/10

## 5. Code Review (Dev Agent)
   - Code smells 🚨
   - Security vulnerabilities 🔐
   - Performance anti-patterns ⚡
   - Missing validations
   - Code quality score: X/10

## 6. Prioritized Recommendations
   | Priority | Issue | Location | Suggested Fix |
   P1 = Critical, P2 = Important, P3 = Nice-to-have

## 7. Corrected Code Snippets
   For every P1 issue found, provide the corrected code in a ```java block

Rules:
- Be brutally honest but constructive
- Security > Correctness > Performance > Maintainability
- Provide specific fixes for all critical issues
- Never leave a P1 issue without a concrete remediation""",
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Dev Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #070B14; color: #E2E8F0; }

section[data-testid="stSidebar"] {
    background: #0C1220 !important;
    border-right: 1px solid #1A2744;
}

.studio-header {
    background: linear-gradient(135deg, #0D1526 0%, #14093A 50%, #0D1526 100%);
    border: 1px solid #2A3F6F; border-radius: 18px;
    padding: 1.8rem 2.2rem; margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
}
.studio-header::after {
    content: ''; position: absolute; bottom: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.studio-header h1 {
    font-size: 2rem; font-weight: 800; margin: 0 0 0.25rem 0;
    background: linear-gradient(90deg, #818CF8 0%, #C084FC 50%, #38BDF8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.studio-header p { color: #7A8FAF; font-size: 0.88rem; margin: 0; font-weight: 300; }

.key-valid {
    background: rgba(16,185,129,0.1); border: 1px solid #059669;
    border-radius: 8px; padding: 0.5rem 0.8rem;
    font-size: 0.78rem; color: #34D399; margin-top: 0.4rem;
}
.key-invalid {
    background: rgba(239,68,68,0.1); border: 1px solid #DC2626;
    border-radius: 8px; padding: 0.5rem 0.8rem;
    font-size: 0.78rem; color: #FCA5A5; margin-top: 0.4rem;
}
.key-hint {
    background: rgba(245,158,11,0.08); border: 1px solid #92400E;
    border-radius: 8px; padding: 0.5rem 0.8rem;
    font-size: 0.75rem; color: #FDE68A; margin-top: 0.4rem;
}

.agent-row {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.55rem 0.8rem; background: #0F1829;
    border: 1px solid #1A2744; border-radius: 10px; margin-bottom: 0.4rem;
}
.agent-row.running { border-color: #F59E0B; background: rgba(245,158,11,0.06); }
.agent-row.done    { border-color: #059669; background: rgba(16,185,129,0.06); }
.agent-row.error   { border-color: #DC2626; background: rgba(239,68,68,0.06); }
.a-icon  { font-size: 1.1rem; flex-shrink: 0; }
.a-name  { font-size: 0.8rem; font-weight: 600; color: #CBD5E1; }
.a-tag   { font-size: 0.62rem; color: #64748B; }
.a-model { font-family: 'JetBrains Mono',monospace; font-size: 0.58rem; color: #475569; margin-top: 1px; }
.a-status{ margin-left: auto; font-size: 0.72rem; font-weight: 600; flex-shrink:0; }

.model-pill {
    display: inline-block; background: #131F35; border: 1px solid #1E3A5F;
    border-radius: 20px; padding: 2px 10px;
    font-family: 'JetBrains Mono',monospace; font-size: 0.6rem; color: #7DD3FC;
}

.metric-box {
    background: #0F1829; border: 1px solid #1A2744;
    border-radius: 10px; padding: 0.8rem 1rem; text-align: center;
}
.metric-box .num {
    font-size: 1.4rem; font-weight: 700; color: #818CF8;
    font-family: 'JetBrains Mono',monospace;
}
.metric-box .lbl { font-size: 0.63rem; color: #475569; text-transform: uppercase; letter-spacing: 0.07em; }

.stTabs [data-baseweb="tab-list"] {
    background: #0F1829; border-radius: 10px; padding: 4px; gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; color: #64748B !important;
    font-family:'Sora',sans-serif !important; font-size:0.82rem !important;
}
.stTabs [aria-selected="true"] { background: #1A2E50 !important; color: #E2E8F0 !important; }

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#4F46E5,#7C3AED) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    font-family: 'Sora',sans-serif !important;
}
div.stDownloadButton > button {
    background: #131F35 !important; color: #7DD3FC !important;
    border: 1px solid #1E3A5F !important; border-radius: 8px !important;
    font-family:'Sora',sans-serif !important; font-size:0.8rem !important;
}
.stTextArea textarea {
    background: #0F1829 !important; border: 1px solid #1A2744 !important;
    border-radius: 10px !important; color: #E2E8F0 !important;
    font-family: 'Sora',sans-serif !important;
}
hr { border-color: #1A2744 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #070B14; }
::-webkit-scrollbar-thumb { background: #1A2744; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
_defaults = {
    "results": {},
    "pipeline_status": {a: "pending" for a in AGENT_META},
    "run_complete": False,
    "elapsed": {},
    "key_valid": None,
    "key_info": {},
    "free_models": [],
    "agent_models": dict(DEFAULT_AGENT_MODELS),
    "models_fetched": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def validate_api_key(api_key: str):
    try:
        r = requests.get(
            OPENROUTER_AUTH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            return True, r.json().get("data", {})
        return False, {"error": r.text[:200]}
    except Exception as e:
        return False, {"error": str(e)}

def fetch_free_models(api_key: str):
    try:
        r = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        free = []
        for m in r.json().get("data", []):
            mid = m.get("id", "")
            if mid.endswith(":free"):
                ctx = m.get("context_length", 0)
                free.append({
                    "id": mid,
                    "name": m.get("name", mid),
                    "label": f"{m.get('name', mid)}  [{ctx//1000}K]",
                    "context": ctx,
                })
        free.sort(key=lambda x: x["context"], reverse=True)
        return free
    except Exception:
        return []

def fetch_available_models(api_key: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(OPENROUTER_MODELS_URL, headers=headers)
        response.raise_for_status()
        return response.json()  # Return the list of models
    except Exception as e:
        return f"⚠️ **Error fetching models**: {str(e)}"

def call_openrouter(api_key: str, model: str, user_message: str) -> str:
    system_prompt = AGENT_SYSTEM_PROMPTS.get(model, "Default system prompt if model not found.")  # Use a default prompt if model is not found

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit AI Architect App"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.3
    }

    try:
        resp = requests.post(ANTHROPIC_BASE_URL, headers=headers, json=payload, timeout=180)
        print(f"Raw Response: {resp.text}")  # Log the raw response
        resp.raise_for_status()
        data = resp.json()
        if resp.status_code != 200:
            print(f"⚠️ **Error**: Received status code {resp.status_code} with response: {resp.text}")
            return f"⚠️ **Error**: Received status code {resp.status_code}"
        choices = data.get("choices", [])
        if not choices:
            return f"⚠️ **Empty Response**: {str(data)[:300]}"
        return choices[0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ **Timeout (180s)**: Model is busy. Try a lighter model or retry."
    except Exception as e:
        return f"⚠️ **Error**: {str(e)}"


def run_agent(api_key: str, agent_name: str, user_prompt: str, prior_context: str = "") -> str:
    model = st.session_state.agent_models.get(agent_name, DEFAULT_AGENT_MODELS[agent_name])
    return call_openrouter(api_key, model, user_prompt)


def build_reviewer_ctx(results: dict, prompt: str) -> str:
    ctx = f"## Original User Prompt\n{prompt}\n\n"
    for a, out in results.items():
        if a != "Reviewer Agent":
            ctx += f"---\n## {a} Output\n{out}\n\n"
    return ctx


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.4rem 0 1rem 0;">
        <div style="font-size:1.25rem;font-weight:800;
             background:linear-gradient(90deg,#818CF8,#C084FC);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            ⚡ AI Dev Studio
        </div>
        <div style="font-size:0.65rem;color:#334155;font-family:'JetBrains Mono',monospace;">
            v2.0 · OpenRouter Multi-Agent Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API KEY ──
    # st.markdown("#### 🔑 OpenRouter API Key")
    api_key = st.secrets["or_api_key"]
    # st.text_input(
    #     "API Key", type="password",
    #     placeholder="sk-or-v1-...",
    #     label_visibility="collapsed",
    #     key="api_key_input", value=""
    # )

    c1, c2 = st.columns(2)
    with c1:
        do_validate = st.button("✅ Validate", use_container_width=True)
    with c2:
        do_fetch = st.button("🔄 Load Models", use_container_width=True)

    if do_validate:
        if api_key:
            with st.spinner("Validating…"):
                ok, info = validate_api_key(api_key)
                st.session_state.key_valid = ok
                st.session_state.key_info  = info
        else:
            st.warning("Enter an API key first.")

    if do_fetch:
        if api_key:
            with st.spinner("Fetching free models…"):
                models = fetch_free_models(api_key)
                if models:
                    st.session_state.free_models   = models
                    st.session_state.models_fetched = True
                    st.success(f"✅ {len(models)} free models loaded!")
                else:
                    st.error("No models returned. Check your API key.")
        else:
            st.warning("Enter an API key first.")

    # Key status banner
    if st.session_state.key_valid is True:
        info  = st.session_state.key_info
        label = info.get("label", "API Key")
        usage = f"${info.get('usage', 0):.4f}"
        st.markdown(f'<div class="key-valid">✓ <strong>Valid</strong> · {label}<br>Usage: {usage}</div>',
                    unsafe_allow_html=True)
    elif st.session_state.key_valid is False:
        err = st.session_state.key_info.get("error", "Unknown")
        st.markdown(f'<div class="key-invalid">✗ Invalid key — {err[:80]}</div>',
                    unsafe_allow_html=True)
    elif not api_key:
        st.markdown("""<div class="key-hint">⚠️ Add your OpenRouter key.<br>
            <a href="https://openrouter.ai/keys" target="_blank" style="color:#FCD34D;">
            Get free key →</a></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── MODEL ASSIGNMENT ──
    st.markdown("#### 🤖 Agent → Model")

    if st.session_state.models_fetched and st.session_state.free_models:
        model_labels = [m["label"] for m in st.session_state.free_models]
        id_to_lbl    = {m["id"]: m["label"] for m in st.session_state.free_models}
        lbl_to_id    = {m["label"]: m["id"] for m in st.session_state.free_models}

        for aname, ameta in AGENT_META.items():
            cur_id  = st.session_state.agent_models.get(aname, DEFAULT_AGENT_MODELS[aname])
            cur_lbl = id_to_lbl.get(cur_id, model_labels[0] if model_labels else "")
            st.markdown(
                f"<div style='font-size:0.76rem;font-weight:600;color:#94A3B8;"
                f"margin:4px 0 2px 0;'>{ameta['icon']} {aname}</div>",
                unsafe_allow_html=True,
            )
            chosen = st.selectbox(
                f"_mdl_{aname}",
                model_labels,
                index=model_labels.index(cur_lbl) if cur_lbl in model_labels else 0,
                label_visibility="collapsed",
                key=f"sel_{aname}",
            )
            st.session_state.agent_models[aname] = lbl_to_id[chosen]

            # Add a display for the selected agent prompt
            selected_agent_prompt = AGENT_SYSTEM_PROMPTS.get(aname, "No prompt available.")
            if st.checkbox("Show Prompt", key=f"show_prompt_{aname}"):  # Use a unique key for each agent
                st.text_area("Agent Prompt", selected_agent_prompt, height=300)
    else:
        for aname, ameta in AGENT_META.items():
            mid = st.session_state.agent_models.get(aname, DEFAULT_AGENT_MODELS[aname])
            st.markdown(f"""
            <div class="agent-row">
                <span class="a-icon">{ameta['icon']}</span>
                <div style="flex:1;min-width:0;">
                    <div class="a-name">{aname}</div>
                    <div class="a-model">{mid.split('/')[-1]}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.caption("Click **Load Models** to assign models interactively.")

    st.markdown("---")

    # ── PIPELINE STATUS ──
    st.markdown("#### 📊 Pipeline Status")
    s_icons  = {"pending":"○","running":"◎","done":"✓","error":"✗"}
    s_colors = {"pending":"#475569","running":"#F59E0B","done":"#10B981","error":"#EF4444"}

    for aname, ameta in AGENT_META.items():
        status  = st.session_state.pipeline_status.get(aname, "pending")
        elapsed = st.session_state.elapsed.get(aname, "")
        e_str   = f"{elapsed:.1f}s" if isinstance(elapsed, float) else "—"
        css_cls = status if status in ("running","done","error") else ""
        st.markdown(f"""
        <div class="agent-row {css_cls}">
            <span class="a-icon">{ameta['icon']}</span>
            <div style="flex:1;min-width:0;">
                <div class="a-name">{aname}</div>
                <div class="a-tag">{ameta['tag']}</div>
            </div>
            <div class="a-status" style="color:{s_colors[status]};">
                {s_icons[status]} {e_str}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── OPTIONS ──
    st.markdown("#### ⚙️ Options")
    selected_agents = st.multiselect(
        "Agents to run", list(AGENT_META.keys()),
        default=list(AGENT_META.keys()),
    )
    chain_context = st.checkbox(
        "Chain agent context", value=True,
        help="Pass each agent's output as context to the next agent",
    )

    st.markdown("""
    <div style="font-size:0.62rem;color:#1E293B;text-align:center;padding-top:0.8rem;">
        Powered by OpenRouter · Free Tier<br>Enterprise Java · Spring Boot
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class="studio-header">
    <h1>⚡ AI Dev Studio</h1>
    <p>5-Agent Enterprise Software Pipeline &nbsp;·&nbsp;
       BA → Architecture → LLD → Java Code → Code Review</p>
</div>
""", unsafe_allow_html=True)

# ── FREE MODEL BROWSER ──
if st.session_state.models_fetched and st.session_state.free_models:
    with st.expander(
        f"🆓 Free Model Browser — {len(st.session_state.free_models)} models available",
        expanded=False,
    ):
        search = st.text_input("Filter", placeholder="llama, qwen, mistral…",
                               label_visibility="collapsed")
        shown  = [m for m in st.session_state.free_models
                  if search.lower() in m["id"].lower() or
                     search.lower() in m["name"].lower()] \
                 if search else st.session_state.free_models

        cols = st.columns(3)
        for i, m in enumerate(shown[:60]):
            with cols[i % 3]:
                ctx_k = m["context"] // 1000
                st.markdown(f"""
                <div style="background:#0F1829;border:1px solid #1A2744;border-radius:8px;
                     padding:0.45rem 0.65rem;margin-bottom:0.35rem;">
                    <div style="font-size:0.72rem;font-weight:600;color:#CBD5E1;
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                         max-width:100%;">{m['name']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                         color:#475569;margin-top:2px;word-break:break-all;">{m['id']}</div>
                    <div style="font-size:0.62rem;color:#38BDF8;margin-top:3px;">
                        📐 {ctx_k}K ctx</div>
                </div>""", unsafe_allow_html=True)
        if len(shown) > 60:
            st.caption(f"Showing 60 of {len(shown)}. Refine search above.")

# ── PROMPT ──
st.markdown("### 📝 Project / BRD Prompt")
user_prompt = st.text_area(
    "prompt",
    placeholder="""Describe your project or paste your Business Requirements Document.

Example:
Build an enterprise loan origination system that:
- Accepts applications from individual and corporate customers
- Validates eligibility based on configurable credit scoring rules
- Integrates with external credit bureau REST APIs (Experian, CIBIL)
- Routes through workflow: auto-approve / manual review / reject
- Sends email + SMS notifications at each stage
- REST API consumed by a React frontend dashboard
- PostgreSQL storage with full audit trail + soft delete
- Role-based access: Applicant, Loan Officer, Manager, Admin
- Must handle 1,000 concurrent applications, <2s P95 response""",
    height=175,
    label_visibility="collapsed",
)

r1, r2, _sp = st.columns([2, 1, 5])
with r1:
    run_btn = st.button("⚡ Run Pipeline", type="primary", use_container_width=True)
with r2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.results         = {}
        st.session_state.pipeline_status = {a: "pending" for a in AGENT_META}
        st.session_state.run_complete    = False
        st.session_state.elapsed         = {}
        st.rerun()

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("🔑 Enter your OpenRouter API key in the sidebar.")
    elif not user_prompt.strip():
        st.error("📝 Enter a project prompt first.")
    elif not selected_agents:
        st.error("🤖 Select at least one agent in the sidebar.")
    else:
        st.session_state.results         = {}
        st.session_state.pipeline_status = {a: "pending" for a in AGENT_META}
        st.session_state.run_complete    = False
        st.session_state.elapsed         = {}

        prog     = st.progress(0, text="Initializing pipeline…")
        stat_ph  = st.empty()
        ctx_acc  = ""
        total    = len(selected_agents)

        for step, aname in enumerate(AGENT_META.keys()):
            if aname not in selected_agents:
                continue

            ameta    = AGENT_META[aname]
            model_id = st.session_state.agent_models.get(aname, DEFAULT_AGENT_MODELS[aname])
            short    = model_id.split("/")[-1]

            st.session_state.pipeline_status[aname] = "running"
            pct = min(max(int((step / max(total, 1)) * 92), 0), 100)
            prog.progress(pct, text=f"{ameta['icon']} {aname} → {short}…")
            stat_ph.info(f"{ameta['icon']} **{aname}** running on `{short}`…")

            t0 = time.time()

            if aname == "Reviewer Agent":
                ctx    = build_reviewer_ctx(st.session_state.results, user_prompt)
                result = run_agent(api_key, aname, user_prompt, ctx)
            elif chain_context and ctx_acc:
                result = run_agent(api_key, aname, user_prompt, ctx_acc)
            else:
                result = run_agent(api_key, aname, user_prompt)

            elapsed = time.time() - t0
            st.session_state.elapsed[aname] = elapsed

            if result.startswith("⚠️"):
                st.session_state.pipeline_status[aname] = "error"
            else:
                st.session_state.pipeline_status[aname] = "done"
                if chain_context:
                    ctx_acc += f"\n\n---\n## {aname} Output\n{result}"

            st.session_state.results[aname] = result

        prog.progress(100, text="✅ Pipeline complete!")
        stat_ph.success("✅ All selected agents finished! See results below.")
        st.session_state.run_complete = True
        st.rerun()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if st.session_state.results:
    st.markdown("---")

    done_c  = sum(1 for s in st.session_state.pipeline_status.values() if s == "done")
    err_c   = sum(1 for s in st.session_state.pipeline_status.values() if s == "error")
    t_time  = sum(v for v in st.session_state.elapsed.values() if isinstance(v, float))
    t_chars = sum(len(v) for v in st.session_state.results.values())

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, num, lbl, clr in [
        (m1, f"{done_c}/{len(AGENT_META)}", "Agents Done",     "#818CF8"),
        (m2, f"{t_time:.0f}s",              "Total Runtime",   "#818CF8"),
        (m3, f"{t_chars//1000}K",           "Chars Generated", "#818CF8"),
        (m4, str(len(st.session_state.results)), "Outputs",    "#818CF8"),
        (m5, str(err_c),                    "Errors",          "#EF4444" if err_c else "#10B981"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="num" style="color:{clr};">{num}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    present = [a for a in AGENT_META if a in st.session_state.results]
    tabs    = st.tabs([f"{AGENT_META[a]['icon']} {a}" for a in present])

    for tab, aname in zip(tabs, present):
        with tab:
            ameta    = AGENT_META[aname]
            result   = st.session_state.results[aname]
            elapsed  = st.session_state.elapsed.get(aname, "")
            e_str    = f"{elapsed:.1f}s" if isinstance(elapsed, float) else "—"
            model_id = st.session_state.agent_models.get(aname, DEFAULT_AGENT_MODELS[aname])
            status   = st.session_state.pipeline_status.get(aname, "pending")

            hc1, hc2 = st.columns([6, 2])
            with hc1:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.3rem 0 0.5rem 0;flex-wrap:wrap;">
                    <span style="font-size:1.2rem;">{ameta['icon']}</span>
                    <div>
                        <div style="font-size:0.92rem;font-weight:700;color:#E2E8F0;">{aname}</div>
                        <div style="font-size:0.72rem;color:#64748B;">{ameta['role']} · {ameta['tag']}</div>
                    </div>
                    <span class="model-pill">{model_id.split('/')[-1]}</span>
                    <span style="font-size:0.7rem;color:#475569;">⏱ {e_str}</span>
                </div>""", unsafe_allow_html=True)
            with hc2:
                st.download_button(
                    "⬇️ Download",
                    data=result,
                    file_name=f"{aname.replace(' ','_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    key=f"dl_{aname}",
                    use_container_width=True,
                )

            st.divider()
            if status == "error":
                st.error(result)
            else:
                st.markdown(result)

    # Full export
    st.markdown("---")
    st.markdown("### 📦 Export Full Report")

    full_md  = f"# AI Dev Studio — Full Pipeline Report\n"
    full_md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
    full_md += f"**Total Runtime**: {t_time:.1f}s  \n**Agents**: {done_c} done, {err_c} errors\n\n---\n\n"
    full_md += f"## 📝 Original Prompt\n{user_prompt}\n\n---\n"
    for a in AGENT_META:
        if a not in st.session_state.results:
            continue
        ameta    = AGENT_META[a]
        model_id = st.session_state.agent_models.get(a, DEFAULT_AGENT_MODELS[a])
        elapsed  = st.session_state.elapsed.get(a, "")
        e_str    = f"{elapsed:.1f}s" if isinstance(elapsed, float) else "—"
        full_md += f"\n# {ameta['icon']} {a}\n**Role**: {ameta['role']}  \n"
        full_md += f"**Model**: `{model_id}`  \n**Time**: {e_str}  \n\n"
        full_md += st.session_state.results[a] + "\n\n---\n"

    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "⬇️ Full Report (.md)",
            data=full_md,
            file_name=f"ai_dev_studio_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            "⬇️ Full Report (.txt)",
            data=full_md,
            file_name=f"ai_dev_studio_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# Empty state
elif not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 2rem;background:#0F1829;
         border:1px dashed #1A2744;border-radius:16px;margin-top:0.5rem;">
        <div style="font-size:2.6rem;margin-bottom:0.7rem;">⚡</div>
        <div style="font-size:0.98rem;font-weight:600;color:#94A3B8;margin-bottom:0.5rem;">
            Ready to generate your enterprise system
        </div>
        <div style="font-size:0.8rem;color:#475569;max-width:560px;margin:0 auto 1.2rem auto;line-height:1.7;">
            <strong style="color:#818CF8;">1️⃣</strong> Enter OpenRouter API key &nbsp;→&nbsp;
            <strong style="color:#818CF8;">2️⃣</strong> Click Validate &nbsp;→&nbsp;
            <strong style="color:#818CF8;">3️⃣</strong> Click Load Models &nbsp;→&nbsp;
            <strong style="color:#818CF8;">4️⃣</strong> Pick models per agent &nbsp;→&nbsp;
            <strong style="color:#818CF8;">5️⃣</strong> Enter prompt &nbsp;→&nbsp;
            <strong style="color:#818CF8;">6️⃣</strong> Run Pipeline
        </div>
        <div style="display:flex;justify-content:center;gap:0.5rem;flex-wrap:wrap;">
            <span style="background:#131F35;border:1px solid #1E3A5F;padding:3px 12px;
                  border-radius:20px;font-size:0.7rem;color:#7DD3FC;">📋 BA Analysis</span>
            <span style="background:#131F35;border:1px solid #1E3A5F;padding:3px 12px;
                  border-radius:20px;font-size:0.7rem;color:#7DD3FC;">🏛️ Architecture + Mermaid</span>
            <span style="background:#131F35;border:1px solid #1E3A5F;padding:3px 12px;
                  border-radius:20px;font-size:0.7rem;color:#7DD3FC;">🔩 LLD + DDL</span>
            <span style="background:#131F35;border:1px solid #1E3A5F;padding:3px 12px;
                  border-radius:20px;font-size:0.7rem;color:#7DD3FC;">💻 Spring Boot Code</span>
            <span style="background:#131F35;border:1px solid #1E3A5F;padding:3px 12px;
                  border-radius:20px;font-size:0.7rem;color:#7DD3FC;">🔍 Code Review</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

available_models = fetch_available_models(OPENROUTER_API_KEY)
print(available_models)  # Log the available models to the console
