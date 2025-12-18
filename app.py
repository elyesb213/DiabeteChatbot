import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# ============ 1. CONFIGURATION DE LA PAGE (DOIT ÊTRE EN PREMIER) ============
st.set_page_config(
    page_title="DiabèteBot+", 
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 2. CONFIGURATION DES CHEMINS ============
FAQ_PATH = "faqdiabete.json"
KNOWLEDGE_PATH = "knowledge.txt"

# ============ 3. STYLE CSS PERSONNALISÉ ============
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 15px 20px; border-radius: 20px 20px 5px 20px;
        margin: 10px 0; max-width: 80%; float: right; clear: both; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .bot-message {
        background: white; color: #333; padding: 15px 20px; border-radius: 20px 20px 20px 5px;
        margin: 10px 0; max-width: 80%; float: left; clear: both; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .source-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.75em; margin-top: 8px; font-weight: bold; }
    .source-faq { background: #10b981; color: white; }
    .source-knowledge { background: #3b82f6; color: white; }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 15px; margin-bottom: 30px; color: white; text-align: center;
    }
    .urgence-box { background: #fee2e2; border-left: 5px solid #ef4444; padding: 15px; border-radius: 8px; margin: 20px 0; }
    .clearfix::after { content: ""; clear: both; display: table; }
</style>
""", unsafe_allow_html=True)

# ============ 4. CHARGEMENT DES DONNÉES ET DU MODÈLE ============

@st.cache_resource
def load_model():
    # Utilisation d'un modèle multilingue léger
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_data(faq_p, know_p):
    # Chargement FAQ
    if os.path.exists(faq_p):
        with open(faq_p, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
        questions = [item["question"] for item in faq_data]
        answers = [item["answer"] for item in faq_data]
    else:
        st.error(f"Fichier {faq_p} introuvable !")
        questions, answers = [], []

    # Chargement Knowledge
    if os.path.exists(know_p):
        with open(know_p, "r", encoding="utf-8") as f:
            full_text = f.read()
        chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
    else:
        st.error(f"Fichier {know_p} introuvable !")
        chunks = []
        
    return questions, answers, chunks

# Initialisation
model = load_model()
faq_questions, faq_answers, knowledge_chunks = load_data(FAQ_PATH, KNOWLEDGE_PATH)

# Encodage (mis en cache pour éviter de recalculer à chaque interaction)
@st.cache_data
def get_embeddings(_model, _questions, _chunks):
    faq_emb = _model.encode(_questions) if _questions else []
    know_emb = _model.encode(_chunks) if _chunks else []
    return faq_emb, know_emb

faq_embeddings, knowledge_embeddings = get_embeddings(model, faq_questions, knowledge_chunks)

# ============ 5. LOGIQUE IA ============

def retrieve_from_faq(user_question):
    if not faq_questions: return "Désolé, FAQ vide.", "", 0
    q_vec = model.encode([user_question])
    sims = cosine_similarity(q_vec, faq_embeddings)[0]
    best_idx = int(np.argmax(sims))
    return faq_answers[best_idx], faq_questions[best_idx], float(sims[best_idx])

def retrieve_from_knowledge(user_question, top_k=2):
    if not knowledge_chunks: return [("Base de connaissances indisponible.", 0)]
    q_vec = model.encode([user_question])
    sims = cosine_similarity(q_vec, knowledge_embeddings)[0]
    best_indices = sims.argsort()[::-1][:top_k]
    return [(knowledge_chunks[i], float(sims[i])) for i in best_indices]

def answer_user(user_question):
    faq_answer, matched_q, faq_score = retrieve_from_faq(user_question)

    if faq_score > 0.65:
        final_text = f"📌 **Réponse :** {faq_answer}\n\n*Basé sur : \"{matched_q}\"*"
        source = "FAQ"
    else:
        passages = retrieve_from_knowledge(user_question, top_k=2)
        synthese = "**Informations pertinentes :**\n\n"
        for (p, sc) in passages:
            synthese += "• " + p.strip() + "\n\n"
        synthese += "---\n\n⚠️ *Ceci est une information générale.*"
        final_text = synthese
        source = "knowledge"

    return final_text, source

# ============ 6. INTERFACE ET HISTORIQUE ============

if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR
with st.sidebar:
    st.markdown("### 🩸 DiabèteBot+")
    st.info("Assistant IA pour le diabète de type 2.")
    st.metric("Base FAQ", len(faq_questions))
    if st.button("🔄 Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("""<div class="urgence-box"><strong>☎️ Urgence : Appelez le 15</strong></div>""", unsafe_allow_html=True)

# EN-TÊTE
st.markdown("""<div class="header-container"><h1>🩸 DiabèteBot+</h1><p>Votre assistant intelligent</p></div>""", unsafe_allow_html=True)

# AFFICHAGE CHAT
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "bot-message"
    label = "👤 Vous" if message["role"] == "user" else "🤖 DiabèteBot+"
    source_badge = ""
    if message["role"] != "user":
        s_class = "source-faq" if message.get("source") == "FAQ" else "source-knowledge"
        s_text = "FAQ ✅" if message.get("source") == "FAQ" else "Base de connaissances 📚"
        source_badge = f'<br><span class="source-badge {s_class}">{s_text}</span>'
    
    st.markdown(f"""<div class="clearfix"><div class="{role_class}"><strong>{label}</strong><br>{message["content"]}{source_badge}</div></div>""", unsafe_allow_html=True)

# ZONE DE SAISIE
st.markdown("---")
user_input = st.chat_input("Posez votre question sur le diabète...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Réflexion en cours..."):
        bot_reply, source_used = answer_user(user_input)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply, "source": source_used})
    st.rerun()

st.warning("⚠️ Ce chatbot ne remplace pas un avis médical.")