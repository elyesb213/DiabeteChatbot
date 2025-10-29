import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ============ CONFIG ============
FAQ_PATH = "faqdiabete.json"
KNOWLEDGE_PATH = "knowledge.txt"

# ============ CHARGEMENT DES DONNÉES ============

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_faq(faq_path):
    with open(faq_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    questions = [item["question"] for item in faq_data]
    answers = [item["answer"] for item in faq_data]
    return questions, answers

@st.cache_data
def load_knowledge(knowledge_path):
    with open(knowledge_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
    return chunks

model = load_model()
faq_questions, faq_answers = load_faq(FAQ_PATH)
knowledge_chunks = load_knowledge(KNOWLEDGE_PATH)

faq_embeddings = model.encode(faq_questions)
knowledge_embeddings = model.encode(knowledge_chunks)

# ============ LOGIQUE IA ============

def retrieve_from_faq(user_question):
    q_vec = model.encode([user_question])
    sims = cosine_similarity(q_vec, faq_embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    return faq_answers[best_idx], faq_questions[best_idx], best_score

def retrieve_from_knowledge(user_question, top_k=2):
    q_vec = model.encode([user_question])
    sims = cosine_similarity(q_vec, knowledge_embeddings)[0]
    best_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in best_idx:
        results.append((knowledge_chunks[i], float(sims[i])))
    return results

def answer_user(user_question):
    faq_answer, matched_q, faq_score = retrieve_from_faq(user_question)

    if faq_score > 0.60:
        final_text = (
            f"📌 **Réponse :** {faq_answer}\n\n"
            f"*Basé sur : \"{matched_q}\"*"
        )
        source = "FAQ"
    else:
        passages = retrieve_from_knowledge(user_question, top_k=2)
        synthese = "**Informations pertinentes :**\n\n"
        for (p, sc) in passages:
            synthese += "• " + p.strip() + "\n\n"
        synthese += (
            "---\n\n"
            "⚠️ *Ceci est une information générale. "
            "Pour un avis personnalisé, contactez un professionnel de santé.*"
        )
        final_text = synthese
        source = "knowledge"

    return final_text, source

# ============ STYLE CSS PERSONNALISÉ ============

st.markdown("""
<style>
    /* Style général */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Messages utilisateur */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Messages bot */
    .bot-message {
        background: white;
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Badge source */
    .source-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75em;
        margin-top: 8px;
        font-weight: bold;
    }
    
    .source-faq {
        background: #10b981;
        color: white;
    }
    
    .source-knowledge {
        background: #3b82f6;
        color: white;
    }
    
    /* Boutons de suggestions */
    .suggestion-btn {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        padding: 10px 20px;
        border-radius: 25px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s;
        display: inline-block;
    }
    
    .suggestion-btn:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* En-tête */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }
    
    /* Alerte d'urgence */
    .urgence-box {
        background: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
</style>
""", unsafe_allow_html=True)

# ============ INTERFACE STREAMLIT ============

st.set_page_config(
    page_title="DiabèteBot+", 
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 🩸 DiabèteBot+")
    st.markdown("---")
    
    st.markdown("#### 📊 À propos")
    st.info(
        "Ce chatbot utilise l'intelligence artificielle pour répondre à vos questions "
        "sur le diabète de type 2. Il combine une FAQ précise et une base de connaissances étendue."
    )
    
    st.markdown("#### 🎯 Capacités")
    st.markdown("""
    - ✅ Symptômes et diagnostic
    - ✅ Alimentation adaptée
    - ✅ Activité physique
    - ✅ Traitements et suivi
    - ✅ Complications possibles
    - ✅ Prévention
    """)
    
    st.markdown("---")
    
    st.markdown("#### ⚡ Statistiques")
    st.metric("Questions posées", len([m for m in st.session_state.messages if m["role"] == "user"]))
    st.metric("Base FAQ", len(faq_questions))
    st.metric("Documents", len(knowledge_chunks))
    
    st.markdown("---")
    
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("#### 🚨 Urgences")
    st.markdown("""
    <div class="urgence-box">
    <strong>En cas d'urgence :</strong><br>
    • Malaise grave<br>
    • Confusion importante<br>
    • Douleur thoracique<br>
    • Difficulté respiratoire<br>
    <br>
    <strong>☎️ Appelez le 15 immédiatement</strong>
    </div>
    """, unsafe_allow_html=True)

# ============ EN-TÊTE PRINCIPAL ============
st.markdown("""
<div class="header-container">
    <h1>🩸 DiabèteBot+</h1>
    <p style="font-size: 1.2em; margin-top: 10px;">
        Votre assistant intelligent pour le diabète de type 2
    </p>
</div>
""", unsafe_allow_html=True)

# ============ QUESTIONS SUGGÉRÉES ============
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Questions fréquentes")
    
    suggestions = [
        "Quels sont les symptômes du diabète de type 2 ?",
        "Quelle alimentation adopter ?",
        "Quels sont les risques de complications ?",
        "Comment surveiller ma glycémie ?",
        "Quel rôle joue l'activité physique ?"
    ]
    
    cols = st.columns(2)
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % 2]:
            if st.button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# ============ AFFICHAGE DE L'HISTORIQUE ============
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="clearfix">
                <div class="user-message">
                    <strong>👤 Vous</strong><br>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            source_class = "source-faq" if message.get("source") == "FAQ" else "source-knowledge"
            source_text = "FAQ ✅" if message.get("source") == "FAQ" else "Base de connaissances 📚"
            
            st.markdown(f"""
            <div class="clearfix">
                <div class="bot-message">
                    <strong>🤖 DiabèteBot+</strong><br><br>
                    {message["content"]}
                    <br>
                    <span class="source-badge {source_class}">{source_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============ ZONE DE SAISIE ============
st.markdown("---")

col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Posez votre question ici...",
        key="user_input",
        placeholder="Ex: Comment gérer mon diabète au quotidien ?",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Envoyer", use_container_width=True, type="primary")

# ============ TRAITEMENT DE LA QUESTION ============
if send_button and user_input:
    # Ajouter la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Afficher un spinner pendant le traitement
    with st.spinner("🤔 DiabèteBot+ réfléchit..."):
        time.sleep(0.5)  # Petite pause pour l'effet visuel
        bot_reply, source_used = answer_user(user_input)
    
    # Ajouter la réponse du bot
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply,
        "source": source_used
    })
    
    # Recharger la page pour afficher les nouveaux messages
    st.rerun()

# ============ AVERTISSEMENT MÉDICAL ============
st.markdown("---")
st.warning(
    "⚠️ **Avertissement médical :** Ce chatbot fournit des informations générales à but éducatif. "
    "Il ne remplace en aucun cas une consultation médicale professionnelle. "
    "Pour toute question concernant votre santé, consultez votre médecin ou un professionnel de santé qualifié."
)