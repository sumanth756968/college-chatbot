import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------- CONFIG ----------------
MODEL_NAME = "all-MiniLM-L6-v2"
FAQ_INDEX_PATH = "faq_index.faiss"
FAQ_META_PATH = "faq_meta.pkl"
SITE_INDEX_PATH = "site_index.faiss"
SITE_META_PATH = "site_meta.pkl"

FAQ_THRESHOLD = 0.70
SITE_THRESHOLD = 0.35

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_all():
    embedder = SentenceTransformer(MODEL_NAME)
    faq_index = faiss.read_index(FAQ_INDEX_PATH)
    with open(FAQ_META_PATH, "rb") as f:
        faq_meta = pickle.load(f)

    site_index = faiss.read_index(SITE_INDEX_PATH)
    with open(SITE_META_PATH, "rb") as f:
        site_meta = pickle.load(f)

    generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    return embedder, faq_index, faq_meta, site_index, site_meta, generator

embedder, faq_index, faq_meta, site_index, site_meta, generator = load_all()

# ---------------- HELPERS ----------------
def l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms

def retrieve_context(query, top_k=3):
    q = query.strip().lower()
    q_emb = embedder.encode([q], convert_to_numpy=True)
    q_emb = l2_normalize(q_emb.astype("float32"))

    # FAQ search
    D, I = faq_index.search(q_emb, top_k)
    if I[0][0] != -1 and D[0][0] >= FAQ_THRESHOLD:
        best_idx = I[0][0]
        return {
            "source": "faq",
            "context": faq_meta["answers"][best_idx],
            "meta": {
                "question": faq_meta["questions"][best_idx],
                "intent": faq_meta["intents"][best_idx],
                "score": float(D[0][0])
            }
        }

    # Site search
    D, I = site_index.search(q_emb, top_k)
    if I[0][0] != -1 and D[0][0] >= SITE_THRESHOLD:
        best_idx = I[0][0]
        chunk = site_meta["chunks"][best_idx]
        return {
            "source": "site",
            "context": chunk["text"],
            "meta": {"url": chunk["url"], "score": float(D[0][0])}
        }

    return {"source": "none", "context": None, "meta": {}}

def generate_answer(user_query, context):
    if not context:
        return "Sorry, I couldn’t find relevant information."
    prompt = f"Answer the question below using the given context.\n\nQuestion: {user_query}\nContext: {context}\nAnswer:"
    result = generator(prompt, max_length=200, temperature=0.4)
    return result[0]['generated_text']

# ---------------- STREAMLIT UI ----------------
st.title("🎓 College FAQ + Website Chatbot")
st.write("Ask me anything about the college. I’ll first check FAQ, then the website.")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.text_input("Your Question:")

if user_query:
    result = retrieve_context(user_query)
    answer = generate_answer(user_query, result["context"])
    st.session_state.history.append(("You", user_query))
    st.session_state.history.append(("Bot", answer))

for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
