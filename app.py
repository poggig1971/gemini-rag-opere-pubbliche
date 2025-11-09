import streamlit as st
import os
import json
import tempfile
from dotenv import load_dotenv
import chromadb

# --- Config Streamlit ---
st.set_page_config(page_title="🏗️ Agente Opere Pubbliche RAG", page_icon="🏗️")
st.title("🏗️ Agente Esperto Opere Pubbliche – RAG (Drive)")
st.caption("Sincronizza e interroga i documenti della cartella Google Drive.")
load_dotenv()

# --- Variabili ambiente ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS = os.getenv("GOOGLE_CREDENTIALS_JSON")
FOLDER_ID = "1-TRQN_5hpCYkI7P1wmDAUCfBdKQ-Dk16"  # <-- il tuo ID cartella

st.markdown("### 🔐 Verifica chiavi API")
st.write("✅ OPENAI_API_KEY:", bool(OPENAI_KEY))
st.write("✅ GOOGLE_CREDENTIALS_JSON:", bool(GOOGLE_CREDS))

# --- Import LlamaIndex (nuova sintassi >= 0.11.x) ---
try:
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.readers.google_drive import GoogleDriveReader
except Exception as e:
    st.error(f"❌ Errore di import: {e}")
    st.stop()

# --- Setup LLM ed embedding ---
llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Funzione: carica e indicizza file da Google Drive ---
@st.cache_resource
def load_and_index_drive_files(folder_id):
    if not GOOGLE_CREDS:
        st.error("❌ Variabile 'GOOGLE_CREDENTIALS_JSON' mancante nei secrets.")
        return None

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        tmp.write(GOOGLE_CREDS)
        cred_path = tmp.name

    try:
        st.info("🔐 Connessione a Google Drive...")
        loader = GoogleDriveReader(folder_id=folder_id, service_account_key=cred_path)
        documents = loader.load_data()

        if not documents:
            st.warning("⚠️ Nessun documento trovato nella cartella specificata.")
            return None

        st.success(f"Trovati {len(documents)} documenti. Avvio indicizzazione...")

        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("opere_pubbliche_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            storage_context=storage_context,
        )

        return index

    except Exception as e:
        st.error(f"Errore durante l'accesso o l'indicizzazione: {e}")
        return None
    finally:
        os.remove(cred_path)

# --- Pulizia database locale ---
def cleanup_local_db():
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        return True
    return False

# --- Sidebar ---
with st.sidebar:
    st.header("1️⃣ Sincronizzazione Google Drive")
    if st.button("🔄 Sincronizza e Avvia", type="primary"):
        st.session_state["rag_ready"] = False
        index = load_and_index_drive_files(FOLDER_ID)
        if index:
            st.session_state["index"] = index
            st.session_state["query_engine"] = index.as_query_engine(similarity_top_k=5)
            st.session_state["rag_ready"] = True
            st.rerun()

    st.markdown("---")
    st.header("🧹 Pulizia Locale")
    if st.button("Elimina Indice Locale"):
        if cleanup_local_db():
            st.session_state["rag_ready"] = False
            st.success("🗑️ Indice locale eliminato.")
        else:
            st.info("Nessun indice locale trovato.")

# --- Area Chat ---
if st.session_state.get("rag_ready"):
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Benvenuto! Sono l’Agente Esperto Opere Pubbliche, basato sui documenti Google Drive."
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Fai una domanda sui documenti..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizzo i documenti..."):
                try:
                    response = st.session_state["query_engine"].query(prompt)
                    st.markdown(str(response))
                except Exception as e:
                    st.error(f"Errore: {e}")
