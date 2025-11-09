import streamlit as st
import os
import json
import tempfile
from dotenv import load_dotenv


st.title("🔐 Test Secrets")
st.write("✅ OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))
st.write("✅ GOOGLE_CREDENTIALS_JSON:", bool(os.getenv("GOOGLE_CREDENTIALS_JSON")))

# --- Import principali per LlamaIndex e Google Drive ---
try:
    from llama_index.readers.google import GoogleDriveReader
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
except ImportError:
    st.error("❌ Le librerie LlamaIndex o ChromaDB non sono installate. Esegui 'pip install -r requirements.txt'.")
    st.stop()

# --- Carica variabili d'ambiente ---
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CREDS = os.getenv("GOOGLE_CREDENTIALS_JSON")

# --- Configurazione dinamica LLM ---
try:
    if OPENAI_KEY:
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY)
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.success("✅ Connessione stabilita con GPT-4o (OpenAI API).")

    else:
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.llm = Ollama(model="mistral")  # Modello gratuito
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.warning("⚙️ Nessuna chiave OpenAI trovata — uso modello locale gratuito (Ollama/Mistral).")

except Exception as e:
    st.error(f"Errore durante la configurazione del modello: {e}")
    st.stop()

# --- ID cartella Google Drive ---
FOLDER_ID = "1-TRQN_5hpCYkI7P1wmDAUCfBdKQ-Dk16"  # <-- Inserisci qui il tuo ID cartella Drive

# --- Funzione: carica e indicizza file Drive ---
@st.cache_resource
def load_and_index_drive_files(folder_id):
    st.info("🔐 Connessione a Google Drive in corso...")

    if not GOOGLE_CREDS:
        st.error("ERRORE: variabile 'GOOGLE_CREDENTIALS_JSON' non trovata nei Secrets.")
        return None

    tmp_cred_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_cred_file:
            tmp_cred_file.write(GOOGLE_CREDS)
            tmp_cred_path = tmp_cred_file.name

        loader = GoogleDriveReader(
            is_folder=True,
            folder_id=folder_id,
            credentials_path=tmp_cred_path
        )

        documents = loader.load_data()
        if not documents:
            st.warning("⚠️ Nessun documento trovato nella cartella indicata.")
            return None

        st.success(f"Trovati {len(documents)} documenti. Avvio indicizzazione...")

        with st.spinner("Creazione indice vettoriale (ChromaDB)..."):
            db = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = db.get_or_create_collection("opere_pubbliche_collection")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        st.success("✅ Indicizzazione completata e salvata localmente.")
        return index

    except Exception as e:
        st.error(f"Errore durante l'accesso o l'indicizzazione: {e}")
        return None

    finally:
        if tmp_cred_path and os.path.exists(tmp_cred_path):
            os.remove(tmp_cred_path)

# --- Pulizia database locale ---
def cleanup_local_db():
    import shutil
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        return True
    return False

# --- Interfaccia Streamlit ---
st.set_page_config(page_title="Agente Opere Pubbliche RAG", page_icon="🏗️")
st.title("🏗️ Agente Esperto Opere Pubbliche – RAG (Drive)")
st.caption("Sincronizza e interroga i documenti della cartella Google Drive.")

# Stato sessione
if "rag_ready" not in st.session_state:
    st.session_state["rag_ready"] = False
if "query_engine" not in st.session_state:
    st.session_state["query_engine"] = None

# --- Sidebar ---
with st.sidebar:
    st.header("1️⃣ Sincronizzazione Google Drive")

    if FOLDER_ID == "IL_TUO_ID_CARTELLA_DRIVE_QUI":
        st.error("⚠️ Inserisci l’ID corretto della cartella Drive e riavvia.")
    else:
        if st.button("🔄 Sincronizza e Avvia", type="primary"):
            st.session_state["rag_ready"] = False
            st.session_state["query_engine"] = None
            index = load_and_index_drive_files(FOLDER_ID)
            if index:
                st.session_state["query_engine"] = index.as_query_engine(similarity_top_k=5)
                st.session_state["rag_ready"] = True
                st.rerun()

    st.markdown("---")
    st.header("🧹 Pulizia Locale")
    if st.button("Elimina Indice Locale (ChromaDB)"):
        if cleanup_local_db():
            st.session_state["query_engine"] = None
            st.session_state["rag_ready"] = False
            st.success("🗑️ Indice locale eliminato — ricarica la pagina.")
        else:
            st.info("Nessun indice locale trovato.")
   
    # --- Diagnostica Secrets ---
    st.markdown("---")
    st.header("🧪 Verifica Secrets (solo diagnostica interna)")
    if st.button("Verifica variabili ambiente"):
        secrets = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_CREDENTIALS_JSON": os.getenv("GOOGLE_CREDENTIALS_JSON"),
        }

        for name, value in secrets.items():
            if value:
                st.success(f"✅ {name}: trovata ({len(value)} caratteri, nascosta per sicurezza)")
            else:
                st.error(f"❌ {name}: non trovata")

        st.caption("Le variabili vengono lette dai Secrets di Codespaces o Streamlit Cloud.")

# --- Area Chat ---
if st.session_state["rag_ready"] and st.session_state["query_engine"]:
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Benvenuto! Sono l’Agente Esperto Opere Pubbliche, basato sui documenti di Google Drive. Come posso aiutarti?"
        }]

    # Mostra conversazione
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input utente
    if prompt := st.chat_input("Fai una domanda sui documenti..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Genera risposta
        with st.chat_message("assistant"):
            with st.spinner("Analizzo i documenti e genero la risposta..."):
                try:
                    response = st.session_state["query_engine"].query(prompt)
                    st.markdown(str(response))

                    # Fonti
                    if getattr(response, "source_nodes", None):
                        sources = ", ".join([
                            n.metadata.get("file_name", "Documento sconosciuto") for n in response.source_nodes
                        ])
                        st.caption(f"**Fonti:** {sources}")

                    st.session_state.messages.append({"role": "assistant", "content": str(response)})

                except Exception as e:
                    st.error(f"Errore durante la risposta: {e}")
else:
    st.info("🔐 Inserisci l’ID della cartella Drive e clicca su 'Sincronizza e Avvia' per iniziare.")
