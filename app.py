import streamlit as st
import os
import json
import tempfile
from dotenv import load_dotenv

# --- Nuove Importazioni per LlamaIndex e Google Drive ---
# NOTA: Queste importazioni richiedono che i pacchetti siano installati (vedi requirements.txt)
try:
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google import GoogleEmbedding
    from llama_index.readers.google import GoogleDriveReader
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.agent import ReActAgent
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.vector_stores.chroma import ChromaVectorStore # Usiamo ChromaDB come store locale
    import chromadb
except ImportError:
    st.error("ERRORE: Le librerie LlamaIndex o ChromaDB non sono installate. Esegui 'pip install -r requirements.txt'")
    st.stop()


# Carica la chiave API dal file .env (o dalla variabile d'ambiente di Codespace)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("ERRORE: La variabile d'ambiente GEMINI_API_KEY non è impostata.")
    st.stop()

# --- Configurazione LlamaIndex/Gemini ---

# Impostiamo il client LLM e l'Embedding Model per LlamaIndex
Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=API_KEY)
Settings.embed_model = GoogleEmbedding(model_name="models/text-embedding-004", api_key=API_KEY) 

# ID della cartella RAG OOPP (inserisci l'ID qui)
# L'ID si trova nell'URL di Drive dopo /folders/
FOLDER_ID = "IL_TUO_ID_CARTELLA_DRIVE_QUI" # <--- INSERISCI QUI L'ID DELLA TUA CARTELLA

# --- Funzioni del Client Gemini (Rimosso File Search Tool, introdotto LlamaIndex) ---

@st.cache_resource
def load_and_index_drive_files(folder_id):
    """
    Carica i file da Google Drive tramite LlamaIndex e Secrets di GitHub,
    crea un indice vettoriale e lo salva localmente (ChromaDB).
    """
    st.info("Avvio connessione sicura a Google Drive...")
    
    # 1. Recupero delle Credenziali dal Segreto di GitHub
    credentials_json_content = os.getenv("GOOGLE_CREDENTIALS_JSON")
    
    if not credentials_json_content:
        st.error("ERRORE: La variabile 'GOOGLE_CREDENTIALS_JSON' non è stata trovata come GitHub Secret.")
        return None

    # 2. Creazione del File Temporaneo per l'Autenticazione OAuth
    tmp_cred_path = None
    try:
        # Usa tempfile per creare un file temporaneo che verrà eliminato alla fine
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_cred_file:
            # Scrivi il contenuto del Segreto nel file temporaneo
            tmp_cred_file.write(credentials_json_content)
            tmp_cred_path = tmp_cred_file.name

        st.info("Autenticazione in corso. Ti verrà chiesto di autorizzare l'accesso a Google Drive la prima volta.")

        # 3. Inizializza il Drive Reader usando il file temporaneo
        loader = GoogleDriveReader(
            is_folder=True, 
            folder_id=folder_id,
            credentials_path=tmp_cred_path
        )
        
        # 4. Carica i documenti (qui avviene la richiesta di autorizzazione)
        documents = loader.load_data()
        st.success(f"Trovati {len(documents)} documenti su Google Drive. Avvio indicizzazione...")
        
        if not documents:
            st.warning("Nessun documento trovato nella cartella specificata. Controlla l'ID.")
            return None

        # 5. Creazione dell'Indice Vettoriale (ChromaDB)
        with st.spinner("Creazione e salvataggio dell'indice (embedding) dei documenti..."):
            
            # Inizializza Chroma DB in memoria/locale
            db = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = db.get_or_create_collection("opere_pubbliche_collection")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Crea l'indice
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context
            )
            
            st.success("Indicizzazione completata e salvata localmente. Agente pronto!")
            
        return index

    except Exception as e:
        st.error(f"Errore critico durante l'accesso a Google Drive/l'indicizzazione: {e}")
        return None
        
    finally:
        # 6. Pulisci: elimina il file temporaneo in ogni caso
        if tmp_cred_path and os.path.exists(tmp_cred_path):
            os.remove(tmp_cred_path)
            st.info("File temporaneo di credenziali eliminato.")


def cleanup_local_db():
    """Rimuove la directory di ChromaDB salvata localmente."""
    import shutil
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        return True
    return False

# --- Interfaccia Streamlit ---

st.set_page_config(page_title="Agente Esperto Opere Pubbliche (Gemini RAG)")
st.title("🏗️ Agente Esperto Opere Pubbliche con Gemini RAG (Drive)")
st.caption("Sincronizza i documenti dalla cartella 'RAG OOPP' del tuo Google Drive.")

# Inizializza lo stato della sessione
if 'rag_ready' not in st.session_state:
    st.session_state['rag_ready'] = False
if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = None

# --- Sidebar per Sincronizzazione Dati ---
with st.sidebar:
    st.header("1. Sincronizzazione Google Drive")
    
    if FOLDER_ID == "IL_TUO_ID_CARTELLA_DRIVE_QUI":
        st.error("Inserisci l'ID della tua cartella Drive nel codice sorgente e riavvia l'app.")
    else:
        
        if st.button("Sincronizza e Avvia da Drive", type="primary"):
            # Resetta lo stato se si sincronizza di nuovo
            st.session_state['rag_ready'] = False
            st.session_state['query_engine'] = None 

            index = load_and_index_drive_files(FOLDER_ID)
            
            if index:
                # Crea un Query Engine ottimizzato per la ricerca
                st.session_state['query_engine'] = index.as_query_engine(
                    similarity_top_k=5 # Recupera i 5 chunk più rilevanti
                )
                st.session_state['rag_ready'] = True
                st.rerun()
                
    st.markdown("---")
    
    # Opzione per eliminare lo store locale (pulizia)
    st.header("Pulizia Locale")
    if st.button("Elimina Indice Locale (ChromaDB)"):
        if cleanup_local_db():
            del st.session_state['query_engine']
            st.session_state['rag_ready'] = False
            st.sidebar.success("Indice locale eliminato con successo. Ricarica la pagina.")
        else:
            st.sidebar.info("Nessun indice locale da eliminare.")


# --- Area Chat Principale ---
if st.session_state['rag_ready'] and st.session_state['query_engine']:
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Benvenuto! Sono l'Agente Esperto sulle Opere Pubbliche, basato sui documenti di Google Drive. Come posso assisterti?"})

    # Visualizza i messaggi precedenti
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Gestione dell'input dell'utente
    if prompt := st.chat_input("Fai la tua domanda..."):
        
        # Aggiungi il messaggio utente alla chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Genera la risposta con la RAG (LlamaIndex)
        with st.chat_message("assistant"):
            with st.spinner("Ricerca nei documenti (Google Drive) e generazione della risposta..."):
                
                try:
                    # Chiamata al Query Engine di LlamaIndex
                    response = st.session_state['query_engine'].query(prompt)
                    
                    st.markdown(str(response))
                    
                    # Estrai e mostra le fonti/citazioni di LlamaIndex
                    if response.source_nodes:
                        source_names = ", ".join([node.metadata.get('file_name', 'N/A') for node in response.source_nodes])
                        st.caption(f"**Fonti Trovate:** Dati da: {source_names}")

                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                    
                except Exception as e:
                    st.error(f"Si è verificato un errore durante la generazione della risposta: {e}")

else:
    st.info("Per iniziare, inserisci l'ID della tua cartella Drive nel codice, assicurati di avere il Segreto 'GOOGLE_CREDENTIALS_JSON' e clicca su 'Sincronizza e Avvia da Drive' nella sidebar.")
