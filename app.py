import streamlit as st
from google import genai
from google.genai import types
import time
import os
from dotenv import load_dotenv

# Carica la chiave API dal file .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("ERRORE: La variabile d'ambiente GEMINI_API_KEY non è impostata nel file .env.")
    st.stop()

# --- Funzioni del Client Gemini ---

@st.cache_resource
def get_gemini_client():
    """Inizializza e restituisce il client Gemini."""
    try:
        return genai.Client(api_key=API_KEY)
    except Exception as e:
        st.error(f"Errore nell'inizializzazione del client Gemini: {e}")
        st.stop()

def create_and_upload_files(client, uploaded_files):
    """Crea un File Search Store e carica i file selezionati."""
    
    # 1. Creazione dello Store
    with st.spinner("Creazione del File Search Store..."):
        try:
            # Crea uno store con un nome dinamico
            store = client.file_search_stores.create(config={'display_name': 'Opere Pubbliche RAG App'})
            st.session_state['file_search_store'] = store.name
            st.success(f"Store creato: {store.name}")
        except Exception as e:
            st.error(f"Errore nella creazione dello Store: {e}")
            return False

    # 2. Caricamento e Indicizzazione dei File
    st.info(f"Avvio caricamento e indicizzazione di {len(uploaded_files)} file...")
    
    for file in uploaded_files:
        with st.spinner(f"Indicizzazione di: {file.name}..."):
            try:
                # Scrivi il file temporaneamente su disco (necessario per l'SDK)
                with open(file.name, "wb") as f:
                    f.write(file.getvalue())

                # Carica e importa il file nello Store
                operation = client.file_search_stores.upload_to_file_search_store(
                    file=file.name,
                    file_search_store_name=store.name,
                    config={'display_name': file.name}
                )

                # Attendi il completamento dell'operazione
                while not operation.done:
                    time.sleep(5)
                    operation = client.operations.get(operation)
                
                st.write(f"✅ {file.name} indicizzato con successo.")
                
            except Exception as e:
                st.error(f"Errore durante l'indicizzazione di {file.name}: {e}")
                
            finally:
                # Pulisci il file temporaneo
                if os.path.exists(file.name):
                    os.remove(file.name)
    
    st.success("Tutti i file sono stati indicizzati e il modello è pronto per rispondere!")
    st.session_state['rag_ready'] = True
    return True

# --- Interfaccia Streamlit ---

st.set_page_config(page_title="Agente Esperto Opere Pubbliche (Gemini RAG)")
st.title("🏗️ Agente Esperto Opere Pubbliche con Gemini RAG")
st.caption("Carica i tuoi documenti (Codice Appalti, Capitolati, ecc.) e interrogalo.")

# Inizializza il client e lo stato della sessione
client = get_gemini_client()
if 'rag_ready' not in st.session_state:
    st.session_state['rag_ready'] = False
if 'file_search_store' not in st.session_state:
    st.session_state['file_search_store'] = None


# --- Sidebar per Caricamento Dati ---
with st.sidebar:
    st.header("1. Carica Documenti")
    
    uploaded_files = st.file_uploader(
        "Seleziona i file da indicizzare:",
        type=['pdf', 'docx', 'txt', 'csv'],
        accept_multiple_files=True
    )
    
    if st.button("Indicizza File e Avvia Agente", type="primary") and uploaded_files:
        create_and_upload_files(client, uploaded_files)
        
    st.markdown("---")
    
    # Opzione per eliminare lo store (pulizia)
    if st.session_state['file_search_store']:
        st.header("Pulizia")
        store_to_delete = st.session_state['file_search_store']
        if st.button(f"Elimina Store ({store_to_delete.split('/')[-1]})"):
            try:
                client.file_search_stores.delete(name=store_to_delete, config={'force': True})
                del st.session_state['file_search_store']
                st.session_state['rag_ready'] = False
                st.sidebar.success("Store eliminato con successo. Ricarica la pagina.")
            except Exception as e:
                st.sidebar.error(f"Errore nell'eliminazione dello Store: {e}")


# --- Area Chat Principale ---
if st.session_state['rag_ready']:
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Benvenuto! Sono l'Agente Esperto sulle Opere Pubbliche. Come posso assisterti riguardo ai documenti che hai caricato?"})

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

        # Genera la risposta con la RAG (File Search Tool)
        with st.chat_message("assistant"):
            with st.spinner("Ricerca nei documenti e generazione della risposta..."):
                
                # Configurazione per la RAG
                rag_config = types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[st.session_state['file_search_store']]
                            )
                        )
                    ]
                )
                
                # Chiamata API a Gemini
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=rag_config
                    )
                    
                    st.markdown(response.text)
                    
                    # Estrai e mostra le citazioni
                    if response.candidates and response.candidates[0].grounding_metadata:
                        citations = response.candidates[0].grounding_metadata.retrieval_queries
                        if citations:
                            st.caption(f"**Fonti (Citazioni):** Basato su dati recuperati con le query: {', '.join(citations)}")

                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    
                except Exception as e:
                    st.error(f"Si è verificato un errore durante la generazione della risposta: {e}")

else:
    st.info("Per iniziare, carica i tuoi documenti (PDF, DOCX, ecc.) dalla sidebar e clicca su 'Indicizza File e Avvia Agente'.")
