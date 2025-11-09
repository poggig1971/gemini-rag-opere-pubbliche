import os
from dotenv import load_dotenv

# Carica eventuale .env locale (non serve se usi Secrets)
load_dotenv()

# Elenco delle variabili da verificare
vars_to_check = [
    "OPENAI_API_KEY",
    "GOOGLE_CREDENTIALS_JSON"
]

print("🔍 Verifica variabili d'ambiente:\n")

for var in vars_to_check:
    value = os.getenv(var)
    if value:
        if len(value) > 50:
            print(f"✅ {var}: trovata ({len(value)} caratteri, non mostrata per sicurezza)")
        else:
            print(f"✅ {var}: trovata → {value}")
    else:
        print(f"❌ {var}: non trovata")

print("\n✅ Test completato.")
