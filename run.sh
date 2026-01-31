#!/bin/bash

# Billie Green - Script de lancement

echo "🌿 Billie Green - Démarrage..."

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

# Créer l'environnement virtuel si nécessaire
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement
source venv/bin/activate

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip install -q -r backend/requirements.txt

# Copier les données si nécessaire
if [ ! -f "data/ADEME-CarLabelling.csv" ]; then
    mkdir -p data
    if [ -f "ADEME-CarLabelling.csv" ]; then
        cp ADEME-CarLabelling.csv data/
    fi
fi

# Lancer le serveur
echo ""
echo "🚀 Démarrage du serveur..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Frontend: ouvrir frontend/index.html dans un navigateur"
echo ""

cd "$(dirname "$0")"
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
