# Billie Green 🌿

**Tarification intelligente pour une mobilité durable**

Billie Green est une plateforme SaaS de tarification dynamique destinée aux acteurs de la mobilité longue distance. Elle ajuste les prix en fonction de l'impact carbone et du profil socio-économique, tout en garantissant une approche éthique.

## Concept

> *Le prix est un levier de changement de comportement.*

- **Bonus** pour les véhicules à faibles émissions
- **Malus atténué** pour les profils socio-économiques défavorisés
- **Incitation au covoiturage** intégrée

## Architecture

```
├── backend/           # API FastAPI
├── models/            # Modèle PyTorch de tarification
├── frontend/          # Interface utilisateur
└── ADEME-CarLabelling.csv  # Base de données émissions CO2
```

### Modèle IA (PyTorch)

Architecture neuronale personnalisée avec :

- `CarbonImpactEncoder` : Analyse des émissions CO2
- `SocialProfileEncoder` : Profil socio-économique (via ARGUS)
- `EthicalConstraintLayer` : Protection des profils défavorisés
- `EcoSocialAttention` : Pondération automatique des critères

## Inputs

| Paramètre | Description |
|-----------|-------------|
| Modèle voiture | Marque et modèle du véhicule |
| Année | Année du véhicule |
| Valeur ARGUS | Valeur de reprise (proxy socio-économique) |
| Origine/Destination | Villes de départ et d'arrivée |
| Passagers | Nombre de personnes (covoiturage) |
| Jour/Semaine | Jour et période (vacances, été) |

## Output

- Prix personnalisé avec bonus/malus écologique
- Score écologique (0-100%)
- Score social avec protection éthique
- Détail transparent du calcul

## Installation

```bash
# Cloner le repo
git clone https://github.com/votre-username/billie-green.git
cd billie-green

python3.9 -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

OPENROUTE_API_KEY="Votre clé"

python3 -m uvicorn backend.main:app --reload

open frontend/index.html

```

Puis ouvrir `frontend/index.html` dans un navigateur.

## API Endpoints

- `POST /api/pricing/calculate` - Calcul de tarification
- `POST /api/vehicle/search` - Recherche véhicule ADEME
- `POST /api/trip/calculate` - Calcul de trajet
- `GET /api/brands` - Liste des marques
- `GET /api/cities` - Villes disponibles

## Données

- **ADEME Car Labelling** : 3600+ véhicules avec émissions CO2
- **Étude Symone** : Comportements de mobilité Paris-Lyon-Méditerranée

## Principe éthique

```python
# Le malus écologique est atténué proportionnellement
# au niveau socio-économique défavorable
if eco_score < 0.3 and social_score < 0.4:
    malus *= ethical_protection  # Réduction du malus
```

## Licence

MIT

---

*Billie Green - Faire du prix un moteur de la transition écologique*
