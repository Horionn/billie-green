# 🚛 Module Calculateur Camion Symone - Biogaz

## 📖 Description

Ce module calcule automatiquement le **coût total** d'un trajet en camion Symone fonctionnant au **biogaz**, incluant :
- ✅ **Péages autoroutiers** (tarifs classe 4 - poids lourds)
- ✅ **Carburant biogaz** (production interne Symone)
- ✅ **Émissions CO2** (quasi nulles grâce au biogaz renouvelable)
- ✅ **Comparaison avec diesel** (économies et impact environnemental)

---

## 🎯 Caractéristiques du camion Symone

| Paramètre | Valeur | Notes |
|-----------|--------|-------|
| **Carburant** | Biogaz | Produit en interne par Symone |
| **Consommation** | 25 kg/100 km | Moyenne pour poids lourd |
| **Prix biogaz** | 0.85 €/kg | Compétitif vs diesel (~1.60€/L) |
| **Émissions CO2** | 15 g/km | Quasi neutre (vs 600-800 g/km diesel) |
| **Classe péage** | 4 | Poids lourds > 3.5 tonnes |

---

## 💰 Exemple de calcul : Paris → Lyon (462 km)

```
📍 Trajet: Paris → Lyon
📏 Distance: 462 km

💰 COÛTS:
   • Péage autoroute:    62.80 €
   • Carburant biogaz:   98.18 € (115.5 kg × 0.85 €/kg)
   ────────────────────────────
   • TOTAL:             160.98 €  (0.348 €/km)

🌱 ENVIRONNEMENT:
   • CO2 émis: 6.93 kg (vs ~277 kg pour diesel)
   • Réduction: -97.5% d'émissions

📊 COMPARAISON DIESEL:
   • Coût équivalent diesel: 138.60 €
   • Différence: +22.38 € (+16%)
   
   ⚠️ Note: Le biogaz est légèrement plus cher MAIS:
      - Production locale (pas de dépendance pétrole)
      - Impact environnemental quasi nul
      - Valorisation des déchets organiques
```

---

## 🔌 Intégration dans l'API Billie Green

### Nouveaux endpoints ajoutés

#### 1. **GET `/api/truck/specs`**
Retourne les spécifications du camion Symone

**Réponse :**
```json
{
  "name": "Camion Symone Biogaz",
  "fuel_type": "BIOGAZ",
  "consumption_per_100km": 25.0,
  "biogaz_price_per_kg": 0.85,
  "co2_g_km": 15.0,
  "toll_class": 4
}
```

#### 2. **POST `/api/truck/calculate`**
Calcule le coût d'un trajet

**Requête :**
```json
{
  "origin": "Paris",
  "destination": "Lyon",
  "distance_km": null,  // Optionnel, calculé auto si null
  "custom_consumption": 22.0,  // Optionnel
  "custom_biogaz_price": 0.75   // Optionnel
}
```

**Réponse complète :**
```json
{
  "truck_info": {
    "name": "Camion Symone Biogaz",
    "fuel_type": "BIOGAZ",
    "co2_g_km": 15.0
  },
  "trip": {
    "origin": "Paris",
    "destination": "Lyon",
    "distance_km": 462
  },
  "toll": {
    "price_euros": 62.8,
    "is_exact": true,
    "description": "Tarif réel classe 4"
  },
  "fuel": {
    "fuel_consumption_kg": 115.5,
    "fuel_price_per_kg": 0.85,
    "fuel_cost_euros": 98.18,
    "consumption_per_100km": 25.0
  },
  "total_cost_euros": 160.98,
  "cost_per_km_euros": 0.348,
  "environmental": {
    "co2_total_kg": 6.93,
    "co2_per_km_g": 15.0,
    "fuel_type": "Biogaz (renouvelable)"
  },
  "comparison": {
    "diesel_equivalent_cost": 138.6,
    "savings_vs_diesel_euros": -22.38,
    "savings_percent": -16.1
  },
  "breakdown": [
    {"item": "Péage autoroute", "amount": 62.8, "unit": "€"},
    {"item": "Carburant biogaz (115.5 kg)", "amount": 98.18, "unit": "€"},
    {"item": "TOTAL", "amount": 160.98, "unit": "€"}
  ]
}
```

#### 3. **GET `/api/truck/toll-prices`**
Liste tous les tarifs de péages disponibles (classe 4)

**Réponse :**
```json
{
  "toll_class": 4,
  "description": "Tarifs péages classe 4 (poids lourds > 3.5t)",
  "count": 45,
  "prices": [
    {
      "route": "Paris → Lyon",
      "origin": "paris",
      "destination": "lyon",
      "price_euros": 62.8
    },
    // ... autres trajets
  ]
}
```

#### 4. **POST `/api/truck/compare-with-car`**
Compare le coût camion vs voiture particulière

**Requête :**
```json
{
  "trip": {
    "origin": "Paris",
    "destination": "Lyon"
  },
  "vehicle": {
    "brand": "RENAULT",
    "model": "CLIO",
    "energy": "ESSENCE"
  },
  "passengers": 4
}
```

---

## 🗺️ Trajets disponibles (tarifs péages exacts)

Le système contient **45+ trajets** avec tarifs péages réels. Exemples :

| Trajet | Distance | Péage | Carburant | Total |
|--------|----------|-------|-----------|-------|
| Paris → Lyon | 462 km | 62.80€ | 98.18€ | 160.98€ |
| Paris → Marseille | 774 km | 105.40€ | 164.47€ | 269.87€ |
| Lyon → Marseille | 314 km | 42.90€ | 66.73€ | 109.63€ |
| Lyon → Nice | 472 km | 64.60€ | 100.30€ | 164.90€ |
| Grenoble → Nice | 336 km | 45.90€ | 71.40€ | 117.30€ |

Pour les trajets non listés, le système **estime automatiquement** le péage à 0.136 €/km.

---

## 🌍 Impact environnemental

### Comparaison Biogaz vs Diesel

| Critère | Diesel | Biogaz Symone | Gain |
|---------|--------|---------------|------|
| **CO2/km** | 600-800 g | 15 g | **-97.5%** |
| **Origine** | Fossile | Renouvelable | ♻️ |
| **Dépendance** | Pétrole importé | Production locale | 🇫🇷 |
| **Déchets** | Pollution | Valorisation bio | ✅ |

**Exemple Paris-Lyon :**
- **Diesel:** ~277 kg CO2
- **Biogaz:** ~7 kg CO2
- **Économie:** 270 kg CO2 par trajet !

---

## 🔧 Installation et utilisation

### 1. Installation

```bash
# Copier le fichier dans votre backend
cp symone_truck_calculator.py backend/

# Installer les dépendances (déjà présentes)
# httpx, fastapi, pydantic déjà installés dans votre projet
```

### 2. Intégration dans `main.py`

Suivre le guide dans `INTEGRATION_GUIDE.md`

### 3. Tests

```bash
# Test du module seul
python backend/symone_truck_calculator.py

# Test de l'API (après intégration)
python test_truck_api.py
```

---

## 📊 Cas d'usage

### 1. **Devis automatique pour clients**
```python
# Calculer le prix d'un transport Paris → Marseille
result = calculator.calculate_trip_cost("Paris", "Marseille", 774)
prix_client = result["total_cost_euros"] * 1.2  # Marge 20%
```

### 2. **Optimisation de la flotte**
```python
# Comparer plusieurs routes pour choisir la plus économique
routes = [("Paris", "Lyon"), ("Lyon", "Marseille"), ("Marseille", "Nice")]
for origin, dest in routes:
    cost = calculator.calculate_trip_cost(origin, dest, distance)
    # Choisir la route la moins chère
```

### 3. **Reporting environnemental**
```python
# Calculer l'impact CO2 mensuel de la flotte
total_km = 50000  # km/mois
co2_monthly = (total_km * 15) / 1000  # kg
# Comparer avec équivalent diesel
co2_saved = (total_km * 700) / 1000 - co2_monthly
```

### 4. **Arguments commerciaux**
```python
# Montrer les économies CO2 au client
result = calculator.calculate_trip_cost("Paris", "Lyon", 462)
print(f"Économie CO2: {270 - result['environmental']['co2_total_kg']} kg")
print("Équivalent à planter X arbres")
```

---

## ⚙️ Paramètres ajustables

### Modifier la consommation

```python
# Dans symone_truck_calculator.py, ligne 16
consumption_per_100km: float = 22.0  # Exemple: camion optimisé
```

### Modifier le prix du biogaz

```python
# Dans symone_truck_calculator.py, ligne 18
biogaz_price_per_kg: float = 0.75  # Exemple: baisse des coûts
```

### Ajouter des trajets péages

```python
# Dans symone_truck_calculator.py, dictionnaire TRUCK_TOLL_PRICES
("toulouse", "bordeaux"): 38.50,
("nantes", "paris"): 52.30,
```

---

## 🚀 Évolutions possibles

### Court terme
- [ ] Import automatique des tarifs péages (API Sanef/APRR)
- [ ] Calcul multi-étapes avec arrêts
- [ ] Export PDF des devis

### Moyen terme
- [ ] Prise en compte du poids de charge (impact consommation)
- [ ] Optimisation des routes (économies)
- [ ] Dashboard analytics (coûts, CO2, économies)

### Long terme
- [ ] Prédiction de trafic (horaires optimaux)
- [ ] API de tarification dynamique biogaz
- [ ] Intégration avec système de réservation

---

## 📞 Support

Pour toute question technique :
1. Consultez `INTEGRATION_GUIDE.md`
2. Lancez les tests : `python test_truck_api.py`
3. Vérifiez les logs API : `uvicorn backend.main:app --reload`

---

## 📄 Licence

Module développé pour **Symone** - Billie Green
© 2025 - Tous droits réservés

---

## 🙏 Crédits

- **Données péages** : Tarifs 2024-2025 des sociétés concessionnaires (Sanef, APRR, Vinci)
- **Consommation biogaz** : Données constructeurs poids lourds
- **Émissions CO2** : Études ADEME sur le biogaz carburant
