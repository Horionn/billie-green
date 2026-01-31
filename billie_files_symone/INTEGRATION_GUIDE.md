# Guide d'Intégration - Calculateur Camion Symone

## 📋 Vue d'ensemble

Ce module ajoute le calcul automatique du coût (péage + carburant biogaz) pour les camions Symone à votre API Billie Green existante.

## 🚀 Installation

### Étape 1 : Copier les fichiers

Copiez ces fichiers dans votre projet :
- `symone_truck_calculator.py` → `/backend/symone_truck_calculator.py`

### Étape 2 : Modifier `main.py`

Ajoutez les imports en haut du fichier :

```python
from backend.symone_truck_calculator import (
    SymoneTruckCalculator, 
    TruckSpecs, 
    calculate_symone_truck_cost,
    TRUCK_TOLL_PRICES
)
```

### Étape 3 : Ajouter les modèles Pydantic

Après vos modèles existants (`VehicleInput`, `TripInput`, etc.), ajoutez :

```python
class TruckTripInput(BaseModel):
    """Input pour calcul de coût camion."""
    origin: str = Field(..., description="Ville de départ", example="Paris")
    destination: str = Field(..., description="Ville d'arrivée", example="Lyon")
    distance_km: Optional[float] = Field(None, description="Distance en km (auto si None)")
    custom_consumption: Optional[float] = Field(None, description="Consommation (kg/100km)", example=25.0)
    custom_biogaz_price: Optional[float] = Field(None, description="Prix biogaz (€/kg)", example=0.85)


class TruckCostResponse(BaseModel):
    """Réponse du calcul de coût camion."""
    truck_info: Dict
    trip: Dict
    toll: Dict
    fuel: Dict
    total_cost_euros: float
    cost_per_km_euros: float
    environmental: Dict
    comparison: Dict
    breakdown: List[Dict]
```

### Étape 4 : Initialiser le calculateur

Après l'initialisation de `trip_calculator` et `vehicle_db`, ajoutez :

```python
# Calculateur pour camions Symone
truck_calculator = SymoneTruckCalculator()
```

### Étape 5 : Ajouter les endpoints

À la fin de votre fichier `main.py`, avant le `if __name__ == "__main__":`, ajoutez :

```python
# ============================================
# ENDPOINTS CAMION SYMONE
# ============================================

@app.get("/api/truck/specs")
async def get_truck_specs():
    """Retourne les spécifications du camion Symone."""
    specs = TruckSpecs()
    return {
        "name": specs.name,
        "fuel_type": specs.fuel_type,
        "consumption_per_100km": specs.consumption_per_100km,
        "biogaz_price_per_kg": specs.biogaz_price_per_kg,
        "co2_g_km": specs.co2_g_km,
        "toll_class": specs.toll_class
    }


@app.post("/api/truck/calculate", response_model=TruckCostResponse)
async def calculate_truck_cost(trip: TruckTripInput):
    """
    Calcule le coût total (péage + carburant) pour un camion Symone.
    """
    # Si distance non fournie, calculer automatiquement
    if trip.distance_km is None:
        distance_info = await calculate_trip(TripInput(
            origin=trip.origin,
            destination=trip.destination
        ))
        distance_km = distance_info["distance_km"]
    else:
        distance_km = trip.distance_km
    
    # Calculer le coût
    result = calculate_symone_truck_cost(
        origin=trip.origin,
        destination=trip.destination,
        distance_km=distance_km,
        custom_consumption=trip.custom_consumption,
        custom_biogaz_price=trip.custom_biogaz_price
    )
    
    return TruckCostResponse(**result)


@app.get("/api/truck/toll-prices")
async def list_truck_toll_prices():
    """Liste tous les prix de péages disponibles pour poids lourds."""
    return {
        "toll_class": 4,
        "description": "Tarifs péages classe 4 (poids lourds > 3.5t)",
        "count": len(TRUCK_TOLL_PRICES),
        "prices": [
            {
                "route": f"{origin.capitalize()} → {destination.capitalize()}",
                "origin": origin,
                "destination": destination,
                "price_euros": price
            }
            for (origin, destination), price in sorted(TRUCK_TOLL_PRICES.items())
        ]
    }


@app.post("/api/truck/compare-with-car")
async def compare_truck_with_car(
    trip: TruckTripInput,
    vehicle: VehicleInput,
    passengers: int = Field(1, ge=1, le=8)
):
    """
    Compare le coût camion Symone vs voiture particulière.
    """
    # 1. Coût camion
    truck_result = await calculate_truck_cost(trip)
    
    # 2. Coût voiture
    car_request = PricingRequest(
        vehicle=vehicle,
        trip=TripInput(
            origin=trip.origin,
            destination=trip.destination,
            day_of_week=0,
            hour=10,
            is_holiday=False,
            is_summer=False
        ),
        passengers=passengers
    )
    car_result = await calculate_pricing(car_request)
    
    # 3. Comparaison
    total_car_cost = car_result.final_price * passengers
    
    return {
        "truck": {
            "total_cost": truck_result.total_cost_euros,
            "cost_per_km": truck_result.cost_per_km_euros,
            "co2_kg": truck_result.environmental["co2_total_kg"],
            "fuel_type": "Biogaz (renouvelable)"
        },
        "car": {
            "price_per_passenger": car_result.final_price,
            "total_cost": total_car_cost,
            "co2_category": car_result.co2_category,
            "fuel_type": car_result.vehicle_info.get("energie", "Unknown")
        },
        "comparison": {
            "truck_cheaper": truck_result.total_cost_euros < total_car_cost,
            "cost_difference_euros": round(truck_result.total_cost_euros - total_car_cost, 2),
            "truck_vs_car_ratio": round(truck_result.total_cost_euros / total_car_cost, 2) if total_car_cost > 0 else 0,
            "environmental_winner": "Truck (Biogaz)" if truck_result.environmental["co2_total_kg"] < 100 else "Depends"
        }
    }
```

## 📊 Utilisation des nouveaux endpoints

### 1. Obtenir les specs du camion

```bash
GET /api/truck/specs
```

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

### 2. Calculer le coût d'un trajet

```bash
POST /api/truck/calculate
Content-Type: application/json

{
  "origin": "Paris",
  "destination": "Lyon"
}
```

**Réponse :**
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
    {
      "item": "Péage autoroute",
      "amount": 62.8,
      "unit": "€"
    },
    {
      "item": "Carburant biogaz (115.5 kg)",
      "amount": 98.18,
      "unit": "€"
    },
    {
      "item": "TOTAL",
      "amount": 160.98,
      "unit": "€"
    }
  ]
}
```

### 3. Lister les prix de péages

```bash
GET /api/truck/toll-prices
```

### 4. Comparer camion vs voiture

```bash
POST /api/truck/compare-with-car
Content-Type: application/json

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

## 🎯 Paramètres du calculateur

### Valeurs par défaut (modifiables)

- **Consommation** : 25 kg de biogaz / 100 km
- **Prix du biogaz** : 0.85 €/kg (production Symone)
- **Émissions CO2** : 15 g/km (quasi neutre)
- **Classe de péage** : 4 (poids lourds)

### Personnalisation

Vous pouvez override ces valeurs dans chaque requête :

```json
{
  "origin": "Paris",
  "destination": "Lyon",
  "custom_consumption": 22.0,
  "custom_biogaz_price": 0.75
}
```

## 💡 Cas d'usage

1. **Calcul automatique de devis** pour clients Symone
2. **Comparaison compétitivité** biogaz vs diesel
3. **Analyse environnementale** des trajets
4. **Planification de coûts** pour la flotte
5. **Comparaison transport marchandises** vs covoiturage

## 🔧 Ajustements possibles

### Modifier le prix du biogaz

Dans `symone_truck_calculator.py`, ligne 18 :
```python
biogaz_price_per_kg: float = 0.85  # Modifiez ici
```

### Ajouter des trajets de péage

Dans `symone_truck_calculator.py`, dictionnaire `TRUCK_TOLL_PRICES` :
```python
("nantes", "bordeaux"): 45.50,  # Ajoutez vos trajets
```

### Modifier la consommation

Ligne 16 :
```python
consumption_per_100km: float = 25.0  # Modifiez ici
```

## ✅ Tests

Exécutez le module en standalone pour tester :

```bash
python backend/symone_truck_calculator.py
```

Vous verrez des exemples de calculs pour différents trajets.

## 📈 Évolutions futures possibles

1. Intégration avec un système de tarification dynamique du biogaz
2. Prise en compte du poids de la cargaison (impact sur consommation)
3. Calcul multi-étapes (avec arrêts)
4. Historique des trajets et analytics
5. API de prédiction de trafic pour optimiser les coûts

## 🆘 Support

Pour toute question sur l'intégration, consultez :
- La documentation FastAPI : https://fastapi.tiangolo.com
- Les exemples dans `symone_truck_calculator.py`
