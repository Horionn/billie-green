"""
Extension de l'API Billie Green pour les camions Symone
À ajouter dans main.py
"""

from pydantic import BaseModel, Field
from typing import Optional

# Importer le calculateur
from symone_truck_calculator import SymoneTruckCalculator, TruckSpecs, calculate_symone_truck_cost


# ============================================
# MODÈLES PYDANTIC POUR L'API
# ============================================

class TruckTripInput(BaseModel):
    """Input pour calcul de coût camion."""
    origin: str = Field(..., description="Ville de départ", example="Paris")
    destination: str = Field(..., description="Ville d'arrivée", example="Lyon")
    distance_km: Optional[float] = Field(None, description="Distance en km (calculée auto si None)")
    custom_consumption: Optional[float] = Field(None, description="Consommation personnalisée (kg/100km)", example=25.0)
    custom_biogaz_price: Optional[float] = Field(None, description="Prix personnalisé du biogaz (€/kg)", example=0.85)


class TruckCostResponse(BaseModel):
    """Réponse du calcul de coût camion."""
    truck_info: dict
    trip: dict
    toll: dict
    fuel: dict
    total_cost_euros: float
    cost_per_km_euros: float
    environmental: dict
    comparison: dict
    breakdown: list


# ============================================
# ENDPOINTS À AJOUTER DANS VOTRE APP FASTAPI
# ============================================

# Initialiser le calculateur
truck_calculator = SymoneTruckCalculator()


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
    
    Si distance_km n'est pas fournie, elle sera calculée automatiquement 
    en utilisant le calculateur de distance existant.
    """
    # Si distance non fournie, utiliser le calculateur existant
    if trip.distance_km is None:
        from backend.distance_calculator import calculate_distance
        distance_info = calculate_distance(trip.origin, trip.destination)
        
        if "error" in distance_info:
            raise HTTPException(status_code=400, detail=distance_info["error"])
        
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


@app.post("/api/truck/compare-with-car")
async def compare_truck_with_car(
    trip: TruckTripInput,
    vehicle: VehicleInput,
    passengers: int = Field(1, ge=1, le=8)
):
    """
    Compare le coût d'un trajet en camion Symone vs voiture particulière.
    Utile pour analyser la compétitivité du transport de marchandises vs covoiturage.
    """
    # 1. Calculer le coût camion
    truck_result = await calculate_truck_cost(trip)
    
    # 2. Calculer le prix voiture (utiliser le endpoint existant)
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
    return {
        "truck": {
            "total_cost": truck_result.total_cost_euros,
            "cost_per_km": truck_result.cost_per_km_euros,
            "co2_kg": truck_result.environmental["co2_total_kg"],
            "fuel_type": "Biogaz"
        },
        "car": {
            "total_cost": car_result.final_price,
            "cost_per_passenger": car_result.final_price,
            "total_for_all_passengers": car_result.final_price * passengers,
            "co2_category": car_result.co2_category,
            "fuel_type": car_result.vehicle_info.get("energie", "Unknown")
        },
        "comparison": {
            "truck_cheaper": truck_result.total_cost_euros < (car_result.final_price * passengers),
            "cost_difference": round(truck_result.total_cost_euros - (car_result.final_price * passengers), 2),
            "truck_vs_car_ratio": round(truck_result.total_cost_euros / (car_result.final_price * passengers), 2) if car_result.final_price > 0 else 0
        }
    }


@app.get("/api/truck/toll-prices")
async def list_truck_toll_prices():
    """Liste tous les prix de péages disponibles pour poids lourds."""
    from symone_truck_calculator import TRUCK_TOLL_PRICES
    
    return {
        "toll_class": 4,
        "description": "Tarifs péages classe 4 (poids lourds > 3.5t)",
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
