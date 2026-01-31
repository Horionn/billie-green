"""
Billie Green - API Backend
API FastAPI pour la plateforme de tarification intelligente
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import torch
import numpy as np

from models.pricing_model import BillieGreenPricingModel, create_model
from backend.vehicle_database import VehicleDatabase, get_vehicle_database
from backend.distance_calculator import TripCalculator, calculate_distance


# Initialisation de l'application
app = FastAPI(
    title="Billie Green API",
    description="API de tarification intelligente pour la mobilité durable",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modèles Pydantic pour les requêtes/réponses
class VehicleInput(BaseModel):
    brand: str = Field(..., description="Marque du véhicule", example="RENAULT")
    model: str = Field(..., description="Modèle du véhicule", example="CLIO")
    year: Optional[int] = Field(None, description="Année du véhicule", example=2018)
    energy: Optional[str] = Field(None, description="Type d'énergie", example="ESSENCE")
    argus_value: Optional[float] = Field(None, description="Valeur ARGUS (€)", example=8000)


class TripInput(BaseModel):
    origin: str = Field(..., description="Ville de départ", example="Paris")
    destination: str = Field(..., description="Ville d'arrivée", example="Lyon")
    day_of_week: int = Field(0, ge=0, le=6, description="Jour de la semaine (0=lundi)")
    hour: int = Field(10, ge=0, le=23, description="Heure de départ")
    is_holiday: bool = Field(False, description="Période de vacances")
    is_summer: bool = Field(False, description="Période estivale (juillet-août)")


class PricingRequest(BaseModel):
    vehicle: VehicleInput
    trip: TripInput
    passengers: int = Field(1, ge=1, le=8, description="Nombre de passagers")


class PricingResponse(BaseModel):
    # Prix
    final_price: float
    base_price: float
    eco_adjustment_percent: float
    demand_adjustment_percent: float

    # Scores
    eco_score: float
    social_score: float

    # Détails véhicule
    vehicle_info: Dict
    co2_category: str
    co2_color: str

    # Détails trajet
    trip_info: Dict

    # Explication
    explanation: str
    breakdown: List[Dict]


class VehicleSearchResponse(BaseModel):
    vehicles: List[Dict]


class CitySearchResponse(BaseModel):
    cities: List[Dict]


# Initialisation des services
trip_calculator = TripCalculator()
pricing_model = create_model()

# Charger la base de données véhicules
DATA_PATH = Path(__file__).parent.parent / "ADEME-CarLabelling.csv"


@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage."""
    global vehicle_db
    try:
        vehicle_db = VehicleDatabase(str(DATA_PATH))
        print(f"Base de données chargée: {len(vehicle_db.df)} véhicules")
    except Exception as e:
        print(f"Erreur chargement base: {e}")
        vehicle_db = None


@app.get("/")
async def root():
    """Page d'accueil."""
    return {
        "message": "Bienvenue sur l'API Billie Green",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/brands", response_model=List[str])
async def get_brands():
    """Liste des marques de véhicules disponibles."""
    if vehicle_db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible")
    return vehicle_db.get_all_brands()


@app.get("/api/models/{brand}", response_model=List[str])
async def get_models(brand: str):
    """Liste des modèles pour une marque."""
    if vehicle_db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible")
    return vehicle_db.get_models_for_brand(brand)


@app.post("/api/vehicle/search", response_model=Dict)
async def search_vehicle(vehicle: VehicleInput):
    """Recherche un véhicule et retourne ses caractéristiques."""
    if vehicle_db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible")

    result = vehicle_db.search_vehicle(
        brand=vehicle.brand,
        model=vehicle.model,
        year=vehicle.year,
        energy=vehicle.energy
    )

    if result is None:
        # Retourner des valeurs par défaut
        return {
            "found": False,
            "marque": vehicle.brand,
            "modele": vehicle.model,
            "co2_g_km": 150.0,  # Valeur moyenne
            "energie": vehicle.energy or "INCONNU",
            "message": "Véhicule non trouvé, valeurs estimées utilisées"
        }

    # Calculer la valeur ARGUS si non fournie
    if vehicle.argus_value:
        argus = vehicle.argus_value
    elif vehicle.year and result.get('prix_neuf', 0) > 0:
        age = 2024 - vehicle.year
        argus = vehicle_db.estimate_argus(result['prix_neuf'], age)
    else:
        argus = 10000  # Valeur par défaut

    # Catégorie CO2
    co2_cat, co2_color = vehicle_db.get_co2_category(result['co2_g_km'])

    return {
        "found": True,
        **result,
        "argus_estimé": round(argus, 0),
        "co2_category": co2_cat,
        "co2_color": co2_color
    }


@app.post("/api/trip/calculate", response_model=Dict)
async def calculate_trip(trip: TripInput):
    """Calcule les informations d'un trajet."""
    result = trip_calculator.calculate_trip(
        origin=trip.origin,
        destination=trip.destination,
        day_of_week=trip.day_of_week,
        hour=trip.hour,
        is_holiday=trip.is_holiday,
        is_summer=trip.is_summer
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/api/pricing/calculate", response_model=PricingResponse)
async def calculate_pricing(request: PricingRequest):
    """
    Calcule la tarification complète pour un trajet.

    Cette endpoint est le coeur de Billie Green:
    - Analyse le profil écologique du véhicule
    - Évalue le profil socio-économique
    - Applique les bonus/malus de manière éthique
    - Retourne un prix ajusté avec explications
    """
    # 1. Rechercher le véhicule
    vehicle_info = await search_vehicle(request.vehicle)

    # 2. Calculer le trajet
    trip_info = await calculate_trip(request.trip)

    # 3. Préparer les inputs pour le modèle PyTorch
    co2_per_km = vehicle_info.get('co2_g_km', 150.0)

    # Valeur ARGUS
    if request.vehicle.argus_value:
        argus = request.vehicle.argus_value
    else:
        argus = vehicle_info.get('argus_estimé', 10000)

    # Âge du véhicule
    if request.vehicle.year:
        car_age = 2024 - request.vehicle.year
    else:
        car_age = 5  # Défaut

    # Convertir en tenseurs
    inputs = {
        'co2_per_km': torch.tensor([co2_per_km], dtype=torch.float32),
        'distance': torch.tensor([trip_info['distance_km']], dtype=torch.float32),
        'traffic_factor': torch.tensor([trip_info['traffic_factor']], dtype=torch.float32),
        'passengers': torch.tensor([request.passengers], dtype=torch.float32),
        'argus_value': torch.tensor([argus], dtype=torch.float32),
        'car_age': torch.tensor([car_age], dtype=torch.float32),
        'day_of_week': torch.tensor([trip_info['day_of_week']], dtype=torch.long),
        'week_type': torch.tensor([trip_info['week_type']], dtype=torch.long)
    }

    # 4. Prédiction du modèle
    with torch.no_grad():
        output = pricing_model(**inputs)

    # 5. Extraire les résultats
    final_price = output['final_price'].item()
    base_price = output['base_price'].item()
    eco_score = output['eco_score'].item()
    social_score = output['social_score'].item()
    eco_adjustment = output['eco_adjustment'].item()
    demand_factor = output['demand_factor'].item()

    # 6. Générer l'explication
    explanation_parts = []
    breakdown = []

    # Prix de base
    breakdown.append({
        "label": "Prix de base",
        "value": f"{base_price:.2f}€",
        "detail": f"{trip_info['distance_km']} km × 0.15€/km"
    })

    # Score écologique
    if eco_score > 0.7:
        explanation_parts.append("Votre trajet est éco-responsable")
        if co2_per_km == 0:
            breakdown.append({
                "label": "Bonus véhicule électrique",
                "value": f"-{abs(eco_adjustment)*100:.1f}%",
                "detail": "Zéro émission"
            })
        else:
            breakdown.append({
                "label": "Bonus faibles émissions",
                "value": f"-{abs(eco_adjustment)*100:.1f}%",
                "detail": f"Seulement {co2_per_km:.0f}g CO2/km"
            })
    elif eco_score < 0.3:
        if social_score < 0.4:
            explanation_parts.append("Impact carbone élevé mais malus atténué (profil protégé)")
            breakdown.append({
                "label": "Malus écologique atténué",
                "value": f"+{eco_adjustment*100:.1f}%",
                "detail": "Protection sociale appliquée"
            })
        else:
            explanation_parts.append("Contribution écologique pour véhicule polluant")
            breakdown.append({
                "label": "Contribution écologique",
                "value": f"+{eco_adjustment*100:.1f}%",
                "detail": f"Émissions: {co2_per_km:.0f}g CO2/km"
            })
    else:
        explanation_parts.append("Impact carbone modéré")

    # Covoiturage
    if request.passengers > 1:
        explanation_parts.append(f"Bonus covoiturage ({request.passengers} passagers)")
        breakdown.append({
            "label": "Bonus covoiturage",
            "value": "inclus",
            "detail": f"Émissions divisées par {request.passengers}"
        })

    # Demande
    if demand_factor > 1.1:
        breakdown.append({
            "label": "Période de forte demande",
            "value": f"+{(demand_factor-1)*100:.0f}%",
            "detail": trip_info['traffic_description']
        })
    elif demand_factor < 0.95:
        breakdown.append({
            "label": "Période creuse",
            "value": f"{(demand_factor-1)*100:.0f}%",
            "detail": "Tarif avantageux"
        })

    # Prix final
    breakdown.append({
        "label": "Prix final",
        "value": f"{final_price:.2f}€",
        "detail": "Tarif Billie Green"
    })

    return PricingResponse(
        final_price=round(final_price, 2),
        base_price=round(base_price, 2),
        eco_adjustment_percent=round(eco_adjustment * 100, 1),
        demand_adjustment_percent=round((demand_factor - 1) * 100, 1),
        eco_score=round(eco_score, 2),
        social_score=round(social_score, 2),
        vehicle_info=vehicle_info,
        co2_category=vehicle_info.get('co2_category', 'D'),
        co2_color=vehicle_info.get('co2_color', '#F9A01B'),
        trip_info=trip_info,
        explanation=" | ".join(explanation_parts),
        breakdown=breakdown
    )


@app.get("/api/cities")
async def list_cities():
    """Liste des villes disponibles."""
    from backend.distance_calculator import CITIES_DB

    return [
        {
            "name": city.name,
            "region": city.region,
            "key": key
        }
        for key, city in CITIES_DB.items()
    ]


@app.get("/api/stats")
async def get_stats():
    """Statistiques générales."""
    if vehicle_db is None:
        return {"error": "Base de données non disponible"}

    co2_by_energy = vehicle_db.get_average_co2_by_energy()

    return {
        "total_vehicles": len(vehicle_db.df),
        "total_brands": len(vehicle_db.get_all_brands()),
        "co2_by_energy": co2_by_energy
    }


# Servir les fichiers statiques du frontend
frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Sert le frontend React."""
        file_path = frontend_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_path / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
