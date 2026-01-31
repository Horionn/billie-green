"""
Billie Green - API Backend
API FastAPI pour la plateforme de tarification intelligente
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from backend.vehicle_database import VehicleDatabase
from backend.distance_calculator import TripCalculator


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


# Modèles Pydantic
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
    is_summer: bool = Field(False, description="Période estivale")


class PricingRequest(BaseModel):
    vehicle: VehicleInput
    trip: TripInput
    passengers: int = Field(1, ge=1, le=8, description="Nombre de passagers")


class PricingResponse(BaseModel):
    final_price: float
    base_price: float
    eco_adjustment_percent: float
    eco_score: float
    social_score: float
    vehicle_info: Dict
    co2_category: str
    co2_color: str
    trip_info: Dict
    explanation: str
    breakdown: List[Dict]


# Services
trip_calculator = TripCalculator()
vehicle_db = None
DATA_PATH = Path(__file__).parent.parent / "ADEME-CarLabelling.csv"


# Modèles électriques connus
ELECTRIC_MODELS = [
    'MODEL 3', 'MODEL S', 'MODEL X', 'MODEL Y',  # Tesla
    'ZOE', 'MEGANE E-TECH',  # Renault
    'E-TRON', 'Q4 E-TRON',  # Audi
    'ID.3', 'ID.4', 'ID.5',  # VW
    'LEAF',  # Nissan
    'KONA ELECTRIC', 'IONIQ',  # Hyundai
    'E-208', 'E-2008',  # Peugeot
    'SPRING', 'E-C4',  # Dacia/Citroën
    'I3', 'IX3', 'I4', 'IX',  # BMW
    'EQA', 'EQB', 'EQC', 'EQS',  # Mercedes
]

# Modèles SUV/gros véhicules typiquement diesel (non présents dans la base ADEME)
DIESEL_SUV_MODELS = [
    'X5', 'X6', 'X7',  # BMW SUV
    'Q5', 'Q7', 'Q8',  # Audi SUV
    'GLC', 'GLE', 'GLS', 'ML',  # Mercedes SUV
    'CAYENNE', 'MACAN',  # Porsche
    'TOUAREG', 'TIGUAN',  # VW
    'RANGE ROVER', 'DISCOVERY', 'DEFENDER',  # Land Rover
    'GRAND CHEROKEE', 'CHEROKEE',  # Jeep
    'XC60', 'XC90',  # Volvo
    'SANTA FE', 'TUCSON',  # Hyundai
    'SORENTO', 'SPORTAGE',  # Kia
]


def calculate_eco_score(energy: str, model: str, co2_g_km: float = None) -> float:
    """
    Calcule le score écologique (0 = polluant, 1 = propre).

    PRIORITÉ: Utiliser le CO2 réel si disponible, sinon se baser sur le type d'énergie.
    """
    energy = (energy or '').upper()
    model = (model or '').upper()

    # PRIORITÉ 1: Si on a le CO2 réel, l'utiliser pour calculer le score
    if co2_g_km is not None and co2_g_km >= 0:
        if co2_g_km == 0:
            return 0.95  # Électrique pur
        elif co2_g_km <= 50:
            return 0.85  # Hybride rechargeable très efficace
        elif co2_g_km <= 100:
            return 0.70  # Hybride efficace
        elif co2_g_km <= 120:
            return 0.55  # Véhicule économe
        elif co2_g_km <= 150:
            return 0.40  # Véhicule moyen
        elif co2_g_km <= 200:
            return 0.30  # Véhicule polluant
        else:
            return 0.20  # Très polluant

    # PRIORITÉ 2: Détection par modèle électrique connu
    is_electric_model = any(m in model for m in ELECTRIC_MODELS)
    if is_electric_model:
        return 0.95

    # PRIORITÉ 3: Détection par type d'énergie
    # ATTENTION: "GAZ+ELEC" = hybride diesel, PAS électrique !
    if energy == 'ELECTRIC' or energy == 'ELECTRIQUE':
        return 0.95

    # Hybrides (incluant GAZ+ELEC, ESS+ELEC)
    if 'ELEC' in energy and ('GAZ' in energy or 'ESS' in energy):
        return 0.65  # Hybride = score moyen, PAS bonus électrique

    # Détection SUV diesel (souvent non dans la base ADEME)
    is_diesel_suv = any(m in model for m in DIESEL_SUV_MODELS)
    if is_diesel_suv:
        return 0.35  # Score diesel

    energy_scores = {
        'HYBRIDE': 0.65,
        'ESSENCE': 0.40,
        'GAZOLE': 0.35,
        'DIESEL': 0.35,
    }

    for key, score in energy_scores.items():
        if key in energy:
            return score

    # Par défaut: considérer comme diesel (conservateur pour le malus)
    return 0.35


def calculate_social_score(argus_value: float) -> float:
    """Calcule la capacité contributive (0 = protégé, 1 = élevée)"""
    if argus_value is None or argus_value <= 0:
        argus_value = 10000  # Défaut
    return min(argus_value / 35000, 1.0)


def calculate_ethical_adjustment(eco_score: float, social_score: float) -> tuple:
    """
    Calcule l'ajustement éthique.

    Logique:
    - Véhicule propre → bonus -12%
    - Véhicule polluant + riche → malus +15%
    - Véhicule polluant + moyen → malus +5%
    - Véhicule polluant + pauvre → PAS de malus (protégé)
    """
    if eco_score >= 0.7:
        return -0.12, "Bonus écologique", "Véhicule à faibles émissions"

    if eco_score < 0.45:
        if social_score > 0.6:
            return 0.15, "Contribution écologique", "Véhicule polluant, capacité contributive élevée"
        elif social_score > 0.3:
            return 0.05, "Contribution écologique", "Véhicule polluant, capacité contributive moyenne"
        else:
            return 0.0, "Malus exonéré", "Capacité contributive limitée"

    return 0.0, "", ""


def get_co2_category(co2_g_km: float = None, eco_score: float = None) -> tuple:
    """
    Retourne la catégorie CO2 et sa couleur.

    Basé sur les étiquettes énergie/CO2 françaises officielles.
    PRIORITÉ: Utiliser le CO2 réel si disponible.
    """
    # Si on a le CO2 réel, utiliser les seuils officiels
    if co2_g_km is not None and co2_g_km >= 0:
        if co2_g_km == 0:
            return "A", "#16a34a"  # Électrique
        elif co2_g_km <= 100:
            return "A", "#16a34a"  # Très faibles émissions
        elif co2_g_km <= 120:
            return "B", "#22c55e"  # Faibles émissions
        elif co2_g_km <= 140:
            return "C", "#eab308"  # Modérées
        elif co2_g_km <= 160:
            return "D", "#f97316"  # Élevées
        elif co2_g_km <= 200:
            return "E", "#ef4444"  # Très élevées
        else:
            return "F", "#dc2626"  # Extrêmes

    # Fallback sur le score écologique
    if eco_score is not None:
        if eco_score > 0.8:
            return "A", "#16a34a"
        elif eco_score > 0.6:
            return "B", "#22c55e"
        elif eco_score > 0.4:
            return "C", "#eab308"
        elif eco_score > 0.3:
            return "D", "#f97316"
        else:
            return "E", "#ef4444"

    return "D", "#f97316"  # Défaut


@app.on_event("startup")
async def startup_event():
    global vehicle_db
    try:
        vehicle_db = VehicleDatabase(str(DATA_PATH))
        print(f"✓ Base de données chargée: {len(vehicle_db.df)} véhicules")
    except Exception as e:
        print(f"✗ Erreur chargement base: {e}")
        vehicle_db = None


@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API Billie Green",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "database": vehicle_db is not None}


@app.get("/api/brands")
async def get_brands():
    if vehicle_db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible")
    return vehicle_db.get_all_brands()


@app.get("/api/models/{brand}")
async def get_models(brand: str):
    if vehicle_db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible")
    return vehicle_db.get_models_for_brand(brand)


@app.post("/api/vehicle/search")
async def search_vehicle(vehicle: VehicleInput):
    if vehicle_db is None:
        return {
            "found": False,
            "marque": vehicle.brand,
            "modele": vehicle.model,
            "co2_g_km": 150.0,
            "energie": vehicle.energy or "INCONNU",
        }

    result = vehicle_db.search_vehicle(
        brand=vehicle.brand,
        model=vehicle.model,
        year=vehicle.year,
        energy=vehicle.energy
    )

    if result is None:
        return {
            "found": False,
            "marque": vehicle.brand,
            "modele": vehicle.model,
            "co2_g_km": 150.0,
            "energie": vehicle.energy or "INCONNU",
        }

    # Calculer ARGUS si non fourni
    if vehicle.argus_value:
        argus = vehicle.argus_value
    elif vehicle.year and result.get('prix_neuf', 0) > 0:
        age = 2025 - vehicle.year
        argus = vehicle_db.estimate_argus(result['prix_neuf'], age)
    else:
        argus = 10000

    co2_g_km = result.get('co2_g_km', 150)
    eco_score = calculate_eco_score(result.get('energie', ''), vehicle.model, co2_g_km)
    co2_cat, co2_color = get_co2_category(co2_g_km, eco_score)

    return {
        "found": True,
        **result,
        "argus_estime": round(argus, 0),
        "co2_category": co2_cat,
        "co2_color": co2_color
    }


@app.post("/api/trip/calculate")
async def calculate_trip(trip: TripInput):
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
    """Calcule la tarification complète pour un trajet."""

    # 1. Rechercher le véhicule
    vehicle_info = await search_vehicle(request.vehicle)

    # 2. Calculer le trajet
    trip_info = await calculate_trip(request.trip)

    # 3. Calculer les scores
    energy = vehicle_info.get('energie', request.vehicle.energy or '')
    co2_g_km = vehicle_info.get('co2_g_km', 150)

    # Utiliser le CO2 réel pour calculer le score écologique
    eco_score = calculate_eco_score(energy, request.vehicle.model, co2_g_km)

    argus = request.vehicle.argus_value or vehicle_info.get('argus_estime', 10000)
    social_score = calculate_social_score(argus)

    # 4. Prix de base
    distance = trip_info['distance_km']
    base_price = max(distance * 0.15, 20)

    # 5. Ajustement éthique
    adjustment, adj_label, adj_detail = calculate_ethical_adjustment(eco_score, social_score)

    # 6. Bonus covoiturage
    passenger_bonus = (request.passengers - 1) * 0.05

    # 7. Prix final
    final_price = base_price * (1 + adjustment) * (1 - passenger_bonus)

    # 8. Catégorie CO2 (basée sur le CO2 réel)
    co2_cat, co2_color = get_co2_category(co2_g_km, eco_score)

    # 9. Breakdown
    breakdown = [
        {
            "label": "Prix de base",
            "value": f"{base_price:.2f} €",
            "detail": f"{distance} km × 0,15 €/km"
        }
    ]

    if adjustment != 0:
        breakdown.append({
            "label": adj_label,
            "value": f"{'+' if adjustment > 0 else ''}{adjustment*100:.0f}%",
            "detail": adj_detail
        })
    elif eco_score < 0.45 and social_score <= 0.3:
        breakdown.append({
            "label": "Malus exonéré",
            "value": "0%",
            "detail": "Capacité contributive limitée"
        })

    if request.passengers > 1:
        breakdown.append({
            "label": "Bonus covoiturage",
            "value": f"-{passenger_bonus*100:.0f}%",
            "detail": f"{request.passengers} passagers"
        })

    breakdown.append({
        "label": "Total",
        "value": f"{final_price:.0f} €",
        "detail": "par personne"
    })

    # 10. Explication
    if eco_score >= 0.7:
        explanation = "Véhicule à faibles émissions"
    elif eco_score < 0.45 and social_score <= 0.3:
        explanation = "Véhicule polluant mais malus exonéré (capacité limitée)"
    elif eco_score < 0.45:
        explanation = "Contribution écologique appliquée"
    else:
        explanation = "Tarification standard"

    return PricingResponse(
        final_price=round(final_price, 2),
        base_price=round(base_price, 2),
        eco_adjustment_percent=round(adjustment * 100, 1),
        eco_score=round(eco_score, 2),
        social_score=round(social_score, 2),
        vehicle_info=vehicle_info,
        co2_category=co2_cat,
        co2_color=co2_color,
        trip_info=trip_info,
        explanation=explanation,
        breakdown=breakdown
    )


@app.get("/api/cities")
async def list_cities():
    from backend.distance_calculator import CITIES_DB
    return [
        {"name": city.name, "region": city.region, "key": key}
        for key, city in CITIES_DB.items()
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
