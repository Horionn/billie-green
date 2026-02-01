"""
Billie Green - API Backend
API FastAPI pour la plateforme de tarification intelligente
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from backend.vehicle_database import VehicleDatabase
from backend.distance_calculator import TripCalculator
from backend.openroute_service import (
    get_openroute_service,
    get_city_coords_fallback,
    OpenRouteService
)
from backend.co2_calculator import get_co2_calculator, CO2Calculator


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
    use_openroute: bool = Field(True, description="Utiliser OpenRouteService pour le calcul de distance")


class CO2Details(BaseModel):
    """Détails complets des émissions CO2 (modèle GreenGo)."""
    # Émissions du trajet
    co2_total_kg: float
    co2_per_person_kg: float
    co2_fabrication_kg: float
    co2_utilisation_kg: float

    # Quota annuel (2 tonnes pour limiter à 2°C)
    quota_percent: float

    # Comparaison transports alternatifs
    comparison: Dict[str, float]

    # Équivalences parlantes
    equivalences: Dict

    # Billie Green (votre service)
    billie_green_co2_kg: float
    co2_saved_vs_car_kg: float
    co2_saved_percent: float

    # Messages
    savings_message: Dict


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
    co2_details: CO2Details  # Nouveau: détails CO2 enrichis
    explanation: str
    breakdown: List[Dict]


# Services
trip_calculator = TripCalculator()
vehicle_db = None
openroute_service: Optional[OpenRouteService] = None
DATA_PATH = Path(__file__).parent.parent / "ADEME-CarLabelling.csv"


CO2_REFERENCE_POINTS = [
    (0,   0.95),
    (50,  0.85),
    (100, 0.70),
    (120, 0.55),
    (150, 0.40),
    (200, 0.30),
    (300, 0.20),  # plafond de pollution
]


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

def interpolate_score(x: float, points: list[tuple[float, float]]) -> float:
    """
    Interpolation linéaire entre des points (x, score).
    """
    if x <= points[0][0]:
        return points[0][1]

    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]

        if x <= x1:
            # interpolation linéaire
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)

    return points[-1][1]



def calculate_eco_score(
    energy: str,
    model: str,
    co2_g_km: float = None
) -> float:
    """
    Score écologique continu (0 = polluant, 1 = propre).
    """
    energy = (energy or '').upper()
    model = (model or '').upper()

    # PRIORITÉ 1: CO2 réel → score continu
    if co2_g_km is not None and co2_g_km >= 0:
        return interpolate_score(co2_g_km, CO2_REFERENCE_POINTS)

    # PRIORITÉ 2: modèle électrique connu
    if any(m in model for m in ELECTRIC_MODELS):
        return 0.95

    # PRIORITÉ 3: énergie (fallback)
    if energy in ('ELECTRIC', 'ELECTRIQUE'):
        return 0.95

    if 'ELEC' in energy and ('GAZ' in energy or 'ESS' in energy):
        return 0.65  # hybride sans bonus excessif

    if any(m in model for m in DIESEL_SUV_MODELS):
        return 0.35

    energy_scores = {
        'HYBRIDE': 0.65,
        'ESSENCE': 0.40,
        'GAZOLE': 0.35,
        'DIESEL': 0.35,
    }

    for key, score in energy_scores.items():
        if key in energy:
            return score

    return 0.35



def calculate_social_score(argus_value: float) -> float:
    """Calcule la capacité contributive (0 = protégé, 1 = élevée)"""
    if argus_value is None or argus_value <= 0:
        argus_value = 10000  # Défaut
    return min(argus_value / 35000, 1.0)


def calculate_ethical_adjustment(eco_score: float, social_score: float) -> tuple:
    """
    Ajustement progressif, sans seuils durs.
    """
    # Bonus écologique progressif
    if eco_score >= 0.6:
        bonus = -0.12 * ((eco_score - 0.6) / 0.4)
        return bonus, "Bonus écologique", "Véhicule à faibles émissions"

    # Malus écologique progressif
    if eco_score <= 0.45:
        pollution_factor = (0.45 - eco_score) / 0.45

        if social_score > 0.6:
            return 0.15 * pollution_factor, "Contribution écologique", "Capacité contributive élevée"
        elif social_score > 0.3:
            return 0.05 * pollution_factor, "Contribution écologique", "Capacité contributive moyenne"
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
    global vehicle_db, openroute_service
    try:
        vehicle_db = VehicleDatabase(str(DATA_PATH))
        print(f"✓ Base de données ADEME chargée: {len(vehicle_db.df)} véhicules")
    except Exception as e:
        print(f"✗ Erreur chargement base ADEME: {e}")
        vehicle_db = None

    # Initialiser OpenRouteService
    openroute_service = get_openroute_service()
    if openroute_service.is_configured():
        print("✓ OpenRouteService API configurée")
    else:
        print("⚠ OpenRouteService API non configurée (OPENROUTE_API_KEY manquante)")
        print("  → Utilisation du calcul de distance local en fallback")


@app.on_event("shutdown")
async def shutdown_event():
    """Ferme les connexions proprement."""
    global openroute_service
    if openroute_service:
        await openroute_service.close()


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

    # 2. Calculer le trajet - Utiliser OpenRouteService si activé
    trip_info = None

    # Utiliser OpenRouteService seulement si demandé ET configuré
    if request.use_openroute and openroute_service and openroute_service.is_configured():
        try:
            route = await openroute_service.calculate_route_by_names(
                request.trip.origin,
                request.trip.destination
            )
            if route:
                from backend.distance_calculator import estimate_traffic_factor, get_week_type

                traffic_factor, traffic_desc = estimate_traffic_factor(
                    request.trip.day_of_week,
                    request.trip.hour,
                    request.trip.is_holiday,
                    request.trip.is_summer
                )

                adjusted_duration = route.duration_hours * traffic_factor
                adj_hours = int(adjusted_duration)
                adj_minutes = int((adjusted_duration - adj_hours) * 60)

                trip_info = {
                    "source": "openroute",
                    "distance_km": route.distance_km,
                    "duration_hours": route.duration_hours,
                    "duration_formatted": route.duration_formatted,
                    "traffic_factor": traffic_factor,
                    "traffic_description": traffic_desc,
                    "adjusted_duration_hours": round(adjusted_duration, 2),
                    "adjusted_duration_formatted": f"{adj_hours}h{adj_minutes:02d}",
                    "week_type": get_week_type(request.trip.is_holiday, request.trip.is_summer),
                    "origin": {"name": request.trip.origin, "lon": route.origin_coords[0], "lat": route.origin_coords[1]},
                    "destination": {"name": request.trip.destination, "lon": route.destination_coords[0], "lat": route.destination_coords[1]}
                }
        except Exception as e:
            print(f"Erreur OpenRouteService: {e}, fallback local")

    # Fallback sur le calculateur local (ou si use_openroute=false)
    if trip_info is None:
        trip_info = trip_calculator.calculate_trip(
            origin=request.trip.origin,
            destination=request.trip.destination,
            day_of_week=request.trip.day_of_week,
            hour=request.trip.hour,
            is_holiday=request.trip.is_holiday,
            is_summer=request.trip.is_summer
        )
        trip_info["source"] = "local"

    if "error" in trip_info:
        raise HTTPException(status_code=400, detail=trip_info["error"])

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

    # 10. Calcul du CO2 enrichi (modèle GreenGo)
    co2_calc = get_co2_calculator()
    co2_result = co2_calc.calculate(
        distance_km=distance,
        energy=energy,
        passengers=request.passengers,
        co2_ademe_g_km=co2_g_km
    )

    # Messages d'économie
    savings_message = co2_calc.get_savings_message(
        co2_result.co2_saved_vs_car_kg,
        co2_result.co2_saved_percent
    )

    co2_details = CO2Details(
        co2_total_kg=co2_result.co2_total_kg,
        co2_per_person_kg=co2_result.co2_per_person_kg,
        co2_fabrication_kg=co2_result.co2_fabrication_kg,
        co2_utilisation_kg=co2_result.co2_utilisation_kg,
        quota_percent=co2_result.quota_percent,
        comparison=co2_result.comparison,
        equivalences=co2_result.equivalences,
        billie_green_co2_kg=co2_result.billie_green_co2_kg,
        co2_saved_vs_car_kg=co2_result.co2_saved_vs_car_kg,
        co2_saved_percent=co2_result.co2_saved_percent,
        savings_message=savings_message
    )

    # 11. Explication
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


# ============================================================================
# NOUVEAUX ENDPOINTS: AUTOCOMPLÉTION ADEME
# ============================================================================

@app.get("/api/vehicle/autocomplete/brands")
async def autocomplete_brands(q: str = Query(..., min_length=1, description="Texte à rechercher")):
    """
    Autocomplétion des marques de véhicules depuis la base ADEME.

    Args:
        q: Texte à rechercher (min 1 caractère)

    Returns:
        Liste des marques correspondantes
    """
    if vehicle_db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible")

    query = q.upper().strip()
    all_brands = vehicle_db.get_all_brands()

    # Filtrer les marques qui commencent par la requête ou la contiennent
    matches = []
    for brand in all_brands:
        if brand.startswith(query):
            matches.insert(0, brand)  # Priorité aux correspondances exactes au début
        elif query in brand:
            matches.append(brand)

    return matches[:10]  # Limiter à 10 résultats


@app.get("/api/vehicle/autocomplete/models")
async def autocomplete_models(
    brand: str = Query(..., description="Marque du véhicule"),
    q: str = Query("", description="Texte à rechercher dans les modèles")
):
    """
    Autocomplétion des modèles pour une marque donnée depuis la base ADEME.

    Args:
        brand: Marque du véhicule
        q: Texte à rechercher (optionnel)

    Returns:
        Liste des modèles correspondants avec leurs infos CO2
    """
    if vehicle_db is None:
        raise HTTPException(status_code=500, detail="Base de données non disponible")

    brand = brand.upper().strip()
    query = q.upper().strip()

    # Obtenir tous les modèles pour cette marque
    all_models = vehicle_db.get_models_for_brand(brand)

    if not query:
        # Retourner tous les modèles avec leurs infos
        results = []
        seen = set()
        for model in all_models[:30]:  # Limiter à 30
            if model not in seen:
                seen.add(model)
                # Récupérer les infos CO2 du modèle
                vehicle_info = vehicle_db.search_vehicle(brand, model)
                if vehicle_info:
                    results.append({
                        "model": model,
                        "co2_g_km": vehicle_info.get("co2_g_km", 150),
                        "energie": vehicle_info.get("energie", "INCONNU")
                    })
                else:
                    results.append({
                        "model": model,
                        "co2_g_km": 150,
                        "energie": "INCONNU"
                    })
        return results

    # Filtrer par la requête
    matches = []
    seen = set()
    for model in all_models:
        if model in seen:
            continue
        seen.add(model)

        if model.startswith(query) or query in model:
            vehicle_info = vehicle_db.search_vehicle(brand, model)
            if vehicle_info:
                matches.append({
                    "model": model,
                    "co2_g_km": vehicle_info.get("co2_g_km", 150),
                    "energie": vehicle_info.get("energie", "INCONNU"),
                    "priority": 0 if model.startswith(query) else 1
                })

    # Trier par priorité (commence par > contient)
    matches.sort(key=lambda x: (x.get("priority", 1), x["model"]))
    for m in matches:
        m.pop("priority", None)

    return matches[:15]


@app.get("/api/vehicle/search-quick")
async def search_vehicle_quick(
    brand: str = Query(..., description="Marque"),
    model: str = Query(..., description="Modèle")
):
    """
    Recherche rapide d'un véhicule pour obtenir ses émissions CO2.

    Args:
        brand: Marque du véhicule
        model: Modèle du véhicule

    Returns:
        Infos du véhicule (CO2, énergie, etc.)
    """
    if vehicle_db is None:
        return {
            "found": False,
            "co2_g_km": 150,
            "energie": "INCONNU"
        }

    result = vehicle_db.search_vehicle(brand, model)

    if result:
        co2 = result.get("co2_g_km", 150)
        eco_score = calculate_eco_score(result.get("energie", ""), model, co2)
        co2_cat, co2_color = get_co2_category(co2, eco_score)

        return {
            "found": True,
            "marque": result.get("marque"),
            "modele": result.get("modele"),
            "co2_g_km": co2,
            "energie": result.get("energie", "INCONNU"),
            "co2_category": co2_cat,
            "co2_color": co2_color,
            "eco_score": round(eco_score, 2)
        }

    return {
        "found": False,
        "co2_g_km": 150,
        "energie": "INCONNU"
    }


# ============================================================================
# NOUVEAUX ENDPOINTS: OPENROUTESERVICE
# ============================================================================

@app.get("/api/openroute/status")
async def openroute_status():
    """Vérifie le statut de l'API OpenRouteService."""
    if openroute_service is None:
        return {"configured": False, "message": "Service non initialisé"}

    return {
        "configured": openroute_service.is_configured(),
        "message": "API OpenRouteService prête" if openroute_service.is_configured()
                   else "Clé API manquante (OPENROUTE_API_KEY)"
    }


@app.get("/api/openroute/autocomplete")
async def openroute_autocomplete(
    q: str = Query(..., min_length=2, description="Texte à rechercher"),
    country: str = Query("FR", description="Code pays ISO")
):
    """
    Autocomplétion d'adresses/villes via OpenRouteService.

    Args:
        q: Texte à rechercher (min 2 caractères)
        country: Code pays (défaut: FR)

    Returns:
        Liste de suggestions avec coordonnées
    """
    if openroute_service is None or not openroute_service.is_configured():
        # Fallback: retourner les villes locales correspondantes
        from backend.openroute_service import FRENCH_CITIES_COORDS
        query = q.lower()
        results = []
        for city, coords in FRENCH_CITIES_COORDS.items():
            if query in city:
                results.append({
                    "name": city.replace("-", " ").title(),
                    "label": f"{city.replace('-', ' ').title()}, France",
                    "region": "",
                    "country": "France",
                    "lon": coords[0],
                    "lat": coords[1]
                })
        return results[:5]

    return await openroute_service.autocomplete(q, country, limit=5)


@app.get("/api/openroute/geocode")
async def openroute_geocode(
    place: str = Query(..., description="Nom du lieu à géocoder")
):
    """
    Géocode un nom de lieu en coordonnées.

    Args:
        place: Nom de la ville ou adresse

    Returns:
        Coordonnées (lon, lat) ou erreur
    """
    # Essayer d'abord le fallback local (plus rapide)
    local_coords = get_city_coords_fallback(place)
    if local_coords:
        return {
            "found": True,
            "source": "local",
            "lon": local_coords[0],
            "lat": local_coords[1],
            "place": place
        }

    # Sinon, utiliser l'API OpenRoute
    if openroute_service and openroute_service.is_configured():
        coords = await openroute_service.geocode(place)
        if coords:
            return {
                "found": True,
                "source": "openroute",
                "lon": coords[0],
                "lat": coords[1],
                "place": place
            }

    return {"found": False, "place": place}


@app.post("/api/openroute/route")
async def openroute_calculate_route(trip: TripInput):
    """
    Calcule un itinéraire précis via OpenRouteService.

    Args:
        trip: Informations du trajet (origin, destination)

    Returns:
        Distance et durée précises
    """
    origin = trip.origin
    destination = trip.destination

    # Essayer d'abord avec OpenRouteService si configuré
    if openroute_service and openroute_service.is_configured():
        route = await openroute_service.calculate_route_by_names(origin, destination)

        if route:
            # Ajouter le facteur de trafic
            from backend.distance_calculator import estimate_traffic_factor, get_week_type

            traffic_factor, traffic_desc = estimate_traffic_factor(
                trip.day_of_week, trip.hour, trip.is_holiday, trip.is_summer
            )

            adjusted_duration = route.duration_hours * traffic_factor
            adj_hours = int(adjusted_duration)
            adj_minutes = int((adjusted_duration - adj_hours) * 60)

            return {
                "source": "openroute",
                "distance_km": route.distance_km,
                "duration_hours": route.duration_hours,
                "duration_formatted": route.duration_formatted,
                "traffic_factor": traffic_factor,
                "traffic_description": traffic_desc,
                "adjusted_duration_hours": round(adjusted_duration, 2),
                "adjusted_duration_formatted": f"{adj_hours}h{adj_minutes:02d}",
                "week_type": get_week_type(trip.is_holiday, trip.is_summer),
                "origin": {"name": origin, "lon": route.origin_coords[0], "lat": route.origin_coords[1]},
                "destination": {"name": destination, "lon": route.destination_coords[0], "lat": route.destination_coords[1]}
            }

    # Fallback sur le calculateur local
    result = trip_calculator.calculate_trip(
        origin=origin,
        destination=destination,
        day_of_week=trip.day_of_week,
        hour=trip.hour,
        is_holiday=trip.is_holiday,
        is_summer=trip.is_summer
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    result["source"] = "local"
    return result


@app.post("/api/trip/calculate-enhanced")
async def calculate_trip_enhanced(trip: TripInput):
    """
    Calcule un trajet avec OpenRouteService si disponible, sinon fallback local.

    Combine les avantages de l'API externe (précision) et du calcul local (rapidité).
    """
    # Essayer OpenRouteService en premier
    if openroute_service and openroute_service.is_configured():
        try:
            return await openroute_calculate_route(trip)
        except Exception as e:
            print(f"Erreur OpenRoute, fallback local: {e}")

    # Fallback local
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

    result["source"] = "local"
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
