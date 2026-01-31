"""
Billie Green - Calculateur de Distance et Trajet
Calcule la distance entre deux villes et estime le temps de trajet
"""

import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class City:
    """Représente une ville avec ses coordonnées."""
    name: str
    lat: float
    lon: float
    region: str


# Base de données des principales villes françaises sur l'axe Paris-Lyon-Méditerranée
CITIES_DB = {
    # Île-de-France
    "paris": City("Paris", 48.8566, 2.3522, "Île-de-France"),
    "fleury": City("Fleury-en-Bière", 48.4287, 2.5428, "Île-de-France"),
    "fontainebleau": City("Fontainebleau", 48.4041, 2.7010, "Île-de-France"),

    # Bourgogne
    "auxerre": City("Auxerre", 47.7989, 3.5714, "Bourgogne"),
    "avallon": City("Avallon", 47.4900, 3.9067, "Bourgogne"),
    "beaune": City("Beaune", 47.0258, 4.8400, "Bourgogne"),
    "dijon": City("Dijon", 47.3220, 5.0415, "Bourgogne"),
    "chalon-sur-saone": City("Chalon-sur-Saône", 46.7806, 4.8536, "Bourgogne"),
    "macon": City("Mâcon", 46.3067, 4.8328, "Bourgogne"),

    # Rhône-Alpes
    "lyon": City("Lyon", 45.7640, 4.8357, "Rhône-Alpes"),
    "villefranche": City("Villefranche-sur-Saône", 45.9833, 4.7167, "Rhône-Alpes"),
    "vienne": City("Vienne", 45.5247, 4.8758, "Rhône-Alpes"),
    "valence": City("Valence", 44.9333, 4.8920, "Rhône-Alpes"),
    "montelimar": City("Montélimar", 44.5580, 4.7497, "Rhône-Alpes"),
    "grenoble": City("Grenoble", 45.1885, 5.7245, "Rhône-Alpes"),

    # Provence
    "orange": City("Orange", 44.1361, 4.8083, "Provence"),
    "avignon": City("Avignon", 43.9493, 4.8055, "Provence"),
    "nimes": City("Nîmes", 43.8367, 4.3601, "Occitanie"),
    "arles": City("Arles", 43.6767, 4.6278, "Provence"),
    "salon": City("Salon-de-Provence", 43.6406, 5.0969, "Provence"),
    "aix": City("Aix-en-Provence", 43.5297, 5.4474, "Provence"),
    "marseille": City("Marseille", 43.2965, 5.3698, "Provence"),
    "lancon": City("Lançon-de-Provence", 43.5919, 5.1278, "Provence"),

    # Languedoc
    "montpellier": City("Montpellier", 43.6108, 3.8767, "Occitanie"),
    "beziers": City("Béziers", 43.3442, 3.2194, "Occitanie"),
    "narbonne": City("Narbonne", 43.1833, 3.0000, "Occitanie"),
    "perpignan": City("Perpignan", 42.6986, 2.8956, "Occitanie"),

    # Côte d'Azur
    "toulon": City("Toulon", 43.1242, 5.9280, "PACA"),
    "cannes": City("Cannes", 43.5528, 7.0174, "PACA"),
    "nice": City("Nice", 43.7102, 7.2620, "PACA"),
}

# Distances autoroutières réelles (en km) pour les trajets principaux
# Source: données de l'étude Symone + Google Maps
HIGHWAY_DISTANCES = {
    # Paris - destinations
    ("paris", "lyon"): 462,
    ("paris", "marseille"): 774,
    ("paris", "aix"): 759,
    ("paris", "montpellier"): 763,
    ("paris", "nice"): 932,
    ("paris", "grenoble"): 574,
    ("paris", "dijon"): 311,
    ("paris", "valence"): 560,
    ("paris", "avignon"): 683,
    ("paris", "toulon"): 839,
    ("paris", "cannes"): 902,

    # Lyon - destinations
    ("lyon", "marseille"): 314,
    ("lyon", "aix"): 299,
    ("lyon", "montpellier"): 303,
    ("lyon", "nice"): 472,
    ("lyon", "grenoble"): 111,
    ("lyon", "valence"): 100,
    ("lyon", "avignon"): 230,
    ("lyon", "toulon"): 378,
    ("lyon", "cannes"): 442,
    ("lyon", "dijon"): 192,

    # Marseille - destinations
    ("marseille", "montpellier"): 169,
    ("marseille", "nice"): 198,
    ("marseille", "grenoble"): 306,
    ("marseille", "aix"): 31,
    ("marseille", "avignon"): 99,
    ("marseille", "toulon"): 66,
    ("marseille", "cannes"): 178,
    ("marseille", "valence"): 215,

    # Grenoble - destinations
    ("grenoble", "nice"): 336,
    ("grenoble", "montpellier"): 295,
    ("grenoble", "aix"): 290,
    ("grenoble", "avignon"): 235,
    ("grenoble", "valence"): 95,
    ("grenoble", "toulon"): 350,
    ("grenoble", "cannes"): 315,
    ("grenoble", "dijon"): 299,

    # Montpellier - destinations
    ("montpellier", "nice"): 329,
    ("montpellier", "aix"): 153,
    ("montpellier", "avignon"): 92,
    ("montpellier", "toulon"): 233,
    ("montpellier", "cannes"): 303,
    ("montpellier", "valence"): 210,

    # Nice - destinations
    ("nice", "aix"): 177,
    ("nice", "avignon"): 256,
    ("nice", "toulon"): 150,
    ("nice", "cannes"): 33,
    ("nice", "valence"): 374,

    # Autres
    ("dijon", "lyon"): 192,
    ("valence", "orange"): 90,
    ("orange", "marseille"): 95,
    ("avignon", "aix"): 82,
    ("toulon", "cannes"): 123,
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la distance à vol d'oiseau entre deux points GPS.
    Retourne la distance en km.
    """
    R = 6371  # Rayon de la Terre en km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def normalize_city_name(name: str) -> str:
    """Normalise le nom d'une ville pour la recherche."""
    import unicodedata

    # Mettre en minuscule
    name = name.lower().strip()

    # Enlever les accents
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')

    # Gestion des alias courants
    aliases = {
        "aix-en-provence": "aix",
        "aix en provence": "aix",
        "salon-de-provence": "salon",
        "salon de provence": "salon",
        "villefranche-sur-saone": "villefranche",
        "villefranche sur saone": "villefranche",
        "chalon-sur-saone": "chalon-sur-saone",
        "chalon sur saone": "chalon-sur-saone",
        "lancon-de-provence": "lancon",
        "lancon de provence": "lancon",
    }

    return aliases.get(name, name)


def find_city(name: str) -> Optional[City]:
    """Trouve une ville dans la base de données."""
    normalized = normalize_city_name(name)

    # Recherche exacte
    if normalized in CITIES_DB:
        return CITIES_DB[normalized]

    # Recherche partielle
    for key, city in CITIES_DB.items():
        if normalized in key or key in normalized:
            return city

    return None


def get_highway_distance(origin: str, destination: str) -> Optional[float]:
    """
    Retourne la distance autoroutière entre deux villes si disponible.
    """
    o = normalize_city_name(origin)
    d = normalize_city_name(destination)

    # Chercher dans les deux sens
    if (o, d) in HIGHWAY_DISTANCES:
        return HIGHWAY_DISTANCES[(o, d)]
    if (d, o) in HIGHWAY_DISTANCES:
        return HIGHWAY_DISTANCES[(d, o)]

    return None


def calculate_distance(origin: str, destination: str) -> Dict:
    """
    Calcule la distance entre deux villes.

    Returns:
        Dict avec distance, temps estimé, et infos supplémentaires
    """
    city_origin = find_city(origin)
    city_dest = find_city(destination)

    if not city_origin:
        return {"error": f"Ville d'origine '{origin}' non trouvée"}
    if not city_dest:
        return {"error": f"Ville de destination '{destination}' non trouvée"}

    # Distance à vol d'oiseau
    direct_distance = haversine_distance(
        city_origin.lat, city_origin.lon,
        city_dest.lat, city_dest.lon
    )

    # Distance autoroutière (réelle si disponible, sinon estimation)
    highway_distance = get_highway_distance(origin, destination)
    if highway_distance is None:
        # Estimation: distance routière ≈ 1.3 × distance à vol d'oiseau
        highway_distance = direct_distance * 1.3

    # Temps de trajet estimé (vitesse moyenne autoroute: 110 km/h)
    base_duration_hours = highway_distance / 110

    # Convertir en heures et minutes
    hours = int(base_duration_hours)
    minutes = int((base_duration_hours - hours) * 60)

    return {
        "origin": {
            "name": city_origin.name,
            "region": city_origin.region,
            "lat": city_origin.lat,
            "lon": city_origin.lon
        },
        "destination": {
            "name": city_dest.name,
            "region": city_dest.region,
            "lat": city_dest.lat,
            "lon": city_dest.lon
        },
        "distance_km": round(highway_distance, 1),
        "direct_distance_km": round(direct_distance, 1),
        "duration_hours": round(base_duration_hours, 2),
        "duration_formatted": f"{hours}h{minutes:02d}",
        "is_exact_distance": get_highway_distance(origin, destination) is not None
    }


def estimate_traffic_factor(
    day_of_week: int,  # 0=lundi, 6=dimanche
    hour: int,         # 0-23
    is_holiday: bool = False,
    is_summer: bool = False
) -> Tuple[float, str]:
    """
    Estime le facteur de trafic selon les conditions.

    Returns:
        Tuple (facteur, description)
        facteur: 1.0 = trafic normal, 1.5 = embouteillage
    """
    # Périodes de pointe
    is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
    is_weekend = day_of_week >= 5
    is_friday_evening = day_of_week == 4 and hour >= 15
    is_sunday_evening = day_of_week == 6 and hour >= 16

    factor = 1.0
    description = "Trafic fluide"

    if is_friday_evening or is_sunday_evening:
        factor = 1.4
        description = "Trafic chargé (retours week-end)"
    elif is_holiday and (is_friday_evening or is_weekend):
        factor = 1.5
        description = "Trafic très dense (départs vacances)"
    elif is_summer and is_weekend:
        factor = 1.3
        description = "Trafic estival"
    elif is_rush_hour and not is_weekend:
        factor = 1.2
        description = "Heure de pointe"
    elif is_weekend:
        factor = 1.1
        description = "Trafic modéré (week-end)"

    return factor, description


def get_week_type(is_holiday: bool, is_summer: bool, is_bridge: bool = False) -> int:
    """
    Détermine le type de semaine pour le modèle de tarification.

    Returns:
        0=normal, 1=vacances, 2=pont, 3=été
    """
    if is_bridge:
        return 2
    if is_holiday:
        return 1
    if is_summer:
        return 3
    return 0


class TripCalculator:
    """
    Calculateur complet pour un trajet.
    """

    def __init__(self):
        pass

    def calculate_trip(
        self,
        origin: str,
        destination: str,
        day_of_week: int = 0,
        hour: int = 10,
        is_holiday: bool = False,
        is_summer: bool = False
    ) -> Dict:
        """
        Calcule toutes les informations d'un trajet.
        """
        # Distance
        distance_info = calculate_distance(origin, destination)

        if "error" in distance_info:
            return distance_info

        # Facteur de trafic
        traffic_factor, traffic_desc = estimate_traffic_factor(
            day_of_week, hour, is_holiday, is_summer
        )

        # Durée ajustée avec trafic
        adjusted_duration = distance_info["duration_hours"] * traffic_factor
        adj_hours = int(adjusted_duration)
        adj_minutes = int((adjusted_duration - adj_hours) * 60)

        # Type de semaine
        week_type = get_week_type(is_holiday, is_summer)

        return {
            **distance_info,
            "traffic_factor": traffic_factor,
            "traffic_description": traffic_desc,
            "adjusted_duration_hours": round(adjusted_duration, 2),
            "adjusted_duration_formatted": f"{adj_hours}h{adj_minutes:02d}",
            "week_type": week_type,
            "day_of_week": day_of_week
        }


if __name__ == "__main__":
    calc = TripCalculator()

    print("=== Test Paris -> Lyon ===")
    result = calc.calculate_trip("Paris", "Lyon", day_of_week=4, hour=17)
    print(result)

    print("\n=== Test Paris -> Marseille (vacances) ===")
    result = calc.calculate_trip("Paris", "Marseille", day_of_week=5, hour=10, is_holiday=True)
    print(result)

    print("\n=== Test Lyon -> Montpellier ===")
    result = calc.calculate_trip("Lyon", "Montpellier")
    print(result)
