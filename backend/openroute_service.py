"""
Billie Green - Service OpenRouteService
Intégration de l'API OpenRouteService pour le calcul précis des distances et itinéraires.
"""

import os
import httpx
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import asyncio
from functools import lru_cache
import json

# Configuration
OPENROUTE_API_KEY = os.environ.get("OPENROUTE_API_KEY", "")
OPENROUTE_BASE_URL = "https://api.openrouteservice.org"

# Cache des géolocalisations pour éviter les appels répétés
_geocode_cache: Dict[str, Tuple[float, float]] = {}


@dataclass
class RouteResult:
    """Résultat d'un calcul d'itinéraire."""
    distance_km: float
    duration_hours: float
    duration_formatted: str
    origin_coords: Tuple[float, float]
    destination_coords: Tuple[float, float]
    geometry: Optional[str] = None  # Polyline encodée
    steps: Optional[List[Dict]] = None
    is_from_api: bool = True


class OpenRouteService:
    """
    Service d'intégration avec OpenRouteService API.

    Fonctionnalités:
    - Géocodage d'adresses (nom de ville → coordonnées)
    - Calcul d'itinéraires avec distance et durée précises
    - Autocomplétion d'adresses
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENROUTE_API_KEY
        self.client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Obtient ou crée le client HTTP."""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": self.api_key,
                    "Content-Type": "application/json"
                }
            )
        return self.client

    async def close(self):
        """Ferme le client HTTP."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()

    def is_configured(self) -> bool:
        """Vérifie si l'API key est configurée."""
        return bool(self.api_key) and len(self.api_key) > 10

    async def geocode(self, place_name: str, country: str = "FR") -> Optional[Tuple[float, float]]:
        """
        Géocode un nom de lieu en coordonnées (lon, lat).

        Args:
            place_name: Nom de la ville ou adresse
            country: Code pays ISO (défaut: FR)

        Returns:
            Tuple (longitude, latitude) ou None si non trouvé
        """
        # Vérifier le cache
        cache_key = f"{place_name.lower()}_{country}"
        if cache_key in _geocode_cache:
            return _geocode_cache[cache_key]

        if not self.is_configured():
            return None

        try:
            client = await self._get_client()

            # API Geocode d'OpenRouteService
            url = f"{OPENROUTE_BASE_URL}/geocode/search"
            params = {
                "api_key": self.api_key,
                "text": place_name,
                "boundary.country": country,
                "size": 1,
                "layers": "locality,localadmin,county"  # Villes uniquement
            }

            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get("features"):
                coords = data["features"][0]["geometry"]["coordinates"]
                result = (coords[0], coords[1])  # (lon, lat)
                _geocode_cache[cache_key] = result
                return result

        except Exception as e:
            print(f"Erreur géocodage '{place_name}': {e}")

        return None

    async def autocomplete(
        self,
        text: str,
        country: str = "FR",
        limit: int = 5
    ) -> List[Dict]:
        """
        Autocomplétion d'adresses/villes.

        Args:
            text: Texte à compléter
            country: Code pays ISO
            limit: Nombre max de résultats

        Returns:
            Liste de suggestions avec nom et coordonnées
        """
        if not self.is_configured() or len(text) < 2:
            return []

        try:
            client = await self._get_client()

            url = f"{OPENROUTE_BASE_URL}/geocode/autocomplete"
            params = {
                "api_key": self.api_key,
                "text": text,
                "boundary.country": country,
                "size": limit,
                "layers": "locality,localadmin,county,region"
            }

            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            results = []
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                coords = feature["geometry"]["coordinates"]

                results.append({
                    "name": props.get("name", ""),
                    "label": props.get("label", ""),
                    "region": props.get("region", ""),
                    "country": props.get("country", "France"),
                    "lon": coords[0],
                    "lat": coords[1]
                })

            return results

        except Exception as e:
            print(f"Erreur autocomplétion '{text}': {e}")
            return []

    async def calculate_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        profile: str = "driving-car"
    ) -> Optional[RouteResult]:
        """
        Calcule un itinéraire entre deux points.

        Args:
            origin: Coordonnées (lon, lat) de départ
            destination: Coordonnées (lon, lat) d'arrivée
            profile: Type de véhicule (driving-car, driving-hgv, cycling-regular, foot-walking)

        Returns:
            RouteResult avec distance et durée
        """
        if not self.is_configured():
            return None

        try:
            client = await self._get_client()

            url = f"{OPENROUTE_BASE_URL}/v2/directions/{profile}"

            payload = {
                "coordinates": [
                    list(origin),
                    list(destination)
                ],
                "instructions": False,
                "geometry": True
            }

            response = await client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()

            if data.get("routes"):
                route = data["routes"][0]
                summary = route.get("summary", {})

                distance_m = summary.get("distance", 0)
                duration_s = summary.get("duration", 0)

                distance_km = distance_m / 1000
                duration_hours = duration_s / 3600

                # Format durée
                hours = int(duration_hours)
                minutes = int((duration_hours - hours) * 60)
                duration_formatted = f"{hours}h{minutes:02d}"

                return RouteResult(
                    distance_km=round(distance_km, 1),
                    duration_hours=round(duration_hours, 2),
                    duration_formatted=duration_formatted,
                    origin_coords=origin,
                    destination_coords=destination,
                    geometry=route.get("geometry"),
                    is_from_api=True
                )

        except httpx.HTTPStatusError as e:
            print(f"Erreur HTTP route: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"Erreur calcul route: {e}")

        return None

    async def calculate_route_by_names(
        self,
        origin_name: str,
        destination_name: str,
        profile: str = "driving-car"
    ) -> Optional[RouteResult]:
        """
        Calcule un itinéraire entre deux villes par leur nom.

        Args:
            origin_name: Nom de la ville de départ
            destination_name: Nom de la ville d'arrivée
            profile: Type de véhicule

        Returns:
            RouteResult avec distance et durée
        """
        # Géocoder les deux villes en parallèle
        origin_coords, dest_coords = await asyncio.gather(
            self.geocode(origin_name),
            self.geocode(destination_name)
        )

        if not origin_coords:
            print(f"Impossible de géocoder l'origine: {origin_name}")
            return None

        if not dest_coords:
            print(f"Impossible de géocoder la destination: {destination_name}")
            return None

        return await self.calculate_route(origin_coords, dest_coords, profile)


# Instance singleton
_service_instance: Optional[OpenRouteService] = None


def get_openroute_service() -> OpenRouteService:
    """Retourne l'instance singleton du service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = OpenRouteService()
    return _service_instance


# Fallback: villes françaises avec coordonnées pré-définies
# Utilisé si l'API OpenRoute n'est pas configurée
FRENCH_CITIES_COORDS = {
    "paris": (2.3522, 48.8566),
    "lyon": (4.8357, 45.7640),
    "marseille": (5.3698, 43.2965),
    "toulouse": (1.4442, 43.6047),
    "nice": (7.2620, 43.7102),
    "nantes": (1.5534, 47.2184),
    "montpellier": (3.8767, 43.6108),
    "strasbourg": (7.7521, 48.5734),
    "bordeaux": (-0.5792, 44.8378),
    "lille": (3.0573, 50.6292),
    "rennes": (-1.6778, 48.1173),
    "grenoble": (5.7245, 45.1885),
    "aix-en-provence": (5.4474, 43.5297),
    "aix": (5.4474, 43.5297),
    "dijon": (5.0415, 47.3220),
    "valence": (4.8920, 44.9333),
    "avignon": (4.8055, 43.9493),
    "toulon": (5.9280, 43.1242),
    "cannes": (7.0174, 43.5528),
    "perpignan": (2.8956, 42.6986),
    "narbonne": (3.0000, 43.1833),
    "beziers": (3.2194, 43.3442),
    "nimes": (4.3601, 43.8367),
    "arles": (4.6278, 43.6767),
    "orange": (4.8083, 44.1361),
    "montelimar": (4.7497, 44.5580),
    "vienne": (4.8758, 45.5247),
    "villefranche": (4.7167, 45.9833),
    "macon": (4.8328, 46.3067),
    "chalon-sur-saone": (4.8536, 46.7806),
    "beaune": (4.8400, 47.0258),
    "auxerre": (3.5714, 47.7989),
    "fontainebleau": (2.7010, 48.4041),
    "salon": (5.0969, 43.6406),
    "salon-de-provence": (5.0969, 43.6406),
    "lancon": (5.1278, 43.5919),
}


def get_city_coords_fallback(city_name: str) -> Optional[Tuple[float, float]]:
    """
    Retourne les coordonnées d'une ville française (fallback sans API).

    Args:
        city_name: Nom de la ville

    Returns:
        Tuple (lon, lat) ou None
    """
    import unicodedata

    # Normaliser le nom
    name = city_name.lower().strip()
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')

    # Aliases
    aliases = {
        "aix-en-provence": "aix",
        "aix en provence": "aix",
        "salon-de-provence": "salon",
        "salon de provence": "salon",
    }
    name = aliases.get(name, name)

    return FRENCH_CITIES_COORDS.get(name)


async def test_service():
    """Test du service OpenRouteService."""
    service = get_openroute_service()

    print(f"API configurée: {service.is_configured()}")

    if service.is_configured():
        # Test autocomplétion
        print("\n=== Test Autocomplétion ===")
        suggestions = await service.autocomplete("Lyon")
        for s in suggestions:
            print(f"  - {s['label']}")

        # Test calcul route
        print("\n=== Test Calcul Route Paris → Lyon ===")
        route = await service.calculate_route_by_names("Paris", "Lyon")
        if route:
            print(f"  Distance: {route.distance_km} km")
            print(f"  Durée: {route.duration_formatted}")
        else:
            print("  Échec du calcul")

    await service.close()


if __name__ == "__main__":
    asyncio.run(test_service())
