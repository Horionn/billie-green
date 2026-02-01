"""
Billie Green - Calculateur CO2 enrichi
Basé sur le modèle GreenGo et les données ADEME

Sources:
- ADEME Base Empreinte
- Impact CO2 (ADEME)
- Méthodologie GreenGo

Périmètre des émissions inclus:
- Émissions directes (combustion)
- Fabrication du véhicule (construction, maintenance, fin de vie)
- Production et distribution du carburant/électricité
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum


# =============================================================================
# CONSTANTES - FACTEURS D'ÉMISSION (gCO2e/km ou gCO2e/pers.km)
# Source: ADEME Impact CO2 - 2024
# =============================================================================

class VehicleType(Enum):
    """Types de véhicules avec leurs facteurs d'émission."""
    # Voitures
    THERMIQUE_ESSENCE = "thermique_essence"
    THERMIQUE_DIESEL = "thermique_diesel"
    HYBRIDE = "hybride"
    HYBRIDE_RECHARGEABLE = "hybride_rechargeable"
    ELECTRIQUE = "electrique"

    # Autres transports (pour comparaison)
    TGV = "tgv"
    TER = "ter"
    INTERCITES = "intercites"
    METRO = "metro"
    BUS = "bus"
    AVION_COURT = "avion_court"
    AVION_MOYEN = "avion_moyen"
    AVION_LONG = "avion_long"
    VELO = "velo"


# Facteurs d'émission en gCO2e/km (pour les voitures)
# ou gCO2e/pers.km (pour les transports en commun)
EMISSION_FACTORS = {
    # Voitures (source: ADEME Impact CO2)
    # Incluent: fabrication + utilisation + fin de vie
    VehicleType.THERMIQUE_DIESEL: {
        "total": 217.6,  # gCO2e/km
        "fabrication": 25.6,  # 12%
        "utilisation": 192.0,  # 88%
        "label": "Voiture diesel"
    },
    VehicleType.THERMIQUE_ESSENCE: {
        "total": 223.0,  # gCO2e/km (légèrement plus que diesel)
        "fabrication": 26.0,
        "utilisation": 197.0,
        "label": "Voiture essence"
    },
    VehicleType.HYBRIDE: {
        "total": 168.0,  # gCO2e/km
        "fabrication": 35.0,
        "utilisation": 133.0,
        "label": "Voiture hybride"
    },
    VehicleType.HYBRIDE_RECHARGEABLE: {
        "total": 120.0,  # gCO2e/km
        "fabrication": 50.0,
        "utilisation": 70.0,
        "label": "Voiture hybride rechargeable"
    },
    VehicleType.ELECTRIQUE: {
        "total": 103.4,  # gCO2e/km
        "fabrication": 83.6,  # 80%
        "utilisation": 19.8,  # 20%
        "label": "Voiture électrique"
    },

    # Transports en commun (source: ADEME Base Empreinte 2021)
    VehicleType.TGV: {
        "total": 2.4,  # gCO2e/pers.km (anciennement 3.3 mis à jour)
        "label": "TGV"
    },
    VehicleType.INTERCITES: {
        "total": 5.9,
        "label": "Intercités"
    },
    VehicleType.TER: {
        "total": 31.7,
        "label": "TER"
    },
    VehicleType.METRO: {
        "total": 4.0,
        "label": "Métro"
    },
    VehicleType.BUS: {
        "total": 29.4,  # Autocar
        "label": "Bus/Autocar"
    },

    # Avion (source: ADEME Impact CO2, incluant traînées)
    VehicleType.AVION_COURT: {
        "total": 258.0,  # < 1000km
        "label": "Avion court courrier"
    },
    VehicleType.AVION_MOYEN: {
        "total": 187.0,  # 1000-3500km
        "label": "Avion moyen courrier"
    },
    VehicleType.AVION_LONG: {
        "total": 152.0,  # > 3500km
        "label": "Avion long courrier"
    },

    # Vélo
    VehicleType.VELO: {
        "total": 0.0,
        "label": "Vélo"
    }
}

# Quota CO2 annuel par personne pour limiter le réchauffement à 2°C
# Source: Accord de Paris / GIEC
QUOTA_CO2_ANNUEL_KG = 2000  # 2 tonnes de CO2e

# Équivalences pour rendre le CO2 plus parlant
# Source: diverses études environnementales
EQUIVALENCES = {
    "arbre_absorption_annuelle_kg": 25,  # Un arbre absorbe ~25kg CO2/an
    "km_tgv_par_kg_co2": 1000 / 2.4,  # km en TGV pour 1kg de CO2
    "km_avion_par_kg_co2": 1000 / 258,  # km en avion court courrier pour 1kg
    "smartphone_charges": 0.005,  # kg CO2 par charge de smartphone
    "repas_boeuf_kg": 7.0,  # kg CO2 par repas avec boeuf
    "repas_vegetarien_kg": 0.5,  # kg CO2 par repas végétarien
    "douche_chaude_kg": 0.5,  # kg CO2 par douche de 5 min
    "netflix_heure_kg": 0.036,  # kg CO2 par heure de streaming
}


@dataclass
class CO2Result:
    """Résultat complet du calcul CO2."""
    # Émissions du trajet
    co2_total_kg: float
    co2_per_person_kg: float
    co2_fabrication_kg: float
    co2_utilisation_kg: float

    # Quota annuel
    quota_percent: float  # % du quota annuel de 2 tonnes

    # Comparaison avec autres transports
    comparison: Dict[str, float]

    # Équivalences parlantes
    equivalences: Dict[str, any]

    # Bus électrique Billie Green (votre service)
    billie_green_co2_kg: float
    co2_saved_vs_car_kg: float
    co2_saved_percent: float

    # Métadonnées
    vehicle_type: str
    distance_km: float
    passengers: int


class CO2Calculator:
    """
    Calculateur d'émissions CO2 pour les trajets.

    Basé sur la méthodologie GreenGo et les données ADEME.
    """

    # Facteur d'émission du bus électrique Billie Green
    # Bus électrique avec voiture sur le toit
    # Estimation: ~50 gCO2e/pers.km (incluant transport de la voiture)
    BILLIE_GREEN_FACTOR = 50.0  # gCO2e/pers.km

    def __init__(self):
        pass

    def get_vehicle_type(self, energy: str, co2_g_km: float = None) -> VehicleType:
        """Détermine le type de véhicule à partir de l'énergie."""
        energy = (energy or "").upper()

        if "ELEC" in energy and not ("ESS" in energy or "GAZ" in energy):
            return VehicleType.ELECTRIQUE
        elif "ELEC" in energy and ("ESS" in energy or "GAZ" in energy):
            # Hybride - vérifier si rechargeable selon CO2
            if co2_g_km and co2_g_km < 100:
                return VehicleType.HYBRIDE_RECHARGEABLE
            return VehicleType.HYBRIDE
        elif "GAZOLE" in energy or "DIESEL" in energy:
            return VehicleType.THERMIQUE_DIESEL
        elif "ESSENCE" in energy or "ESS" in energy:
            return VehicleType.THERMIQUE_ESSENCE
        else:
            # Par défaut diesel (conservateur)
            return VehicleType.THERMIQUE_DIESEL

    def get_emission_factor(self, vehicle_type: VehicleType) -> Dict:
        """Retourne le facteur d'émission pour un type de véhicule."""
        return EMISSION_FACTORS.get(vehicle_type, EMISSION_FACTORS[VehicleType.THERMIQUE_DIESEL])

    def calculate(
        self,
        distance_km: float,
        energy: str,
        passengers: int = 1,
        co2_ademe_g_km: float = None  # CO2 réel ADEME si disponible
    ) -> CO2Result:
        """
        Calcule les émissions CO2 complètes pour un trajet.

        Args:
            distance_km: Distance du trajet en km
            energy: Type d'énergie du véhicule
            passengers: Nombre de passagers
            co2_ademe_g_km: CO2 réel de la base ADEME (optionnel)

        Returns:
            CO2Result avec toutes les métriques
        """
        vehicle_type = self.get_vehicle_type(energy, co2_ademe_g_km)
        emission_data = self.get_emission_factor(vehicle_type)

        # Utiliser le CO2 ADEME si disponible, sinon le facteur standard
        if co2_ademe_g_km is not None and co2_ademe_g_km > 0:
            # Le CO2 ADEME est généralement l'utilisation seule
            # On ajoute la part fabrication selon le ratio du type de véhicule
            base_factor = emission_data["total"]
            fab_ratio = emission_data.get("fabrication", 0) / base_factor if base_factor > 0 else 0.12

            # Reconstituer le total avec fabrication
            co2_factor = co2_ademe_g_km / (1 - fab_ratio) if fab_ratio < 1 else co2_ademe_g_km
            co2_fabrication_g_km = co2_factor * fab_ratio
            co2_utilisation_g_km = co2_ademe_g_km
        else:
            co2_factor = emission_data["total"]
            co2_fabrication_g_km = emission_data.get("fabrication", 0)
            co2_utilisation_g_km = emission_data.get("utilisation", co2_factor)

        # Calcul des émissions totales
        co2_total_kg = (co2_factor * distance_km) / 1000
        co2_per_person_kg = co2_total_kg / passengers
        co2_fabrication_kg = (co2_fabrication_g_km * distance_km) / 1000
        co2_utilisation_kg = (co2_utilisation_g_km * distance_km) / 1000

        # Quota annuel
        quota_percent = (co2_per_person_kg / QUOTA_CO2_ANNUEL_KG) * 100

        # Comparaison avec autres transports
        comparison = self._calculate_comparison(distance_km, passengers)

        # Équivalences parlantes
        equivalences = self._calculate_equivalences(co2_per_person_kg)

        # Billie Green (bus électrique)
        billie_green_co2_kg = (self.BILLIE_GREEN_FACTOR * distance_km) / 1000
        co2_saved_vs_car_kg = co2_per_person_kg - billie_green_co2_kg
        co2_saved_percent = (co2_saved_vs_car_kg / co2_per_person_kg * 100) if co2_per_person_kg > 0 else 0

        return CO2Result(
            co2_total_kg=round(co2_total_kg, 2),
            co2_per_person_kg=round(co2_per_person_kg, 2),
            co2_fabrication_kg=round(co2_fabrication_kg, 2),
            co2_utilisation_kg=round(co2_utilisation_kg, 2),
            quota_percent=round(quota_percent, 1),
            comparison=comparison,
            equivalences=equivalences,
            billie_green_co2_kg=round(billie_green_co2_kg, 2),
            co2_saved_vs_car_kg=round(max(0, co2_saved_vs_car_kg), 2),
            co2_saved_percent=round(max(0, co2_saved_percent), 1),
            vehicle_type=emission_data["label"],
            distance_km=distance_km,
            passengers=passengers
        )

    def _calculate_comparison(self, distance_km: float, passengers: int) -> Dict[str, float]:
        """Calcule les émissions pour les transports alternatifs."""
        comparison = {}

        alternatives = [
            (VehicleType.TGV, "tgv"),
            (VehicleType.TER, "ter"),
            (VehicleType.BUS, "bus"),
            (VehicleType.AVION_COURT, "avion"),
            (VehicleType.ELECTRIQUE, "voiture_electrique"),
        ]

        for vehicle_type, key in alternatives:
            factor = EMISSION_FACTORS[vehicle_type]["total"]
            # Pour les transports en commun, c'est déjà par personne
            # Pour la voiture électrique, diviser par le nombre de passagers
            if vehicle_type == VehicleType.ELECTRIQUE:
                co2_kg = (factor * distance_km) / 1000 / passengers
            else:
                co2_kg = (factor * distance_km) / 1000
            comparison[key] = round(co2_kg, 2)

        # Ajouter Billie Green
        comparison["billie_green"] = round((self.BILLIE_GREEN_FACTOR * distance_km) / 1000, 2)

        return comparison

    def _calculate_equivalences(self, co2_kg: float) -> Dict:
        """Calcule les équivalences parlantes pour le CO2."""
        return {
            # Arbres nécessaires pour absorber ce CO2 en 1 an
            "arbres_annee": round(co2_kg / EQUIVALENCES["arbre_absorption_annuelle_kg"], 1),

            # Équivalent en km TGV
            "km_tgv_equivalent": round(co2_kg * EQUIVALENCES["km_tgv_par_kg_co2"], 0),

            # Équivalent en charges de smartphone
            "charges_smartphone": round(co2_kg / EQUIVALENCES["smartphone_charges"], 0),

            # Équivalent en repas avec boeuf
            "repas_boeuf": round(co2_kg / EQUIVALENCES["repas_boeuf_kg"], 1),

            # Équivalent en heures de Netflix
            "heures_netflix": round(co2_kg / EQUIVALENCES["netflix_heure_kg"], 0),

            # Équivalent en douches chaudes
            "douches_chaudes": round(co2_kg / EQUIVALENCES["douche_chaude_kg"], 0),

            # Message personnalisé
            "message": self._get_equivalence_message(co2_kg)
        }

    def _get_equivalence_message(self, co2_kg: float) -> str:
        """Génère un message personnalisé selon le niveau de CO2."""
        arbres = co2_kg / EQUIVALENCES["arbre_absorption_annuelle_kg"]

        if co2_kg < 5:
            return f"🌱 Faible impact : équivaut à {arbres:.1f} arbre pendant 1 an"
        elif co2_kg < 20:
            return f"🌳 Impact modéré : équivaut à {arbres:.1f} arbres pendant 1 an"
        elif co2_kg < 50:
            return f"🌲 Impact significatif : {arbres:.0f} arbres seraient nécessaires"
        else:
            return f"🏭 Impact important : {arbres:.0f} arbres seraient nécessaires pour compenser"

    def get_savings_message(self, co2_saved_kg: float, co2_saved_percent: float) -> Dict:
        """Génère des messages sur les économies réalisées."""
        arbres = co2_saved_kg / EQUIVALENCES["arbre_absorption_annuelle_kg"]

        return {
            "headline": f"🌿 Vous économisez {co2_saved_kg:.1f} kg de CO₂",
            "percent": f"Soit {co2_saved_percent:.0f}% d'émissions en moins",
            "trees": f"Équivalent à {arbres:.1f} arbre(s) planté(s)",
            "quota": f"Vous préservez {(co2_saved_kg / QUOTA_CO2_ANNUEL_KG * 100):.1f}% de votre quota annuel",
            "icon": "🌳" if arbres >= 1 else "🌱"
        }


# Instance singleton
_calculator: Optional[CO2Calculator] = None


def get_co2_calculator() -> CO2Calculator:
    """Retourne l'instance singleton du calculateur."""
    global _calculator
    if _calculator is None:
        _calculator = CO2Calculator()
    return _calculator


# Test
if __name__ == "__main__":
    calc = get_co2_calculator()

    # Test Paris-Lyon (465 km) avec une voiture diesel
    result = calc.calculate(
        distance_km=465,
        energy="GAZOLE",
        passengers=1,
        co2_ademe_g_km=150
    )

    print("=== Trajet Paris-Lyon (465 km) - Voiture Diesel ===")
    print(f"CO2 total: {result.co2_total_kg} kg")
    print(f"CO2 par personne: {result.co2_per_person_kg} kg")
    print(f"Quota annuel: {result.quota_percent}%")
    print(f"\nComparaison:")
    for mode, co2 in result.comparison.items():
        print(f"  - {mode}: {co2} kg")
    print(f"\nÉquivalences:")
    print(f"  - {result.equivalences['arbres_annee']} arbres pendant 1 an")
    print(f"  - {result.equivalences['km_tgv_equivalent']} km en TGV")
    print(f"\nAvec Billie Green:")
    print(f"  - CO2: {result.billie_green_co2_kg} kg")
    print(f"  - Économie: {result.co2_saved_vs_car_kg} kg ({result.co2_saved_percent}%)")
