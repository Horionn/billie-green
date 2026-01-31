"""
Billie Green - Calculateur de coûts pour camions Symone (biogaz)
Calcule le coût total d'un trajet : péage + carburant biogaz
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TruckSpecs:
    """Spécifications d'un camion Symone."""
    name: str = "Camion Symone Biogaz"
    fuel_type: str = "BIOGAZ"
    # Consommation moyenne en kg de biogaz / 100 km
    consumption_per_100km: float = 25.0  # ~25 kg/100km pour un camion au biogaz
    # Prix du biogaz (€/kg) - production interne Symone
    biogaz_price_per_kg: float = 0.85  # Compétitif vs diesel (~1.60€/L)
    # Catégorie de péage (classe 4 pour poids lourds >3.5t)
    toll_class: int = 4
    # Émissions CO2 (biogaz = quasi neutre, mais on compte les émissions résiduelles)
    co2_g_km: float = 15.0  # Très faible vs diesel (600-800 g/km)


# Tarifs péages autoroutiers pour poids lourds (classe 4)
# Source : tarifs moyens 2024-2025 des sociétés concessionnaires
# Format : (origine, destination) : prix en € pour classe 4
TRUCK_TOLL_PRICES = {
    # Paris - Lyon
    ("paris", "lyon"): 62.80,
    
    # Paris - Sud
    ("paris", "marseille"): 105.40,
    ("paris", "aix"): 103.20,
    ("paris", "montpellier"): 104.60,
    ("paris", "nice"): 127.80,
    ("paris", "avignon"): 93.40,
    ("paris", "toulon"): 114.80,
    ("paris", "cannes"): 123.50,
    
    # Paris - Est
    ("paris", "dijon"): 42.50,
    ("paris", "grenoble"): 78.60,
    ("paris", "valence"): 76.50,
    
    # Lyon - Sud
    ("lyon", "marseille"): 42.90,
    ("lyon", "aix"): 40.80,
    ("lyon", "montpellier"): 41.40,
    ("lyon", "nice"): 64.60,
    ("lyon", "avignon"): 31.40,
    ("lyon", "toulon"): 51.70,
    ("lyon", "cannes"): 60.50,
    ("lyon", "valence"): 13.70,
    ("lyon", "grenoble"): 15.20,
    
    # Lyon - autres
    ("lyon", "dijon"): 26.30,
    
    # Marseille - destinations
    ("marseille", "montpellier"): 23.10,
    ("marseille", "nice"): 27.10,
    ("marseille", "aix"): 4.20,
    ("marseille", "avignon"): 13.50,
    ("marseille", "toulon"): 9.00,
    ("marseille", "cannes"): 24.30,
    ("marseille", "valence"): 29.40,
    ("marseille", "grenoble"): 41.80,
    
    # Grenoble - destinations
    ("grenoble", "nice"): 45.90,
    ("grenoble", "montpellier"): 40.30,
    ("grenoble", "aix"): 39.60,
    ("grenoble", "avignon"): 32.10,
    ("grenoble", "valence"): 13.00,
    ("grenoble", "dijon"): 40.90,
    
    # Montpellier - destinations
    ("montpellier", "nice"): 45.00,
    ("montpellier", "aix"): 20.90,
    ("montpellier", "avignon"): 12.60,
    ("montpellier", "toulon"): 31.90,
    ("montpellier", "cannes"): 41.40,
    
    # Nice - destinations
    ("nice", "aix"): 24.20,
    ("nice", "avignon"): 35.00,
    ("nice", "toulon"): 20.50,
    ("nice", "cannes"): 4.50,
    
    # Autres trajets importants
    ("dijon", "lyon"): 26.30,
    ("valence", "orange"): 12.30,
    ("orange", "marseille"): 13.00,
    ("avignon", "aix"): 11.20,
    ("toulon", "cannes"): 16.80,
}


class SymoneTruckCalculator:
    """
    Calculateur de coûts pour les camions Symone au biogaz.
    """
    
    def __init__(self, truck_specs: Optional[TruckSpecs] = None):
        """
        Args:
            truck_specs: Spécifications du camion (utilise les valeurs par défaut si None)
        """
        self.truck = truck_specs or TruckSpecs()
    
    def _normalize_city(self, city: str) -> str:
        """Normalise le nom de ville pour la recherche de péage."""
        import unicodedata
        city = city.lower().strip()
        city = unicodedata.normalize('NFD', city)
        city = ''.join(c for c in city if unicodedata.category(c) != 'Mn')
        
        aliases = {
            "aix-en-provence": "aix",
            "aix en provence": "aix",
        }
        return aliases.get(city, city)
    
    def get_toll_price(self, origin: str, destination: str) -> Optional[float]:
        """
        Récupère le prix du péage pour poids lourds.
        
        Args:
            origin: Ville de départ
            destination: Ville d'arrivée
            
        Returns:
            Prix du péage en € ou None si non disponible
        """
        o = self._normalize_city(origin)
        d = self._normalize_city(destination)
        
        # Chercher dans les deux sens
        if (o, d) in TRUCK_TOLL_PRICES:
            return TRUCK_TOLL_PRICES[(o, d)]
        if (d, o) in TRUCK_TOLL_PRICES:
            return TRUCK_TOLL_PRICES[(d, o)]
        
        return None
    
    def estimate_toll_price(self, distance_km: float) -> float:
        """
        Estime le prix du péage basé sur la distance si le prix exact n'est pas disponible.
        
        Formule : ~0.136 €/km pour classe 4 (moyenne constatée sur les autoroutes françaises)
        
        Args:
            distance_km: Distance du trajet en km
            
        Returns:
            Prix estimé du péage en €
        """
        return distance_km * 0.136
    
    def calculate_fuel_cost(self, distance_km: float) -> Dict[str, float]:
        """
        Calcule le coût en carburant biogaz.
        
        Args:
            distance_km: Distance du trajet en km
            
        Returns:
            Dict avec détails de consommation et coût
        """
        # Consommation totale en kg
        fuel_kg = (distance_km / 100) * self.truck.consumption_per_100km
        
        # Coût total
        fuel_cost = fuel_kg * self.truck.biogaz_price_per_kg
        
        return {
            "fuel_consumption_kg": round(fuel_kg, 2),
            "fuel_price_per_kg": self.truck.biogaz_price_per_kg,
            "fuel_cost_euros": round(fuel_cost, 2),
            "consumption_per_100km": self.truck.consumption_per_100km
        }
    
    def calculate_trip_cost(
        self,
        origin: str,
        destination: str,
        distance_km: float
    ) -> Dict:
        """
        Calcule le coût total d'un trajet (péage + carburant).
        
        Args:
            origin: Ville de départ
            destination: Ville d'arrivée
            distance_km: Distance en km
            
        Returns:
            Dict avec tous les détails de coût
        """
        # 1. Péage
        toll_price = self.get_toll_price(origin, destination)
        is_exact_toll = toll_price is not None
        
        if toll_price is None:
            toll_price = self.estimate_toll_price(distance_km)
        
        # 2. Carburant
        fuel_details = self.calculate_fuel_cost(distance_km)
        
        # 3. Total
        total_cost = toll_price + fuel_details["fuel_cost_euros"]
        
        # 4. Calcul du coût par km
        cost_per_km = total_cost / distance_km if distance_km > 0 else 0
        
        # 5. Émissions CO2 totales
        co2_total_kg = (distance_km * self.truck.co2_g_km) / 1000
        
        # 6. Comparaison avec diesel (optionnel)
        # Diesel : ~0.30€/km (péage + carburant) pour PL
        diesel_equivalent_cost = distance_km * 0.30
        savings_vs_diesel = diesel_equivalent_cost - total_cost
        savings_percent = (savings_vs_diesel / diesel_equivalent_cost * 100) if diesel_equivalent_cost > 0 else 0
        
        return {
            "truck_info": {
                "name": self.truck.name,
                "fuel_type": self.truck.fuel_type,
                "co2_g_km": self.truck.co2_g_km,
            },
            "trip": {
                "origin": origin,
                "destination": destination,
                "distance_km": distance_km,
            },
            "toll": {
                "price_euros": round(toll_price, 2),
                "is_exact": is_exact_toll,
                "description": "Tarif réel classe 4" if is_exact_toll else "Tarif estimé (0.136€/km)"
            },
            "fuel": fuel_details,
            "total_cost_euros": round(total_cost, 2),
            "cost_per_km_euros": round(cost_per_km, 3),
            "environmental": {
                "co2_total_kg": round(co2_total_kg, 2),
                "co2_per_km_g": self.truck.co2_g_km,
                "fuel_type": "Biogaz (renouvelable)"
            },
            "comparison": {
                "diesel_equivalent_cost": round(diesel_equivalent_cost, 2),
                "savings_vs_diesel_euros": round(savings_vs_diesel, 2),
                "savings_percent": round(savings_percent, 1)
            },
            "breakdown": [
                {
                    "item": "Péage autoroute",
                    "amount": round(toll_price, 2),
                    "unit": "€"
                },
                {
                    "item": f"Carburant biogaz ({fuel_details['fuel_consumption_kg']} kg)",
                    "amount": fuel_details["fuel_cost_euros"],
                    "unit": "€"
                },
                {
                    "item": "TOTAL",
                    "amount": round(total_cost, 2),
                    "unit": "€"
                }
            ]
        }


def calculate_symone_truck_cost(
    origin: str,
    destination: str,
    distance_km: float,
    custom_consumption: Optional[float] = None,
    custom_biogaz_price: Optional[float] = None
) -> Dict:
    """
    Fonction helper pour calculer rapidement le coût d'un trajet.
    
    Args:
        origin: Ville de départ
        destination: Ville d'arrivée
        distance_km: Distance en km
        custom_consumption: Consommation personnalisée (kg/100km)
        custom_biogaz_price: Prix personnalisé du biogaz (€/kg)
    
    Returns:
        Dict avec les détails de coût
    """
    specs = TruckSpecs()
    
    if custom_consumption is not None:
        specs.consumption_per_100km = custom_consumption
    
    if custom_biogaz_price is not None:
        specs.biogaz_price_per_kg = custom_biogaz_price
    
    calculator = SymoneTruckCalculator(specs)
    return calculator.calculate_trip_cost(origin, destination, distance_km)


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    calculator = SymoneTruckCalculator()
    
    print("=" * 80)
    print("CALCULATEUR DE COÛTS CAMION SYMONE (BIOGAZ)")
    print("=" * 80)
    print()
    
    # Test 1: Paris -> Lyon
    print("=== Test 1: Paris -> Lyon (462 km) ===")
    result = calculator.calculate_trip_cost("Paris", "Lyon", 462)
    print(f"Péage: {result['toll']['price_euros']}€ ({result['toll']['description']})")
    print(f"Carburant: {result['fuel']['fuel_consumption_kg']} kg × {result['fuel']['fuel_price_per_kg']}€/kg = {result['fuel']['fuel_cost_euros']}€")
    print(f"TOTAL: {result['total_cost_euros']}€ ({result['cost_per_km_euros']}€/km)")
    print(f"Économie vs diesel: {result['comparison']['savings_vs_diesel_euros']}€ ({result['comparison']['savings_percent']}%)")
    print(f"CO2 émis: {result['environmental']['co2_total_kg']} kg")
    print()
    
    # Test 2: Paris -> Marseille
    print("=== Test 2: Paris -> Marseille (774 km) ===")
    result = calculator.calculate_trip_cost("Paris", "Marseille", 774)
    print(f"Péage: {result['toll']['price_euros']}€")
    print(f"Carburant: {result['fuel']['fuel_cost_euros']}€")
    print(f"TOTAL: {result['total_cost_euros']}€")
    print(f"Économie vs diesel: {result['comparison']['savings_vs_diesel_euros']}€")
    print()
    
    # Test 3: Lyon -> Marseille
    print("=== Test 3: Lyon -> Marseille (314 km) ===")
    result = calculator.calculate_trip_cost("Lyon", "Marseille", 314)
    print(f"Péage: {result['toll']['price_euros']}€")
    print(f"Carburant: {result['fuel']['fuel_cost_euros']}€")
    print(f"TOTAL: {result['total_cost_euros']}€")
    print()
    
    # Test 4: Trajet sans péage exact (estimation)
    print("=== Test 4: Trajet avec estimation péage (200 km) ===")
    result = calculator.calculate_trip_cost("Fleury", "Auxerre", 200)
    print(f"Péage: {result['toll']['price_euros']}€ (estimé)")
    print(f"Carburant: {result['fuel']['fuel_cost_euros']}€")
    print(f"TOTAL: {result['total_cost_euros']}€")
    print()
    
    print("=" * 80)
    print("✓ Tests terminés")
    print("=" * 80)
