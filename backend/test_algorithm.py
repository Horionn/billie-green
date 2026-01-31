"""
Billie Green - Tests exhaustifs de l'algorithme de tarification
Génère un CSV avec tous les cas de test
"""

import csv
from pathlib import Path

# ============================================
# ALGORITHME DE SCORING (copié de main.py)
# ============================================

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


def calculate_eco_score(energy: str, model: str) -> tuple:
    """
    Calcule le score écologique (0 = polluant, 1 = propre)

    LOGIQUE:
    - Score 0.95 = Électrique (zéro émission)
    - Score 0.65 = Hybride (émissions réduites)
    - Score 0.40 = Essence (émissions moyennes)
    - Score 0.35 = Diesel/Gazole (émissions élevées)
    - Score 0.45 = Défaut (inconnu)

    Returns: (score, explication)
    """
    energy = (energy or '').upper()
    model = (model or '').upper()

    # Détection véhicule électrique
    is_electric = 'ELEC' in energy or any(m in model for m in ELECTRIC_MODELS)

    if is_electric:
        return 0.95, "Véhicule électrique détecté (énergie ou modèle connu)"

    energy_scores = {
        'HYBRIDE': (0.65, "Hybride - émissions réduites"),
        'ESS+ELEC': (0.65, "Hybride essence-électrique"),
        'GAZ+ELEC': (0.65, "Hybride gazole-électrique"),
        'ESSENCE': (0.40, "Essence - émissions moyennes"),
        'GAZOLE': (0.35, "Diesel/Gazole - émissions élevées"),
        'DIESEL': (0.35, "Diesel - émissions élevées"),
    }

    for key, (score, expl) in energy_scores.items():
        if key in energy:
            return score, expl

    return 0.45, "Type d'énergie inconnu - score par défaut"


def calculate_social_score(argus_value: float) -> tuple:
    """
    Calcule la capacité contributive (0 = protégé/faible, 1 = élevée)

    LOGIQUE:
    - Basé sur la valeur ARGUS (prix de revente du véhicule)
    - ARGUS = proxy du niveau socio-économique
    - Seuil max: 35 000€ → score = 1.0
    - Proportionnel: score = argus / 35000

    Returns: (score, explication, catégorie)
    """
    if argus_value is None or argus_value <= 0:
        argus_value = 10000  # Défaut

    score = min(argus_value / 35000, 1.0)

    if score <= 0.3:
        cat = "FAIBLE"
        expl = f"ARGUS {argus_value}€ ≤ 10 500€ → Capacité limitée, PROTÉGÉ"
    elif score <= 0.6:
        cat = "MOYENNE"
        expl = f"ARGUS {argus_value}€ entre 10 500€ et 21 000€ → Capacité moyenne"
    else:
        cat = "ÉLEVÉE"
        expl = f"ARGUS {argus_value}€ > 21 000€ → Capacité contributive élevée"

    return score, expl, cat


def calculate_ethical_adjustment(eco_score: float, social_score: float) -> tuple:
    """
    Calcule l'ajustement éthique.

    MATRICE DE DÉCISION:
    ┌─────────────────┬──────────────┬──────────────┬──────────────┐
    │                 │ Social FAIBLE│ Social MOYEN │ Social ÉLEVÉ │
    │                 │   (≤0.3)     │  (0.3-0.6)   │   (>0.6)     │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ Eco PROPRE      │   -12%       │    -12%      │    -12%      │
    │ (≥0.7)          │   BONUS      │    BONUS     │    BONUS     │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ Eco MOYEN       │    0%        │     0%       │     0%       │
    │ (0.45-0.7)      │   NEUTRE     │    NEUTRE    │    NEUTRE    │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ Eco POLLUANT    │    0%        │    +5%       │   +15%       │
    │ (<0.45)         │   PROTÉGÉ    │    MALUS     │    MALUS     │
    └─────────────────┴──────────────┴──────────────┴──────────────┘

    Returns: (adjustment, label, detail)
    """
    # Véhicule propre → toujours bonus
    if eco_score >= 0.7:
        return -0.12, "Bonus écologique", "Véhicule à faibles émissions → -12%"

    # Véhicule polluant
    if eco_score < 0.45:
        if social_score > 0.6:
            return 0.15, "Contribution écologique", "Polluant + Capacité ÉLEVÉE → +15%"
        elif social_score > 0.3:
            return 0.05, "Contribution écologique modérée", "Polluant + Capacité MOYENNE → +5%"
        else:
            return 0.0, "Malus exonéré", "Polluant + Capacité FAIBLE → PROTÉGÉ (0%)"

    # Véhicule moyen
    return 0.0, "Tarification standard", "Émissions moyennes → pas d'ajustement"


def calculate_full_price(
    distance_km: float,
    energy: str,
    model: str,
    argus_value: float,
    passengers: int = 1
) -> dict:
    """Calcule le prix complet avec tous les détails."""

    # 1. Scores
    eco_score, eco_expl = calculate_eco_score(energy, model)
    social_score, social_expl, social_cat = calculate_social_score(argus_value)

    # 2. Prix de base (0.15€/km, minimum 20€)
    base_price = max(distance_km * 0.15, 20)

    # 3. Ajustement éthique
    adjustment, adj_label, adj_detail = calculate_ethical_adjustment(eco_score, social_score)

    # 4. Bonus covoiturage (5% par passager supplémentaire)
    passenger_bonus = (passengers - 1) * 0.05

    # 5. Prix final
    price_after_eco = base_price * (1 + adjustment)
    final_price = price_after_eco * (1 - passenger_bonus)

    return {
        "distance_km": distance_km,
        "energy": energy,
        "model": model,
        "argus_value": argus_value,
        "passengers": passengers,
        "eco_score": round(eco_score, 2),
        "eco_explication": eco_expl,
        "social_score": round(social_score, 2),
        "social_explication": social_expl,
        "social_categorie": social_cat,
        "base_price": round(base_price, 2),
        "adjustment_percent": round(adjustment * 100, 1),
        "adjustment_label": adj_label,
        "adjustment_detail": adj_detail,
        "passenger_bonus_percent": round(passenger_bonus * 100, 1),
        "price_after_eco": round(price_after_eco, 2),
        "final_price": round(final_price, 2),
    }


# ============================================
# CAS DE TEST EXHAUSTIFS
# ============================================

TEST_CASES = [
    # === VÉHICULES ÉLECTRIQUES ===
    # Électrique riche
    {"desc": "Tesla Model 3 - ARGUS 40k€ (riche)", "distance": 300, "energy": "ELECTRIQUE", "model": "MODEL 3", "argus": 40000, "passengers": 1},
    {"desc": "Tesla Model S - ARGUS 55k€ (très riche)", "distance": 300, "energy": "ELECTRIQUE", "model": "MODEL S", "argus": 55000, "passengers": 1},
    # Électrique classe moyenne
    {"desc": "Renault Zoé - ARGUS 12k€ (moyen)", "distance": 300, "energy": "ELECTRIQUE", "model": "ZOE", "argus": 12000, "passengers": 1},
    {"desc": "Dacia Spring - ARGUS 8k€ (modeste)", "distance": 300, "energy": "ELECTRIQUE", "model": "SPRING", "argus": 8000, "passengers": 1},
    # Électrique pauvre (occasion)
    {"desc": "Nissan Leaf occasion - ARGUS 5k€ (faible)", "distance": 300, "energy": "ELECTRIQUE", "model": "LEAF", "argus": 5000, "passengers": 1},

    # === VÉHICULES HYBRIDES ===
    {"desc": "Toyota Prius Hybride - ARGUS 25k€", "distance": 300, "energy": "ESS+ELEC", "model": "PRIUS", "argus": 25000, "passengers": 1},
    {"desc": "Toyota Yaris Hybride - ARGUS 15k€", "distance": 300, "energy": "HYBRIDE", "model": "YARIS", "argus": 15000, "passengers": 1},
    {"desc": "Hybride occasion - ARGUS 6k€", "distance": 300, "energy": "HYBRIDE", "model": "AURIS", "argus": 6000, "passengers": 1},

    # === ESSENCE - DIFFÉRENTES CAPACITÉS ===
    # Essence riche (grosse cylindrée, neuf)
    {"desc": "BMW M3 Essence - ARGUS 60k€ (riche)", "distance": 300, "energy": "ESSENCE", "model": "M3", "argus": 60000, "passengers": 1},
    {"desc": "Audi A4 Essence - ARGUS 30k€ (aisé)", "distance": 300, "energy": "ESSENCE", "model": "A4", "argus": 30000, "passengers": 1},
    # Essence classe moyenne
    {"desc": "Peugeot 308 Essence - ARGUS 18k€ (moyen)", "distance": 300, "energy": "ESSENCE", "model": "308", "argus": 18000, "passengers": 1},
    {"desc": "Renault Clio Essence - ARGUS 12k€ (moyen-)", "distance": 300, "energy": "ESSENCE", "model": "CLIO", "argus": 12000, "passengers": 1},
    # Essence modeste/pauvre
    {"desc": "Citroën C3 Essence - ARGUS 8k€ (modeste)", "distance": 300, "energy": "ESSENCE", "model": "C3", "argus": 8000, "passengers": 1},
    {"desc": "Vieille Twingo Essence - ARGUS 3k€ (faible)", "distance": 300, "energy": "ESSENCE", "model": "TWINGO", "argus": 3000, "passengers": 1},

    # === DIESEL - DIFFÉRENTES CAPACITÉS ===
    # Diesel riche (SUV premium)
    {"desc": "BMW X5 Diesel - ARGUS 45k€ (riche)", "distance": 300, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 1},
    {"desc": "Mercedes GLC Diesel - ARGUS 38k€ (aisé)", "distance": 300, "energy": "DIESEL", "model": "GLC", "argus": 38000, "passengers": 1},
    {"desc": "Audi Q5 Diesel - ARGUS 28k€ (aisé)", "distance": 300, "energy": "GAZOLE", "model": "Q5", "argus": 28000, "passengers": 1},
    # Diesel classe moyenne
    {"desc": "Peugeot 3008 Diesel - ARGUS 20k€ (moyen+)", "distance": 300, "energy": "GAZOLE", "model": "3008", "argus": 20000, "passengers": 1},
    {"desc": "Renault Megane Diesel - ARGUS 14k€ (moyen)", "distance": 300, "energy": "GAZOLE", "model": "MEGANE", "argus": 14000, "passengers": 1},
    # Diesel modeste/pauvre - CAS CRITIQUE: DOIT ÊTRE PROTÉGÉ
    {"desc": "Vieille Clio Diesel - ARGUS 7k€ (modeste)", "distance": 300, "energy": "GAZOLE", "model": "CLIO", "argus": 7000, "passengers": 1},
    {"desc": "Vieux Scenic Diesel - ARGUS 4k€ (faible)", "distance": 300, "energy": "DIESEL", "model": "SCENIC", "argus": 4000, "passengers": 1},
    {"desc": "Vieille 206 Diesel - ARGUS 2k€ (très faible)", "distance": 300, "energy": "GAZOLE", "model": "206", "argus": 2000, "passengers": 1},

    # === TESTS COVOITURAGE ===
    {"desc": "Diesel riche - 1 passager", "distance": 300, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 1},
    {"desc": "Diesel riche - 2 passagers", "distance": 300, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 2},
    {"desc": "Diesel riche - 4 passagers", "distance": 300, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 4},
    {"desc": "Électrique - 1 passager", "distance": 300, "energy": "ELECTRIQUE", "model": "MODEL 3", "argus": 35000, "passengers": 1},
    {"desc": "Électrique - 3 passagers", "distance": 300, "energy": "ELECTRIQUE", "model": "MODEL 3", "argus": 35000, "passengers": 3},

    # === TESTS DISTANCES ===
    {"desc": "Court trajet (50km) - Diesel riche", "distance": 50, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 1},
    {"desc": "Trajet moyen (200km) - Diesel riche", "distance": 200, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 1},
    {"desc": "Long trajet (500km) - Diesel riche", "distance": 500, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 1},
    {"desc": "Très long trajet (800km) - Diesel riche", "distance": 800, "energy": "GAZOLE", "model": "X5", "argus": 45000, "passengers": 1},
    {"desc": "Court trajet (50km) - Diesel pauvre", "distance": 50, "energy": "GAZOLE", "model": "SCENIC", "argus": 4000, "passengers": 1},
    {"desc": "Trajet moyen (200km) - Diesel pauvre", "distance": 200, "energy": "GAZOLE", "model": "SCENIC", "argus": 4000, "passengers": 1},
    {"desc": "Long trajet (500km) - Diesel pauvre", "distance": 500, "energy": "GAZOLE", "model": "SCENIC", "argus": 4000, "passengers": 1},

    # === CAS LIMITES ARGUS ===
    {"desc": "ARGUS seuil bas (10 500€) - Diesel", "distance": 300, "energy": "GAZOLE", "model": "MEGANE", "argus": 10500, "passengers": 1},
    {"desc": "ARGUS seuil haut (21 000€) - Diesel", "distance": 300, "energy": "GAZOLE", "model": "3008", "argus": 21000, "passengers": 1},
    {"desc": "ARGUS max (35 000€) - Diesel", "distance": 300, "energy": "GAZOLE", "model": "X5", "argus": 35000, "passengers": 1},
    {"desc": "ARGUS très élevé (70 000€) - Diesel", "distance": 300, "energy": "GAZOLE", "model": "X7", "argus": 70000, "passengers": 1},

    # === CAS SPÉCIAUX ===
    {"desc": "Énergie inconnue - ARGUS moyen", "distance": 300, "energy": "", "model": "INCONNU", "argus": 15000, "passengers": 1},
    {"desc": "Énergie inconnue - ARGUS faible", "distance": 300, "energy": "", "model": "INCONNU", "argus": 5000, "passengers": 1},
    {"desc": "Modèle électrique sans énergie déclarée", "distance": 300, "energy": "", "model": "MODEL 3", "argus": 35000, "passengers": 1},
]


def run_tests():
    """Exécute tous les tests et génère le CSV."""

    results = []

    print("=" * 80)
    print("TESTS EXHAUSTIFS DE L'ALGORITHME BILLIE GREEN")
    print("=" * 80)
    print()

    for i, test in enumerate(TEST_CASES, 1):
        result = calculate_full_price(
            distance_km=test["distance"],
            energy=test["energy"],
            model=test["model"],
            argus_value=test["argus"],
            passengers=test["passengers"]
        )

        result["test_description"] = test["desc"]
        results.append(result)

        # Affichage console
        print(f"[{i:02d}] {test['desc']}")
        print(f"     Distance: {test['distance']}km | Énergie: {test['energy'] or 'INCONNU'} | ARGUS: {test['argus']}€ | Passagers: {test['passengers']}")
        print(f"     → Eco Score: {result['eco_score']} ({result['eco_explication']})")
        print(f"     → Social Score: {result['social_score']} - {result['social_categorie']} ({result['social_explication']})")
        print(f"     → Ajustement: {result['adjustment_percent']:+.1f}% ({result['adjustment_label']})")
        print(f"     → Prix: {result['base_price']}€ base → {result['final_price']}€ final")
        print()

    # Générer le CSV
    output_path = Path(__file__).parent.parent / "tests_algorithme.csv"

    fieldnames = [
        "test_description",
        "distance_km", "energy", "model", "argus_value", "passengers",
        "eco_score", "eco_explication",
        "social_score", "social_categorie", "social_explication",
        "base_price",
        "adjustment_percent", "adjustment_label", "adjustment_detail",
        "passenger_bonus_percent",
        "price_after_eco", "final_price"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(results)

    print("=" * 80)
    print(f"✓ CSV généré: {output_path}")
    print(f"✓ {len(results)} tests exécutés")
    print("=" * 80)

    # Résumé des catégories
    print("\n=== RÉSUMÉ PAR CATÉGORIE ===\n")

    # Véhicules électriques
    elec = [r for r in results if r['eco_score'] >= 0.7]
    print(f"ÉLECTRIQUES/PROPRES (eco ≥ 0.7): {len(elec)} tests")
    print(f"  → Tous reçoivent -12% de bonus")

    # Hybrides
    hybrid = [r for r in results if 0.6 <= r['eco_score'] < 0.7]
    print(f"\nHYBRIDES (0.6 ≤ eco < 0.7): {len(hybrid)} tests")
    print(f"  → Tarification standard (0%)")

    # Polluants protégés
    protected = [r for r in results if r['eco_score'] < 0.45 and r['social_score'] <= 0.3]
    print(f"\nPOLLUANTS PROTÉGÉS (eco < 0.45 ET social ≤ 0.3): {len(protected)} tests")
    print(f"  → Pas de malus malgré pollution (protection sociale)")

    # Polluants moyens
    medium = [r for r in results if r['eco_score'] < 0.45 and 0.3 < r['social_score'] <= 0.6]
    print(f"\nPOLLUANTS MOYENS (eco < 0.45 ET 0.3 < social ≤ 0.6): {len(medium)} tests")
    print(f"  → Malus modéré +5%")

    # Polluants riches
    rich_polluters = [r for r in results if r['eco_score'] < 0.45 and r['social_score'] > 0.6]
    print(f"\nPOLLUANTS RICHES (eco < 0.45 ET social > 0.6): {len(rich_polluters)} tests")
    print(f"  → Malus maximum +15%")

    return output_path


if __name__ == "__main__":
    run_tests()
