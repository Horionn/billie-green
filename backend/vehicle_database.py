"""
Billie Green - Base de données véhicules
Charge et interroge les données ADEME pour les émissions CO2
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re


class VehicleDatabase:
    """
    Gère la base de données des véhicules ADEME.
    Permet de retrouver les émissions CO2 à partir du modèle de voiture.
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self.load_data()

    def load_data(self):
        """Charge et nettoie les données ADEME."""
        self.df = pd.read_csv(
            self.csv_path,
            sep=';',
            encoding='utf-8-sig',
            low_memory=False
        )

        # Nettoyer les noms de colonnes
        self.df.columns = [col.strip().replace('"', '') for col in self.df.columns]

        # Convertir les colonnes numériques
        numeric_cols = [
            'CO2 vitesse mixte Min', 'CO2 vitesse mixte Max',
            'Puissance fiscale', 'Cylindrée', 'Poids à vide',
            'Prix véhicule'
        ]

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace(',', '.'),
                    errors='coerce'
                )

        # Créer une colonne CO2 moyenne
        if 'CO2 vitesse mixte Min' in self.df.columns and 'CO2 vitesse mixte Max' in self.df.columns:
            self.df['CO2_moyen'] = (
                self.df['CO2 vitesse mixte Min'].fillna(0) +
                self.df['CO2 vitesse mixte Max'].fillna(0)
            ) / 2
            # Si une seule valeur disponible
            self.df.loc[self.df['CO2 vitesse mixte Min'].isna(), 'CO2_moyen'] = \
                self.df.loc[self.df['CO2 vitesse mixte Min'].isna(), 'CO2 vitesse mixte Max']
            self.df.loc[self.df['CO2 vitesse mixte Max'].isna(), 'CO2_moyen'] = \
                self.df.loc[self.df['CO2 vitesse mixte Max'].isna(), 'CO2 vitesse mixte Min']

        # Normaliser les noms de marques
        if 'Marque' in self.df.columns:
            self.df['Marque_norm'] = self.df['Marque'].str.upper().str.strip()

        # Créer un index de recherche
        self._build_search_index()

    def _normalize_brand(self, brand: str) -> str:
        """Normalise les noms de marques pour la recherche."""
        brand = brand.upper().strip()
        # Mapping des variations de noms de marques
        brand_aliases = {
            'BMW': 'B.M.W.',
            'B.M.W.': 'B.M.W.',
            'MERCEDES': 'MERCEDES-BENZ',
            'MERCEDES BENZ': 'MERCEDES-BENZ',
            'MERCEDES-BENZ': 'MERCEDES-BENZ',
            'VW': 'VOLKSWAGEN',
            'VOLKSWAGEN': 'VOLKSWAGEN',
            'ALFA': 'ALFA ROMEO',
            'ALFA ROMEO': 'ALFA ROMEO',
            'LAND ROVER': 'LAND ROVER',
            'LANDROVER': 'LAND ROVER',
            'ASTON': 'ASTON MARTIN',
            'ASTON MARTIN': 'ASTON MARTIN',
        }
        return brand_aliases.get(brand, brand)

    def _build_search_index(self):
        """Construit un index pour la recherche rapide."""
        self.brand_index = {}
        self.model_index = {}

        for idx, row in self.df.iterrows():
            brand = str(row.get('Marque', '')).upper().strip()
            model = str(row.get('Modèle', '')).upper().strip()

            if brand:
                # Indexer avec le nom original ET le nom normalisé
                if brand not in self.brand_index:
                    self.brand_index[brand] = []
                self.brand_index[brand].append(idx)

                # Ajouter aussi les alias inversés
                for alias, normalized in [('B.M.W.', 'BMW'), ('MERCEDES-BENZ', 'MERCEDES')]:
                    if brand == alias:
                        if normalized not in self.brand_index:
                            self.brand_index[normalized] = []
                        self.brand_index[normalized].append(idx)

            if model:
                key = f"{brand}_{model}"
                if key not in self.model_index:
                    self.model_index[key] = []
                self.model_index[key].append(idx)

    def search_vehicle(
        self,
        brand: str,
        model: str,
        year: Optional[int] = None,
        energy: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Recherche un véhicule par marque et modèle.

        Args:
            brand: Marque du véhicule (ex: "RENAULT")
            model: Modèle du véhicule (ex: "CLIO")
            year: Année du véhicule (optionnel)
            energy: Type d'énergie (optionnel: "ESSENCE", "GAZOLE", "ELECTRIC", etc.)

        Returns:
            Dict avec les informations du véhicule ou None
        """
        brand_input = brand.upper().strip()
        model = model.upper().strip()

        # Normaliser la marque pour la recherche
        brand = self._normalize_brand(brand_input)

        # Recherche dans l'index
        key = f"{brand}_{model}"
        matches = []

        # Recherche exacte
        if key in self.model_index:
            matches = self.model_index[key]
        else:
            # Recherche partielle dans les modèles
            for k, indices in self.model_index.items():
                k_brand, k_model = k.split('_', 1) if '_' in k else (k, '')
                # Vérifier si la marque correspond (avec normalisation)
                brand_match = (brand in k_brand or k_brand in brand or
                               brand_input in k_brand or k_brand in brand_input or
                               self._normalize_brand(k_brand) == brand)
                model_match = model in k_model or k_model in model
                if brand_match and model_match:
                    matches.extend(indices)

            # Si toujours rien, recherche par marque seule
            if not matches:
                # Essayer avec la marque normalisée ET l'input original
                for b in [brand, brand_input]:
                    if b in self.brand_index:
                        for idx in self.brand_index[b]:
                            row_model = str(self.df.loc[idx, 'Modèle']).upper()
                            if model in row_model or row_model in model:
                                matches.append(idx)
                        if matches:
                            break

        if not matches:
            return None

        # Filtrer par énergie si spécifié
        if energy and 'Energie' in self.df.columns:
            energy = energy.upper()
            energy_map = {
                'ESSENCE': ['ESSENCE', 'ESS+ELEC', 'SUPERETHANOL'],
                'DIESEL': ['GAZOLE', 'GAZ+ELEC'],
                'GAZOLE': ['GAZOLE', 'GAZ+ELEC'],
                'ELECTRIQUE': ['ELECTRIC'],
                'HYBRIDE': ['ESS+ELEC', 'GAZ+ELEC', 'ELEC+ESSENC', 'ELEC+ESSENC HR']
            }

            energy_types = energy_map.get(energy, [energy])
            filtered = [
                idx for idx in matches
                if any(e in str(self.df.loc[idx, 'Energie']).upper() for e in energy_types)
            ]
            if filtered:
                matches = filtered

        # Prendre le premier résultat (ou moyenner)
        if len(matches) == 1:
            row = self.df.loc[matches[0]]
        else:
            # Moyenner les CO2 des correspondances
            subset = self.df.loc[matches]
            row = subset.iloc[0].copy()
            row['CO2_moyen'] = subset['CO2_moyen'].mean()

        # Gérer le CO2 pour les véhicules électriques (nan ou 0)
        co2_value = row.get('CO2_moyen', 150)
        energie = row.get('Energie', 'INCONNU')
        if pd.isna(co2_value) or co2_value == 0:
            # Véhicule électrique = 0 émissions
            if 'ELEC' in str(energie).upper():
                co2_value = 0
            else:
                co2_value = 150  # Valeur par défaut

        return {
            'marque': row.get('Marque', brand),
            'modele': row.get('Modèle', model),
            'description': row.get('Description Commerciale', ''),
            'energie': energie,
            'co2_g_km': float(co2_value),
            'puissance_fiscale': int(row.get('Puissance fiscale', 0)) if pd.notna(row.get('Puissance fiscale')) else 0,
            'poids': float(row.get('Poids à vide', 0)) if pd.notna(row.get('Poids à vide')) else 0,
            'prix_neuf': float(row.get('Prix véhicule', 0)) if pd.notna(row.get('Prix véhicule')) else 0,
            'bonus_malus': row.get('Bonus-Malus', 'Neutre'),
            'gamme': row.get('Gamme', 'MOYENNE')
        }

    def get_all_brands(self) -> List[str]:
        """Retourne la liste des marques disponibles."""
        return sorted(list(self.brand_index.keys()))

    def get_models_for_brand(self, brand: str) -> List[str]:
        """Retourne les modèles disponibles pour une marque."""
        brand = brand.upper().strip()
        if brand not in self.brand_index:
            return []

        models = set()
        for idx in self.brand_index[brand]:
            model = self.df.loc[idx, 'Modèle']
            if pd.notna(model):
                models.add(str(model))

        return sorted(list(models))

    def estimate_argus(
        self,
        prix_neuf: float,
        age_vehicule: int,
        kilometrage: Optional[int] = None
    ) -> float:
        """
        Estime la valeur ARGUS d'un véhicule.

        Formule simplifiée basée sur:
        - Dépréciation moyenne de 20% la première année
        - Puis ~10-15% par an les années suivantes
        """
        if prix_neuf <= 0:
            return 5000  # Valeur par défaut

        # Coefficients de dépréciation
        depreciation = {
            0: 1.0,
            1: 0.80,
            2: 0.68,
            3: 0.58,
            4: 0.50,
            5: 0.43,
            6: 0.37,
            7: 0.32,
            8: 0.28,
            9: 0.25,
            10: 0.22,
        }

        # Coefficient pour l'âge
        if age_vehicule >= 10:
            coef = 0.22 * (0.9 ** (age_vehicule - 10))  # Continue à baisser lentement
        else:
            coef = depreciation.get(age_vehicule, 0.22)

        # Ajustement kilométrage (optionnel)
        if kilometrage:
            km_moyen = age_vehicule * 15000  # 15000 km/an en moyenne
            if kilometrage > km_moyen * 1.3:
                coef *= 0.9  # Kilométrage élevé
            elif kilometrage < km_moyen * 0.7:
                coef *= 1.1  # Faible kilométrage

        return max(prix_neuf * coef, 500)  # Minimum 500€

    def get_co2_category(self, co2_g_km: float) -> Tuple[str, str]:
        """
        Retourne la catégorie écologique et sa couleur.

        Basé sur les étiquettes énergie/CO2 françaises.
        """
        if co2_g_km == 0:
            return "A", "#00A651"  # Vert foncé - Électrique
        elif co2_g_km <= 100:
            return "A", "#00A651"  # Vert foncé
        elif co2_g_km <= 120:
            return "B", "#78B74A"  # Vert clair
        elif co2_g_km <= 140:
            return "C", "#F0D919"  # Jaune
        elif co2_g_km <= 160:
            return "D", "#F9A01B"  # Orange
        elif co2_g_km <= 200:
            return "E", "#F05A1C"  # Orange foncé
        elif co2_g_km <= 250:
            return "F", "#ED1C24"  # Rouge
        else:
            return "G", "#B30025"  # Rouge foncé

    def get_average_co2_by_energy(self) -> Dict[str, float]:
        """Retourne les émissions moyennes par type d'énergie."""
        if 'Energie' not in self.df.columns:
            return {}

        return self.df.groupby('Energie')['CO2_moyen'].mean().to_dict()


# Singleton pour la base de données
_db_instance: Optional[VehicleDatabase] = None


def get_vehicle_database(csv_path: str = None) -> VehicleDatabase:
    """
    Retourne l'instance singleton de la base de données.
    """
    global _db_instance

    if _db_instance is None:
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "data" / "ADEME-CarLabelling.csv"
        _db_instance = VehicleDatabase(str(csv_path))

    return _db_instance


if __name__ == "__main__":
    # Test
    db = VehicleDatabase("../ADEME-CarLabelling.csv")

    print("=== Marques disponibles ===")
    print(db.get_all_brands()[:20])

    print("\n=== Recherche RENAULT CLIO ===")
    result = db.search_vehicle("RENAULT", "CLIO")
    if result:
        print(result)
        cat, color = db.get_co2_category(result['co2_g_km'])
        print(f"Catégorie: {cat} ({color})")

    print("\n=== Recherche TESLA MODEL 3 ===")
    result = db.search_vehicle("TESLA", "MODEL 3")
    if result:
        print(result)

    print("\n=== CO2 moyen par énergie ===")
    print(db.get_average_co2_by_energy())
