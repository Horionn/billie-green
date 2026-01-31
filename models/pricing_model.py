"""
================================================================================
BILLIE GREEN - MODÈLE DE TARIFICATION INTELLIGENTE
================================================================================

Ce fichier contient l'architecture PyTorch pour une tarification éco-responsable
et éthique. Le modèle utilise le deep learning pour calculer des prix qui:
- Récompensent les véhicules propres (bonus)
- Pénalisent les véhicules polluants (malus)
- MAIS protègent les utilisateurs à faible revenu (éthique)

ARCHITECTURE GÉNÉRALE:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRÉES                                           │
│  - CO2/km du véhicule    - Distance du trajet    - Valeur ARGUS            │
│  - Nombre passagers      - Jour/heure            - Conditions trafic       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENCODEURS (3 modules)                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ CarbonEncoder   │  │ SocialEncoder   │  │ TemporalEncoder │             │
│  │ → eco_score     │  │ → social_score  │  │ → demand_score  │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE D'ATTENTION ÉCO-SOCIALE                           │
│         Apprend à pondérer l'importance de chaque facteur                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COUCHE DE CONTRAINTE ÉTHIQUE                             │
│  SI véhicule polluant ET utilisateur défavorisé → ATTÉNUER le malus        │
│  SI véhicule polluant ET utilisateur aisé → APPLIQUER le malus complet     │
│  SI véhicule propre → TOUJOURS appliquer le bonus                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRIX FINAL                                        │
│  = Prix de base × (1 + ajustement_éthique) × facteur_demande               │
└─────────────────────────────────────────────────────────────────────────────┘

NOTE: Ce modèle PyTorch n'est plus utilisé dans la version actuelle.
L'algorithme a été simplifié en Python pur dans backend/main.py.
Ce fichier est conservé comme référence architecturale.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import torch                          # Framework de deep learning
import torch.nn as nn                 # Modules de réseaux de neurones
import torch.nn.functional as F       # Fonctions (activation, loss, etc.)
import numpy as np                    # Calculs numériques
from typing import Dict, Tuple, Optional  # Typage Python


# ============================================================================
# MODULE 1: ATTENTION ÉCO-SOCIALE
# ============================================================================
class EcoSocialAttention(nn.Module):
    """
    MODULE D'ATTENTION
    ==================

    L'attention est un mécanisme qui permet au modèle d'apprendre QUELS
    facteurs sont les plus importants pour une tarification équitable.

    ANALOGIE: C'est comme un expert humain qui regarde plusieurs critères
    et décide lequel est le plus pertinent pour ce cas précis.

    ENTRÉE: Un vecteur de scores [eco_score, social_score, temporal_score]
    SORTIE: Les mêmes scores mais pondérés par leur importance

    ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────┐
    │  Entrée (3 scores)                                             │
    │       │                                                        │
    │       ▼                                                        │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ Couche linéaire: 3 → 32 dimensions                      │  │
    │  │ (Projette les scores dans un espace plus grand)         │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │       │                                                        │
    │       ▼                                                        │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ Activation Tanh: valeurs entre -1 et +1                 │  │
    │  │ (Non-linéarité pour apprendre des patterns complexes)   │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │       │                                                        │
    │       ▼                                                        │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ Couche linéaire: 32 → 3 dimensions                      │  │
    │  │ (Revient à la dimension originale)                      │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │       │                                                        │
    │       ▼                                                        │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ Softmax: convertit en probabilités (somme = 1)          │  │
    │  │ Ex: [0.5, 0.3, 0.2] = 50% éco, 30% social, 20% temporel │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │       │                                                        │
    │       ▼                                                        │
    │  Sortie: scores pondérés                                       │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, input_dim: int, attention_dim: int = 32):
        """
        Initialise le module d'attention.

        Args:
            input_dim: Nombre de scores en entrée (3 dans notre cas)
            attention_dim: Dimension cachée pour les calculs (32 par défaut)
        """
        super().__init__()

        # Réseau séquentiel qui calcule les poids d'attention
        self.attention = nn.Sequential(
            # Première couche: projette vers un espace plus grand
            nn.Linear(input_dim, attention_dim),  # 3 → 32

            # Activation Tanh: introduit de la non-linéarité
            # Tanh(x) ∈ [-1, 1], permet d'apprendre des patterns complexes
            nn.Tanh(),

            # Deuxième couche: revient à la dimension originale
            nn.Linear(attention_dim, input_dim),  # 32 → 3

            # Softmax: convertit en "poids" qui somment à 1
            # Si sortie = [2.0, 1.0, 0.5], Softmax donne ≈ [0.59, 0.24, 0.17]
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique l'attention sur les scores d'entrée.

        Args:
            x: Tensor de shape (batch_size, 3) contenant les scores
               [eco_score, social_score, temporal_score]

        Returns:
            weighted_x: Scores multipliés par leurs poids d'attention
            weights: Les poids d'attention eux-mêmes (pour visualisation)

        Exemple:
            Si x = [0.8, 0.3, 0.5] (eco=0.8, social=0.3, temporal=0.5)
            Et weights = [0.6, 0.3, 0.1] (l'attention privilégie l'éco)
            Alors weighted_x = [0.48, 0.09, 0.05]
        """
        # Calcule les poids d'attention via le réseau
        weights = self.attention(x)

        # Multiplie chaque score par son poids d'importance
        weighted_x = x * weights

        # Retourne les scores pondérés ET les poids (pour transparence)
        return weighted_x, weights


# ============================================================================
# MODULE 2: CONTRAINTE ÉTHIQUE
# ============================================================================
class EthicalConstraintLayer(nn.Module):
    """
    COUCHE DE CONTRAINTE ÉTHIQUE
    ============================

    C'est le CŒUR de l'algorithme Billie Green. Cette couche s'assure que
    les utilisateurs défavorisés ne soient pas injustement pénalisés.

    PRINCIPE FONDAMENTAL:
    ┌────────────────────────────────────────────────────────────────┐
    │  Le malus écologique est ATTÉNUÉ proportionnellement          │
    │  au niveau socio-économique défavorable de l'utilisateur.     │
    │                                                                │
    │  En clair:                                                     │
    │  - Riche + Polluant  → GROS malus (peut se le permettre)      │
    │  - Pauvre + Polluant → PAS de malus (on protège)              │
    │  - Propre (tous)     → BONUS (on récompense)                  │
    └────────────────────────────────────────────────────────────────┘

    MATRICE DE DÉCISION:
    ┌─────────────────┬──────────────┬──────────────┬──────────────┐
    │                 │ Social FAIBLE│ Social MOYEN │ Social ÉLEVÉ │
    │                 │   (pauvre)   │   (classe    │   (aisé)     │
    │                 │              │   moyenne)   │              │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ Eco BON         │   BONUS ✓    │   BONUS ✓    │   BONUS ✓    │
    │ (électrique,    │   -12%       │   -12%       │   -12%       │
    │  hybride)       │              │              │              │
    ├─────────────────┼──────────────┼──────────────┼──────────────┤
    │ Eco MAUVAIS     │   PROTÉGÉ ✓  │   MALUS      │   MALUS      │
    │ (diesel,        │   0%         │   +5%        │   +15%       │
    │  essence)       │              │              │              │
    └─────────────────┴──────────────┴──────────────┴──────────────┘

    FORMULE MATHÉMATIQUE:

    protection_éthique = 1 - (1 - social_score) × sensitivity

    Où:
    - social_score = 0 → défavorisé, 1 → aisé
    - sensitivity = sensibilité de la protection (0.7 par défaut)

    Exemple avec sensitivity = 0.7:
    - Si social_score = 0 (très défavorisé): protection = 1 - 1×0.7 = 0.3
      → Le malus est réduit à 30% de sa valeur normale
    - Si social_score = 1 (très aisé): protection = 1 - 0×0.7 = 1.0
      → Le malus est appliqué à 100%
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialise la couche de contrainte éthique.

        Args:
            sensitivity: Niveau de protection des défavorisés (0 à 1)
                        0.7 = protection forte (recommandé)
                        0.5 = protection modérée
                        0.0 = pas de protection (désactivé)
        """
        super().__init__()

        # La sensibilité est un paramètre APPRENABLE
        # Le modèle peut l'ajuster pendant l'entraînement
        self.sensitivity = nn.Parameter(torch.tensor(sensitivity))

    def forward(
        self,
        eco_score: torch.Tensor,      # Score écologique: 0=polluant, 1=propre
        social_score: torch.Tensor,   # Score social: 0=défavorisé, 1=aisé
        base_price: torch.Tensor      # Prix de base (distance × tarif/km)
    ) -> torch.Tensor:
        """
        Calcule le prix ajusté en appliquant la contrainte éthique.

        Args:
            eco_score: Score écologique entre 0 et 1
                      0.0 = très polluant (vieux diesel)
                      0.5 = moyen (essence récent)
                      1.0 = très propre (électrique)

            social_score: Score socio-économique entre 0 et 1
                         0.0 = défavorisé (ARGUS < 5000€)
                         0.5 = classe moyenne (ARGUS ~15000€)
                         1.0 = aisé (ARGUS > 35000€)

            base_price: Prix de base en euros (distance × 0.15€/km)

        Returns:
            Prix ajusté tenant compte de l'éthique

        Exemples détaillés:

        CAS 1: Tesla Model 3, ARGUS 40k€
        - eco_score = 0.95 (électrique = propre)
        - social_score = 1.0 (voiture chère = aisé)
        - eco_modifier = (0.95 - 0.5) × 0.3 = +0.135 (BONUS)
        - → Prix réduit de 13.5% car véhicule propre

        CAS 2: BMW X5 Diesel, ARGUS 45k€
        - eco_score = 0.35 (diesel = polluant)
        - social_score = 1.0 (voiture chère = aisé)
        - eco_modifier = (0.35 - 0.5) × 0.3 = -0.045 (MALUS)
        - protection = 1.0 (aisé = pas de protection)
        - → Prix augmenté de 4.5%

        CAS 3: Vieille Clio Diesel, ARGUS 4k€
        - eco_score = 0.35 (diesel = polluant)
        - social_score = 0.11 (voiture pas chère = défavorisé)
        - eco_modifier = (0.35 - 0.5) × 0.3 = -0.045 (MALUS théorique)
        - protection = 1 - (1 - 0.11) × 0.7 = 0.377
        - adjusted_modifier = -0.045 × 0.377 = -0.017
        - → Le malus est ATTÉNUÉ: seulement 1.7% au lieu de 4.5%
        """

        # ====== ÉTAPE 1: Calculer le niveau de protection éthique ======
        # Plus social_score est bas, plus la protection est forte
        #
        # Formule: protection = 1 - (1 - social_score) × sensitivity
        #
        # Intuition:
        # - (1 - social_score) = "niveau de défavorisation"
        #   - Si social = 0 → défavorisation = 1 (max)
        #   - Si social = 1 → défavorisation = 0 (pas défavorisé)
        #
        # - × sensitivity = combien on veut protéger
        #   - sensitivity = 0.7 → on protège beaucoup
        #   - sensitivity = 0.3 → on protège peu
        #
        # - 1 - ... = inverser pour avoir le "facteur d'application du malus"
        #   - protection = 0.3 → on applique 30% du malus seulement
        #   - protection = 1.0 → on applique 100% du malus

        ethical_protection = 1 - (1 - social_score) * self.sensitivity

        # ====== ÉTAPE 2: Calculer le modificateur écologique ======
        # Le score éco va de 0 (polluant) à 1 (propre)
        # On le centre sur 0.5 pour avoir des bonus ET des malus
        #
        # eco_modifier = (eco_score - 0.5) × 0.3
        #
        # Exemples:
        # - eco = 0.0 → modifier = (0.0 - 0.5) × 0.3 = -0.15 (MALUS 15%)
        # - eco = 0.5 → modifier = (0.5 - 0.5) × 0.3 = 0.00 (NEUTRE)
        # - eco = 1.0 → modifier = (1.0 - 0.5) × 0.3 = +0.15 (BONUS 15%)

        eco_modifier = (eco_score - 0.5) * 0.3  # Résultat entre -0.15 et +0.15

        # ====== ÉTAPE 3: Appliquer la protection éthique ======
        # IMPORTANT: La protection ne s'applique QU'AUX MALUS !
        # Les bonus (véhicules propres) sont toujours appliqués intégralement
        #
        # Si eco_modifier < 0 (c'est un malus):
        #   → On l'atténue selon la protection éthique
        # Sinon (c'est un bonus):
        #   → On le garde tel quel

        adjusted_modifier = torch.where(
            eco_modifier < 0,                    # SI c'est un malus...
            eco_modifier * ethical_protection,   # ...on l'atténue
            eco_modifier                         # SINON on garde le bonus
        )

        # ====== ÉTAPE 4: Appliquer au prix de base ======
        # Prix final = prix_base × (1 + modificateur_ajusté)
        #
        # Exemples:
        # - base = 50€, modifier = -0.15 → 50 × 0.85 = 42.50€ (bonus)
        # - base = 50€, modifier = +0.05 → 50 × 1.05 = 52.50€ (malus)
        # - base = 50€, modifier = 0.00  → 50 × 1.00 = 50.00€ (neutre)

        return base_price * (1 + adjusted_modifier)


# ============================================================================
# MODULE 3: ENCODEUR D'IMPACT CARBONE
# ============================================================================
class CarbonImpactEncoder(nn.Module):
    """
    ENCODEUR D'IMPACT CARBONE
    =========================

    Ce module transforme les données brutes du véhicule et du trajet
    en un SCORE ÉCOLOGIQUE unique entre 0 et 1.

    ENTRÉES:
    - co2_per_km: Émissions CO2 du véhicule (grammes par km)
      - 0 g/km = électrique
      - 100 g/km = hybride efficace
      - 150 g/km = essence moyen
      - 200+ g/km = gros diesel/SUV

    - distance: Longueur du trajet en km
      - Impact modéré sur le score

    - traffic_factor: Facteur d'embouteillage
      - 1.0 = trafic fluide
      - 1.5 = embouteillages (plus de pollution)

    - passengers: Nombre de passagers
      - Plus il y a de passagers, meilleur est le score
      - (mutualisation de l'impact)

    SORTIE:
    - eco_score entre 0 (très polluant) et 1 (très propre)

    ARCHITECTURE DU RÉSEAU:
    ┌────────────────────────────────────────────────────────────────┐
    │  4 features normalisées                                        │
    │       │                                                        │
    │       ▼                                                        │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ Linear(4 → 64) + LayerNorm + GELU                       │  │
    │  │ (Couche dense avec normalisation et activation)         │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │       │                                                        │
    │       ▼                                                        │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ Linear(64 → 32) + GELU                                  │  │
    │  │ (Réduction de dimensionnalité)                          │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │       │                                                        │
    │       ▼                                                        │
    │  ┌─────────────────────────────────────────────────────────┐  │
    │  │ Linear(32 → 1) + Sigmoid                                │  │
    │  │ (Score final entre 0 et 1)                              │  │
    │  └─────────────────────────────────────────────────────────┘  │
    │       │                                                        │
    │       ▼                                                        │
    │  eco_score ∈ [0, 1]                                            │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Initialise l'encodeur d'impact carbone.

        Args:
            hidden_dim: Dimension des couches cachées (64 par défaut)
        """
        super().__init__()

        # Réseau de neurones séquentiel
        self.encoder = nn.Sequential(
            # Couche 1: 4 features → 64 neurones
            nn.Linear(4, hidden_dim),

            # LayerNorm: Normalise les activations pour stabiliser l'entraînement
            # Chaque neurone a moyenne=0 et variance=1
            nn.LayerNorm(hidden_dim),

            # GELU: Activation moderne (utilisée dans GPT, BERT)
            # Plus douce que ReLU, permet de petits gradients négatifs
            nn.GELU(),

            # Couche 2: Réduction 64 → 32
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),

            # Couche finale: 32 → 1 score
            nn.Linear(hidden_dim // 2, 1),

            # Sigmoid: Force la sortie entre 0 et 1
            nn.Sigmoid()
        )

    def forward(
        self,
        co2_per_km: torch.Tensor,    # Émissions CO2 en g/km
        distance: torch.Tensor,       # Distance en km
        traffic_factor: torch.Tensor, # Facteur trafic (1.0 = normal)
        passengers: torch.Tensor      # Nombre de passagers
    ) -> torch.Tensor:
        """
        Calcule le score écologique à partir des caractéristiques.

        Le processus:
        1. Normaliser chaque feature entre 0 et 1
        2. Inverser certaines features (CO2 bas = BON score)
        3. Passer dans le réseau de neurones
        4. Obtenir le score final

        Returns:
            Score écologique entre 0 (polluant) et 1 (propre)
        """

        # ====== NORMALISATION DES FEATURES ======
        # On ramène chaque valeur entre 0 et 1 avec torch.clamp
        # Clamp(x, 0, 1) = max(0, min(x, 1))

        # CO2: 250 g/km = maximum (très polluant)
        # 0 g/km → 0.0, 250 g/km → 1.0
        co2_norm = torch.clamp(co2_per_km / 250.0, 0, 1)

        # Distance: 1000 km = maximum (très long trajet)
        # 0 km → 0.0, 1000 km → 1.0
        dist_norm = torch.clamp(distance / 1000.0, 0, 1)

        # Trafic: Le facteur va de 1.0 (fluide) à 1.5 (bouchon)
        # On normalise l'impact additionnel: (factor - 1) / 0.5
        # 1.0 → 0.0, 1.5 → 1.0
        traffic_norm = torch.clamp((traffic_factor - 1) / 0.5, 0, 1)

        # Passagers: 5 passagers = maximum de mutualisation
        # 1 passager → 0.2, 5 passagers → 1.0
        pass_norm = torch.clamp(passengers / 5.0, 0, 1)

        # ====== CONSTRUCTION DU VECTEUR DE FEATURES ======
        # IMPORTANT: On INVERSE certaines features
        # Car on veut que "faible CO2" = "BON score"

        features = torch.stack([
            1 - co2_norm,           # Inverser: faible CO2 = score élevé
            1 - dist_norm * 0.3,    # Distance a un impact MODÉRÉ (×0.3)
            1 - traffic_norm,       # Inverser: embouteillages = mauvais
            pass_norm * 0.5 + 0.5   # Plus de passagers = meilleur (base 0.5)
        ], dim=-1)

        # ====== PASSAGE DANS LE RÉSEAU ======
        # Le réseau apprend à combiner ces features de manière optimale
        return self.encoder(features)


# ============================================================================
# MODULE 4: ENCODEUR DE PROFIL SOCIAL
# ============================================================================
class SocialProfileEncoder(nn.Module):
    """
    ENCODEUR DE PROFIL SOCIAL
    =========================

    Ce module estime la capacité financière de l'utilisateur
    à partir de la VALEUR ARGUS de son véhicule.

    HYPOTHÈSE FONDAMENTALE:
    ┌────────────────────────────────────────────────────────────────┐
    │  La valeur de revente d'un véhicule (ARGUS) est un PROXY      │
    │  acceptable de la capacité financière de son propriétaire.    │
    │                                                                │
    │  - Voiture à 3 000€  → probablement revenus modestes          │
    │  - Voiture à 15 000€ → probablement classe moyenne            │
    │  - Voiture à 45 000€ → probablement revenus aisés             │
    └────────────────────────────────────────────────────────────────┘

    LIMITES (importantes à connaître):
    - Une personne riche peut avoir une vieille voiture par choix
    - Une personne modeste peut avoir une voiture de fonction chère
    - L'ARGUS n'est qu'une ESTIMATION, pas une certitude

    ENTRÉES:
    - argus_value: Valeur ARGUS du véhicule en euros
    - car_age: Âge du véhicule en années (facteur secondaire)

    SORTIE:
    - social_score entre 0 (défavorisé) et 1 (aisé)
    """

    def __init__(self, hidden_dim: int = 32):
        """
        Initialise l'encodeur de profil social.

        Args:
            hidden_dim: Dimension cachée (32, plus petit car moins de features)
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # 2 features → 32 neurones
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            # 32 → 1 score
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Score entre 0 et 1
        )

    def forward(
        self,
        argus_value: torch.Tensor,  # Valeur ARGUS en euros
        car_age: torch.Tensor       # Âge du véhicule en années
    ) -> torch.Tensor:
        """
        Calcule le score social (capacité contributive).

        Returns:
            Score social entre 0 (défavorisé, protégé) et 1 (aisé)

        Exemples:
        - ARGUS 3 000€, âge 15 ans → score ≈ 0.1 (protégé)
        - ARGUS 15 000€, âge 5 ans → score ≈ 0.4 (moyen)
        - ARGUS 40 000€, âge 2 ans → score ≈ 0.95 (aisé)
        """

        # ====== NORMALISATION ======
        # ARGUS: 40 000€ = seuil "voiture chère"
        # Au-delà, le score est plafonné à 1.0
        argus_norm = torch.clamp(argus_value / 40000.0, 0, 1)

        # Âge: 15 ans = très vieille voiture
        # Une voiture vieille a un LÉGER impact négatif
        age_norm = torch.clamp(car_age / 15.0, 0, 1)

        # ====== FEATURES ======
        features = torch.stack([
            argus_norm,              # Valeur ARGUS (facteur principal)
            1 - age_norm * 0.3       # Âge a un impact LÉGER (×0.3)
        ], dim=-1)

        return self.encoder(features)


# ============================================================================
# MODULE 5: ENCODEUR TEMPOREL
# ============================================================================
class TemporalEncoder(nn.Module):
    """
    ENCODEUR TEMPOREL
    =================

    Ce module encode les facteurs temporels qui influencent la DEMANDE.
    La demande affecte légèrement le prix (offre et demande classique).

    FACTEURS PRIS EN COMPTE:

    1. JOUR DE LA SEMAINE (0-6):
       - Lundi-Jeudi: Demande normale
       - Vendredi: Forte demande (départs en week-end)
       - Samedi-Dimanche: Demande variable

    2. TYPE DE SEMAINE:
       - 0 = Normal: Demande standard
       - 1 = Vacances scolaires: Forte demande
       - 2 = Pont (week-end prolongé): Très forte demande
       - 3 = Été (juillet-août): Demande particulière

    TECHNIQUE UTILISÉE: EMBEDDINGS
    ┌────────────────────────────────────────────────────────────────┐
    │  Un embedding transforme un entier en vecteur de nombres.     │
    │                                                                │
    │  Exemple pour les jours (dimension 16):                        │
    │  Lundi    → [0.2, -0.1, 0.8, ..., 0.3]  (16 nombres)          │
    │  Vendredi → [0.9, 0.5, -0.2, ..., 0.7]  (16 nombres)          │
    │                                                                │
    │  Ces vecteurs sont APPRIS pendant l'entraînement.             │
    │  Le modèle découvre que "vendredi" et "vacances" sont         │
    │  similaires (forte demande) et auront des embeddings proches. │
    └────────────────────────────────────────────────────────────────┘

    SORTIE:
    - demand_score entre 0 (faible demande) et 1 (forte demande)
    """

    def __init__(self, hidden_dim: int = 32):
        """
        Initialise l'encodeur temporel.

        Args:
            hidden_dim: Dimension des embeddings (32 par défaut)
        """
        super().__init__()

        # Embedding pour les 7 jours de la semaine
        # Chaque jour est représenté par un vecteur de 16 dimensions
        self.day_embedding = nn.Embedding(
            num_embeddings=7,           # 7 jours possibles (0-6)
            embedding_dim=hidden_dim // 2  # 16 dimensions par jour
        )

        # Embedding pour les 4 types de semaine
        self.week_type_embedding = nn.Embedding(
            num_embeddings=4,           # 4 types (normal, vacances, pont, été)
            embedding_dim=hidden_dim // 2  # 16 dimensions par type
        )

        # Réseau qui combine les embeddings
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 32 → 32
            nn.GELU(),
            nn.Linear(hidden_dim, 1),           # 32 → 1
            nn.Sigmoid()                        # Score entre 0 et 1
        )

    def forward(
        self,
        day_of_week: torch.Tensor,  # Jour: 0=Lundi, 6=Dimanche
        week_type: torch.Tensor     # Type: 0=normal, 1=vacances, 2=pont, 3=été
    ) -> torch.Tensor:
        """
        Calcule le score de demande temporelle.

        Returns:
            Score de demande entre 0 (faible) et 1 (forte)

        Exemples:
        - Mardi normal → score ≈ 0.4 (demande faible)
        - Vendredi vacances → score ≈ 0.9 (forte demande)
        """

        # Récupérer les embeddings (vecteurs appris)
        day_emb = self.day_embedding(day_of_week.long())    # Shape: (batch, 16)
        week_emb = self.week_type_embedding(week_type.long())  # Shape: (batch, 16)

        # Concaténer les deux embeddings
        combined = torch.cat([day_emb, week_emb], dim=-1)   # Shape: (batch, 32)

        # Passer dans le réseau pour obtenir le score
        return self.encoder(combined)


# ============================================================================
# MODÈLE PRINCIPAL: BILLIE GREEN PRICING MODEL
# ============================================================================
class BillieGreenPricingModel(nn.Module):
    """
    MODÈLE PRINCIPAL DE TARIFICATION BILLIE GREEN
    ==============================================

    Ce modèle combine tous les modules précédents pour calculer
    un prix JUSTE, ÉCOLOGIQUE et ÉTHIQUE.

    PIPELINE COMPLET:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ÉTAPE 1: ENCODAGE DES CARACTÉRISTIQUES                                │
    │  ───────────────────────────────────────                               │
    │  Les 3 encodeurs transforment les données brutes en scores:            │
    │                                                                         │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
    │  │   Carbon    │     │   Social    │     │  Temporal   │               │
    │  │  Encoder    │     │  Encoder    │     │  Encoder    │               │
    │  │             │     │             │     │             │               │
    │  │ CO2, dist,  │     │ ARGUS, âge  │     │ Jour, type  │               │
    │  │ trafic,     │     │             │     │ semaine     │               │
    │  │ passagers   │     │             │     │             │               │
    │  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘               │
    │         │                   │                   │                       │
    │         ▼                   ▼                   ▼                       │
    │    eco_score           social_score        temporal_score               │
    │    [0, 1]              [0, 1]              [0, 1]                       │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ÉTAPE 2: CALCUL DU PRIX DE BASE                                       │
    │  ──────────────────────────────                                        │
    │  prix_base = max(distance × 0.15€/km, 20€)                             │
    │                                                                         │
    │  Exemples:                                                              │
    │  - Paris-Lyon (462 km) → 462 × 0.15 = 69.30€                           │
    │  - Trajet court (50 km) → max(7.50, 20) = 20€ (minimum)                │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ÉTAPE 3: MODULE D'ATTENTION                                           │
    │  ───────────────────────────                                           │
    │  Pondère les 3 scores selon leur importance relative                   │
    │                                                                         │
    │  Entrée: [eco=0.35, social=0.8, temporal=0.6]                          │
    │  Poids:  [0.5, 0.3, 0.2] (l'éco est plus important)                    │
    │  Sortie: [0.175, 0.24, 0.12] (scores pondérés)                         │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ÉTAPE 4: CONTRAINTE ÉTHIQUE (LE CŒUR DE L'ALGORITHME)                 │
    │  ─────────────────────────────────────────────────────                 │
    │                                                                         │
    │  CAS 1: Véhicule PROPRE (eco_score ≥ 0.7)                              │
    │  → BONUS appliqué (réduction du prix)                                  │
    │                                                                         │
    │  CAS 2: Véhicule POLLUANT + Utilisateur AISÉ                           │
    │  → MALUS complet (augmentation du prix)                                │
    │                                                                         │
    │  CAS 3: Véhicule POLLUANT + Utilisateur DÉFAVORISÉ                     │
    │  → MALUS ATTÉNUÉ ou SUPPRIMÉ (protection sociale)                      │
    │                                                                         │
    │  Formule: prix_éthique = prix_base × (1 + modifier × protection)       │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ÉTAPE 5: AJUSTEMENT TEMPOREL (DEMANDE)                                │
    │  ──────────────────────────────────────                                │
    │  Le prix varie légèrement selon la demande                             │
    │                                                                         │
    │  facteur_demande = 1 + (temporal_score - 0.5) × 0.2                    │
    │                                                                         │
    │  - Demande faible (score=0.3) → facteur = 0.96 (-4%)                   │
    │  - Demande normale (score=0.5) → facteur = 1.00 (0%)                   │
    │  - Demande forte (score=0.8) → facteur = 1.06 (+6%)                    │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ÉTAPE 6: PRIX FINAL                                                   │
    │  ───────────────────                                                   │
    │  prix_final = prix_éthique × facteur_demande                           │
    │                                                                         │
    │  Le résultat est toujours POSITIF grâce à Softplus                     │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        ethical_sensitivity: float = 0.7
    ):
        """
        Initialise le modèle complet.

        Args:
            hidden_dim: Dimension des couches cachées
            ethical_sensitivity: Force de la protection sociale (0.7 = forte)
        """
        super().__init__()

        # ====== ENCODEURS ======
        # Chaque encodeur transforme des données brutes en score [0, 1]
        self.carbon_encoder = CarbonImpactEncoder(hidden_dim)
        self.social_encoder = SocialProfileEncoder(hidden_dim // 2)
        self.temporal_encoder = TemporalEncoder(hidden_dim // 2)

        # ====== MODULE D'ATTENTION ======
        # Apprend à pondérer les 3 scores (éco, social, temporel)
        self.attention = EcoSocialAttention(3)

        # ====== CONTRAINTE ÉTHIQUE ======
        # Protège les utilisateurs défavorisés
        self.ethical_constraint = EthicalConstraintLayer(ethical_sensitivity)

        # ====== RÉSEAU DE PRIX FINAL ======
        # Combine tous les facteurs pour produire le prix
        self.price_network = nn.Sequential(
            nn.Linear(4, hidden_dim),     # 3 scores + distance → 64
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),              # Régularisation (évite le surapprentissage)
            nn.Linear(hidden_dim, hidden_dim // 2),  # 64 → 32
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # 32 → 1 prix
            nn.Softplus()                   # Assure un prix POSITIF
        )

        # ====== PARAMÈTRES DE BASE ======
        # Ces paramètres sont APPRENABLES (le modèle peut les ajuster)
        self.base_price_per_km = nn.Parameter(torch.tensor(0.15))  # 0.15€ par km
        self.min_price = nn.Parameter(torch.tensor(20.0))          # Minimum 20€

    def forward(
        self,
        co2_per_km: torch.Tensor,     # Émissions CO2 (g/km)
        distance: torch.Tensor,        # Distance du trajet (km)
        traffic_factor: torch.Tensor,  # Facteur trafic (1.0 = normal)
        passengers: torch.Tensor,      # Nombre de passagers
        argus_value: torch.Tensor,     # Valeur ARGUS du véhicule (€)
        car_age: torch.Tensor,         # Âge du véhicule (années)
        day_of_week: torch.Tensor,     # Jour de la semaine (0-6)
        week_type: torch.Tensor        # Type de semaine (0-3)
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule le prix recommandé avec toutes les composantes.

        Returns:
            Dictionnaire contenant:
            - final_price: Prix final en euros
            - base_price: Prix de base (avant ajustements)
            - eco_score: Score écologique [0-1]
            - social_score: Score social [0-1]
            - temporal_score: Score de demande [0-1]
            - eco_adjustment: Bonus/malus appliqué (%)
            - attention_weights: Poids d'attention (pour transparence)
            - demand_factor: Facteur de demande appliqué
        """

        # ===== ÉTAPE 1: Encoder les caractéristiques =====
        # Chaque encodeur produit un score entre 0 et 1
        eco_score = self.carbon_encoder(co2_per_km, distance, traffic_factor, passengers)
        social_score = self.social_encoder(argus_value, car_age)
        temporal_score = self.temporal_encoder(day_of_week, week_type)

        # ===== ÉTAPE 2: Prix de base selon la distance =====
        # Formule: distance × tarif/km, avec un minimum
        base_price = distance * self.base_price_per_km
        base_price = torch.maximum(base_price, self.min_price)

        # ===== ÉTAPE 3: Attention sur les scores =====
        # Combine les 3 scores et calcule leurs poids d'importance
        scores = torch.cat([eco_score, social_score, temporal_score], dim=-1)
        weighted_scores, attention_weights = self.attention(scores)

        # ===== ÉTAPE 4: Appliquer la contrainte éthique =====
        # C'est ici que la magie opère:
        # - Véhicule propre → bonus
        # - Véhicule polluant + riche → malus
        # - Véhicule polluant + pauvre → protection
        ethical_price = self.ethical_constraint(eco_score, social_score, base_price)

        # ===== ÉTAPE 5: Ajustement temporel (offre/demande) =====
        # La demande influence légèrement le prix (±10%)
        # temporal_score = 0.5 → facteur = 1.0 (neutre)
        # temporal_score = 0.0 → facteur = 0.9 (-10%)
        # temporal_score = 1.0 → facteur = 1.1 (+10%)
        demand_factor = 1 + (temporal_score.squeeze(-1) - 0.5) * 0.2

        # ===== ÉTAPE 6: Prix final =====
        final_price = ethical_price * demand_factor

        # ===== Calcul des métriques pour transparence =====
        # L'ajustement éco montre combien le prix a varié par rapport à la base
        eco_adjustment = (ethical_price - base_price) / base_price

        return {
            'final_price': final_price.squeeze(-1),
            'base_price': base_price,
            'eco_score': eco_score.squeeze(-1),
            'social_score': social_score.squeeze(-1),
            'temporal_score': temporal_score.squeeze(-1),
            'eco_adjustment': eco_adjustment.squeeze(-1),
            'attention_weights': attention_weights,
            'demand_factor': demand_factor
        }

    def explain_price(self, output: Dict[str, torch.Tensor]) -> str:
        """
        Génère une explication TEXTUELLE de la tarification.

        Cette méthode est utilisée pour la transparence:
        l'utilisateur comprend POURQUOI il paie ce prix.

        Args:
            output: Dictionnaire retourné par forward()

        Returns:
            Chaîne de caractères explicative

        Exemples:
        - "Votre trajet est éco-responsable | Bonus écologique de 12%"
        - "Impact carbone élevé | Malus écologique atténué (profil protégé)"
        """
        # Extraire les valeurs (convertir tenseur → float)
        eco = output['eco_score'].item()
        social = output['social_score'].item()
        adjustment = output['eco_adjustment'].item()

        explanation = []

        # ===== Explication du score écologique =====
        if eco > 0.7:
            explanation.append("Votre trajet est éco-responsable")
        elif eco < 0.3:
            explanation.append("Impact carbone élevé")
        else:
            explanation.append("Impact carbone modéré")

        # ===== Explication de l'ajustement =====
        if adjustment < -0.05:
            # Bonus (réduction) supérieur à 5%
            explanation.append(f"Bonus écologique de {abs(adjustment)*100:.1f}%")
        elif adjustment > 0.05:
            # Malus (augmentation) supérieur à 5%
            if social < 0.4:
                # Profil défavorisé → on explique la protection
                explanation.append("Malus écologique atténué (profil protégé)")
            else:
                # Profil aisé → malus normal
                explanation.append(f"Contribution écologique de {adjustment*100:.1f}%")

        # Joindre les explications avec un séparateur
        return " | ".join(explanation)


# ============================================================================
# CLASSE D'ENTRAÎNEMENT
# ============================================================================
class PricingModelTrainer:
    """
    CLASSE POUR ENTRAÎNER LE MODÈLE
    ================================

    Cette classe gère l'entraînement du modèle de tarification.

    L'ENTRAÎNEMENT EN DEEP LEARNING:
    ┌────────────────────────────────────────────────────────────────┐
    │  1. Le modèle fait une PRÉDICTION (prix estimé)               │
    │  2. On calcule l'ERREUR par rapport au prix réel              │
    │  3. On ajuste les poids du modèle pour RÉDUIRE l'erreur       │
    │  4. On répète des milliers de fois                            │
    └────────────────────────────────────────────────────────────────┘

    FONCTION DE PERTE (LOSS):
    Notre loss est MULTI-OBJECTIF, elle combine:

    1. PRICE_LOSS: Erreur sur les prix (MSE)
       → Le prix prédit doit être proche du prix réel

    2. ECO_REG: Régularisation écologique
       → Encourage les bonus pour les véhicules verts

    3. ETHICAL_LOSS: Contrainte éthique
       → Pénalise les gros malus pour les défavorisés

    FORMULE:
    total_loss = price_loss + eco_reg + ethical_loss
    """

    def __init__(
        self,
        model: BillieGreenPricingModel,
        learning_rate: float = 1e-3
    ):
        """
        Initialise le trainer.

        Args:
            model: Le modèle à entraîner
            learning_rate: Vitesse d'apprentissage (1e-3 = 0.001)
                          Trop haut → apprentissage instable
                          Trop bas → apprentissage trop lent
        """
        self.model = model

        # AdamW: Optimiseur moderne (variante d'Adam avec weight decay)
        # C'est l'algorithme qui ajuste les poids du modèle
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        target_prices: torch.Tensor,
        eco_targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule la perte multi-objectif.

        Args:
            predictions: Sortie du modèle (dictionnaire)
            target_prices: Prix réels (vérité terrain)
            eco_targets: Scores éco cibles (optionnel)

        Returns:
            total_loss: Perte totale à minimiser
            loss_components: Détail de chaque composante
        """

        # ===== COMPOSANTE 1: Erreur sur les prix (MSE) =====
        # MSE = Mean Squared Error = moyenne des carrés des erreurs
        # Si prediction = 50€ et target = 52€, erreur² = (50-52)² = 4
        price_loss = F.mse_loss(predictions['final_price'], target_prices)

        # ===== COMPOSANTE 2: Régularisation écologique =====
        # On ENCOURAGE le modèle à donner des bonus aux véhicules verts
        # eco_score élevé + eco_adjustment négatif (bonus) = bon comportement
        # Le signe moins (-0.1) fait que c'est une "récompense" (réduit la loss)
        eco_reg = -0.1 * torch.mean(predictions['eco_score'] * predictions['eco_adjustment'])

        # ===== COMPOSANTE 3: Contrainte éthique =====
        # On PÉNALISE le modèle s'il applique des malus aux défavorisés
        #
        # ethical_violation = malus × niveau_défavorisé
        #
        # Si eco_adjustment > 0 (malus) ET social_score faible:
        # → violation élevée → loss augmente → modèle pénalisé
        #
        # relu() garde seulement les valeurs positives (violations)
        ethical_violation = torch.relu(
            -predictions['eco_adjustment'] *    # Malus (négatif = bonus)
            (1 - predictions['social_score']) *  # Niveau de défavorisation
            0.5                                  # Sensibilité
        )
        ethical_loss = torch.mean(ethical_violation)

        # ===== TOTAL =====
        total_loss = price_loss + eco_reg + ethical_loss

        return total_loss, {
            'price_loss': price_loss.item(),
            'eco_reg': eco_reg.item(),
            'ethical_loss': ethical_loss.item()
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        target_prices: torch.Tensor
    ) -> Dict[str, float]:
        """
        Effectue UNE étape d'entraînement.

        Une étape = un batch de données:
        1. Forward pass (prédiction)
        2. Calcul de la loss
        3. Backward pass (calcul des gradients)
        4. Mise à jour des poids

        Args:
            batch: Dictionnaire des features d'entrée
            target_prices: Prix réels à prédire

        Returns:
            Dictionnaire des valeurs de loss
        """
        # Mode entraînement (active dropout, etc.)
        self.model.train()

        # Remet les gradients à zéro (sinon ils s'accumulent)
        self.optimizer.zero_grad()

        # ===== FORWARD PASS =====
        # Le modèle fait sa prédiction
        predictions = self.model(**batch)

        # ===== CALCUL DE LA LOSS =====
        loss, loss_components = self.compute_loss(predictions, target_prices)

        # ===== BACKWARD PASS =====
        # Calcule les gradients (dérivées partielles de la loss)
        # Ces gradients indiquent comment modifier chaque poids
        loss.backward()

        # ===== MISE À JOUR DES POIDS =====
        # L'optimiseur ajuste les poids selon les gradients
        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            **loss_components
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def create_model(
    hidden_dim: int = 64,
    ethical_sensitivity: float = 0.7
) -> BillieGreenPricingModel:
    """
    FONCTION FACTORY POUR CRÉER UN MODÈLE
    =====================================

    Une factory function est un pattern de conception qui encapsule
    la création d'objets complexes.

    Args:
        hidden_dim: Dimension des couches cachées (64 recommandé)
        ethical_sensitivity: Force de la protection sociale
                            0.7 = forte protection (recommandé)
                            0.5 = protection modérée
                            0.0 = pas de protection

    Returns:
        Instance de BillieGreenPricingModel prête à l'emploi

    Exemple:
        model = create_model(hidden_dim=64, ethical_sensitivity=0.7)
        output = model(co2_per_km, distance, ...)
    """
    model = BillieGreenPricingModel(
        hidden_dim=hidden_dim,
        ethical_sensitivity=ethical_sensitivity
    )
    return model


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================
if __name__ == "__main__":
    """
    CODE DE TEST
    ============

    Ce bloc s'exécute uniquement quand on lance le fichier directement:
    $ python models/pricing_model.py

    Il crée 4 cas de test avec des profils différents:
    1. Clio essence modeste (8k€ ARGUS, 10 ans)
    2. BMW X5 diesel riche (35k€ ARGUS, 2 ans)
    3. Tesla électrique aisé (25k€ ARGUS, 3 ans)
    4. Vieille Diesel pauvre (5k€ ARGUS, 15 ans)
    """

    # Créer le modèle
    model = create_model()

    # Données de test (4 profils différents)
    batch_size = 4
    test_input = {
        # Émissions CO2 en g/km
        # 120 = essence moyen, 200 = gros diesel, 0 = électrique, 150 = diesel
        'co2_per_km': torch.tensor([120.0, 200.0, 0.0, 150.0]),

        # Distance: Paris-Lyon pour tous
        'distance': torch.tensor([450.0, 450.0, 450.0, 450.0]),

        # Trafic: 1.0 = fluide, 1.5 = bouchon
        'traffic_factor': torch.tensor([1.0, 1.2, 1.0, 1.5]),

        # Passagers: covoiturage réduit l'impact
        'passengers': torch.tensor([2, 1, 3, 1]),

        # ARGUS: proxy de la capacité financière
        'argus_value': torch.tensor([8000.0, 35000.0, 25000.0, 5000.0]),

        # Âge du véhicule
        'car_age': torch.tensor([10, 2, 3, 15]),

        # Jour: 4 = vendredi
        'day_of_week': torch.tensor([4, 4, 6, 1]),

        # Type semaine: 0 = normal, 1 = vacances
        'week_type': torch.tensor([0, 0, 1, 0])
    }

    # Prédiction (sans calculer les gradients car on n'entraîne pas)
    with torch.no_grad():
        output = model(**test_input)

    # Affichage des résultats
    print("=" * 60)
    print("RÉSULTATS DE TARIFICATION BILLIE GREEN")
    print("=" * 60)

    for i in range(batch_size):
        print(f"\n{'='*40}")
        print(f"CAS {i+1}:")
        print(f"{'='*40}")
        print(f"  📊 ENTRÉES:")
        print(f"     CO2: {test_input['co2_per_km'][i]:.0f} g/km")
        print(f"     ARGUS: {test_input['argus_value'][i]:.0f}€")
        print(f"     Âge: {test_input['car_age'][i]:.0f} ans")
        print(f"     Passagers: {test_input['passengers'][i]}")
        print()
        print(f"  📈 SCORES:")
        print(f"     Score éco: {output['eco_score'][i]:.2f} (0=polluant, 1=propre)")
        print(f"     Score social: {output['social_score'][i]:.2f} (0=protégé, 1=aisé)")
        print()
        print(f"  💰 PRIX:")
        print(f"     Prix de base: {output['base_price'][i]:.2f}€")
        print(f"     Ajustement: {output['eco_adjustment'][i]*100:+.1f}%")
        print(f"     Prix final: {output['final_price'][i]:.2f}€")
