"""
Billie Green - Modèle de Tarification Intelligente
Architecture PyTorch pour une tarification éco-responsable et éthique
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class EcoSocialAttention(nn.Module):
    """
    Module d'attention qui pondère les critères écologiques et sociaux.
    Permet au modèle d'apprendre quels facteurs sont les plus importants
    pour une tarification équitable.
    """

    def __init__(self, input_dim: int, attention_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, input_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.attention(x)
        weighted_x = x * weights
        return weighted_x, weights


class EthicalConstraintLayer(nn.Module):
    """
    Couche de contrainte éthique qui s'assure que les utilisateurs
    à faible revenu avec des voitures polluantes ne soient pas pénalisés.

    Principe: Le malus écologique est atténué proportionnellement
    au niveau socio-économique défavorable.
    """

    def __init__(self, sensitivity: float = 0.7):
        super().__init__()
        self.sensitivity = nn.Parameter(torch.tensor(sensitivity))

    def forward(
        self,
        eco_score: torch.Tensor,      # Score écologique (0=polluant, 1=vert)
        social_score: torch.Tensor,   # Score social (0=défavorisé, 1=aisé)
        base_price: torch.Tensor      # Prix de base
    ) -> torch.Tensor:
        """
        Calcule le prix ajusté en appliquant une contrainte éthique.

        Si eco_score bas ET social_score bas -> pas de malus
        Si eco_score bas ET social_score haut -> malus normal
        Si eco_score haut -> bonus écologique
        """
        # Protection éthique: atténue le malus pour les défavorisés
        ethical_protection = 1 - (1 - social_score) * self.sensitivity

        # Calcul du modificateur écologique
        # eco_score proche de 0 = polluant = malus potentiel
        # eco_score proche de 1 = vert = bonus
        eco_modifier = (eco_score - 0.5) * 0.3  # Entre -0.15 et +0.15

        # Application de la protection éthique sur le malus uniquement
        adjusted_modifier = torch.where(
            eco_modifier < 0,  # Si c'est un malus
            eco_modifier * ethical_protection,  # Atténuer selon le profil social
            eco_modifier  # Garder le bonus tel quel
        )

        return base_price * (1 + adjusted_modifier)


class CarbonImpactEncoder(nn.Module):
    """
    Encode l'impact carbone du trajet en fonction de:
    - Émissions CO2 du véhicule (g/km)
    - Distance du trajet
    - Conditions de trafic (embouteillages)
    - Nombre de passagers (mutualisation)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        co2_per_km: torch.Tensor,    # g/km
        distance: torch.Tensor,       # km
        traffic_factor: torch.Tensor, # 1.0 = normal, 1.5 = embouteillage
        passengers: torch.Tensor      # nombre de passagers
    ) -> torch.Tensor:
        """
        Retourne un score écologique entre 0 (très polluant) et 1 (très vert).
        """
        # Normalisation
        co2_norm = torch.clamp(co2_per_km / 250.0, 0, 1)  # 250g/km = très polluant
        dist_norm = torch.clamp(distance / 1000.0, 0, 1)  # 1000km = très long
        traffic_norm = torch.clamp((traffic_factor - 1) / 0.5, 0, 1)  # Impact embouteillage
        pass_norm = torch.clamp(passengers / 5.0, 0, 1)  # Mutualisation

        # Le score est inversé: plus c'est polluant, plus le score est bas
        features = torch.stack([
            1 - co2_norm,        # Inverser: faible CO2 = bon score
            1 - dist_norm * 0.3, # Distance a un impact modéré
            1 - traffic_norm,    # Embouteillages = mauvais
            pass_norm * 0.5 + 0.5  # Plus de passagers = meilleur score
        ], dim=-1)

        return self.encoder(features)


class SocialProfileEncoder(nn.Module):
    """
    Encode le profil socio-économique de l'utilisateur.
    Utilise la valeur ARGUS comme proxy de la capacité financière.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        argus_value: torch.Tensor,  # Valeur ARGUS du véhicule (€)
        car_age: torch.Tensor       # Âge du véhicule (années)
    ) -> torch.Tensor:
        """
        Retourne un score social entre 0 (défavorisé) et 1 (aisé).

        Hypothèses:
        - Valeur ARGUS faible + voiture vieille = profil potentiellement défavorisé
        - Valeur ARGUS élevée = capacité financière plus importante
        """
        # Normalisation
        argus_norm = torch.clamp(argus_value / 40000.0, 0, 1)  # 40k€ = voiture chère
        age_norm = torch.clamp(car_age / 15.0, 0, 1)  # 15 ans = vieille voiture

        features = torch.stack([
            argus_norm,
            1 - age_norm * 0.3  # Voiture vieille a un léger impact
        ], dim=-1)

        return self.encoder(features)


class TemporalEncoder(nn.Module):
    """
    Encode les facteurs temporels qui influencent la demande.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Embeddings pour jour de la semaine
        self.day_embedding = nn.Embedding(7, hidden_dim // 2)
        # Embeddings pour le type de semaine (vacances, etc.)
        self.week_type_embedding = nn.Embedding(4, hidden_dim // 2)

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        day_of_week: torch.Tensor,  # 0-6 (lundi-dimanche)
        week_type: torch.Tensor     # 0=normal, 1=vacances, 2=pont, 3=été
    ) -> torch.Tensor:
        """
        Retourne un facteur de demande entre 0 (faible) et 1 (forte).
        """
        day_emb = self.day_embedding(day_of_week.long())
        week_emb = self.week_type_embedding(week_type.long())

        combined = torch.cat([day_emb, week_emb], dim=-1)
        return self.encoder(combined)


class BillieGreenPricingModel(nn.Module):
    """
    Modèle principal de tarification Billie Green.

    Architecture:
    1. Encodage des caractéristiques (carbone, social, temporel)
    2. Attention éco-sociale pour pondérer les facteurs
    3. Calcul du prix de base selon la distance
    4. Application des contraintes éthiques
    5. Ajustement final avec bonus/malus
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        ethical_sensitivity: float = 0.7
    ):
        super().__init__()

        # Encodeurs
        self.carbon_encoder = CarbonImpactEncoder(hidden_dim)
        self.social_encoder = SocialProfileEncoder(hidden_dim // 2)
        self.temporal_encoder = TemporalEncoder(hidden_dim // 2)

        # Module d'attention
        self.attention = EcoSocialAttention(3)  # 3 scores: eco, social, temporal

        # Contrainte éthique
        self.ethical_constraint = EthicalConstraintLayer(ethical_sensitivity)

        # Réseau de prix final
        self.price_network = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 3 scores + distance
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Assure un prix positif
        )

        # Paramètres de tarification de base
        self.base_price_per_km = nn.Parameter(torch.tensor(0.15))  # €/km
        self.min_price = nn.Parameter(torch.tensor(20.0))  # Prix minimum

    def forward(
        self,
        co2_per_km: torch.Tensor,
        distance: torch.Tensor,
        traffic_factor: torch.Tensor,
        passengers: torch.Tensor,
        argus_value: torch.Tensor,
        car_age: torch.Tensor,
        day_of_week: torch.Tensor,
        week_type: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calcule le prix recommandé avec toutes les composantes.

        Returns:
            Dict contenant:
            - final_price: Prix final recommandé
            - base_price: Prix de base (avant ajustements)
            - eco_score: Score écologique (0-1)
            - social_score: Score social (0-1)
            - eco_bonus: Bonus/malus écologique (peut être négatif pour bonus)
            - ethical_adjustment: Ajustement éthique appliqué
        """
        # 1. Encoder les caractéristiques
        eco_score = self.carbon_encoder(co2_per_km, distance, traffic_factor, passengers)
        social_score = self.social_encoder(argus_value, car_age)
        temporal_score = self.temporal_encoder(day_of_week, week_type)

        # 2. Calculer le prix de base selon la distance
        base_price = distance * self.base_price_per_km
        base_price = torch.maximum(base_price, self.min_price)

        # 3. Combiner les scores avec attention
        scores = torch.cat([eco_score, social_score, temporal_score], dim=-1)
        weighted_scores, attention_weights = self.attention(scores)

        # 4. Appliquer la contrainte éthique
        ethical_price = self.ethical_constraint(eco_score, social_score, base_price)

        # 5. Ajustement temporel (demande)
        demand_factor = 1 + (temporal_score.squeeze(-1) - 0.5) * 0.2  # ±10% selon demande

        # 6. Prix final
        final_price = ethical_price * demand_factor

        # Calcul des composantes pour transparence
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
        Génère une explication textuelle de la tarification.
        """
        eco = output['eco_score'].item()
        social = output['social_score'].item()
        adjustment = output['eco_adjustment'].item()

        explanation = []

        # Score écologique
        if eco > 0.7:
            explanation.append("Votre trajet est éco-responsable")
        elif eco < 0.3:
            explanation.append("Impact carbone élevé")
        else:
            explanation.append("Impact carbone modéré")

        # Ajustement
        if adjustment < -0.05:
            explanation.append(f"Bonus écologique de {abs(adjustment)*100:.1f}%")
        elif adjustment > 0.05:
            if social < 0.4:
                explanation.append("Malus écologique atténué (profil protégé)")
            else:
                explanation.append(f"Contribution écologique de {adjustment*100:.1f}%")

        return " | ".join(explanation)


class PricingModelTrainer:
    """
    Classe pour entraîner le modèle de tarification.
    """

    def __init__(
        self,
        model: BillieGreenPricingModel,
        learning_rate: float = 1e-3
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        target_prices: torch.Tensor,
        eco_targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcule la perte multi-objectif:
        1. MSE sur les prix
        2. Régularisation pour encourager les comportements verts
        3. Contrainte éthique
        """
        # Perte principale sur les prix
        price_loss = F.mse_loss(predictions['final_price'], target_prices)

        # Régularisation écologique: encourager les bonus pour les verts
        eco_reg = -0.1 * torch.mean(predictions['eco_score'] * predictions['eco_adjustment'])

        # Contrainte éthique: pénaliser les gros malus pour les défavorisés
        ethical_violation = torch.relu(
            -predictions['eco_adjustment'] *  # Malus (négatif = bonus)
            (1 - predictions['social_score']) *  # Score défavorisé
            0.5  # Sensibilité
        )
        ethical_loss = torch.mean(ethical_violation)

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
        Effectue une étape d'entraînement.
        """
        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(**batch)
        loss, loss_components = self.compute_loss(predictions, target_prices)

        loss.backward()
        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            **loss_components
        }


def create_model(
    hidden_dim: int = 64,
    ethical_sensitivity: float = 0.7
) -> BillieGreenPricingModel:
    """
    Factory function pour créer un modèle.
    """
    model = BillieGreenPricingModel(
        hidden_dim=hidden_dim,
        ethical_sensitivity=ethical_sensitivity
    )
    return model


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer le modèle
    model = create_model()

    # Données de test
    batch_size = 4
    test_input = {
        'co2_per_km': torch.tensor([120.0, 200.0, 0.0, 150.0]),  # g/km
        'distance': torch.tensor([450.0, 450.0, 450.0, 450.0]),  # Paris-Lyon
        'traffic_factor': torch.tensor([1.0, 1.2, 1.0, 1.5]),    # Trafic
        'passengers': torch.tensor([2, 1, 3, 1]),                 # Passagers
        'argus_value': torch.tensor([8000.0, 35000.0, 25000.0, 5000.0]),  # €
        'car_age': torch.tensor([10, 2, 3, 15]),                  # Années
        'day_of_week': torch.tensor([4, 4, 6, 1]),                # Jour
        'week_type': torch.tensor([0, 0, 1, 0])                   # Type semaine
    }

    # Prédiction
    with torch.no_grad():
        output = model(**test_input)

    print("=== Résultats de tarification ===")
    for i in range(batch_size):
        print(f"\nCas {i+1}:")
        print(f"  CO2: {test_input['co2_per_km'][i]:.0f} g/km | ARGUS: {test_input['argus_value'][i]:.0f}€")
        print(f"  Score éco: {output['eco_score'][i]:.2f} | Score social: {output['social_score'][i]:.2f}")
        print(f"  Prix de base: {output['base_price'][i]:.2f}€")
        print(f"  Ajustement: {output['eco_adjustment'][i]*100:+.1f}%")
        print(f"  Prix final: {output['final_price'][i]:.2f}€")
