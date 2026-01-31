"""
Script de test pour l'API Camion Symone
Simule des appels HTTP pour tester les nouveaux endpoints
"""

import httpx
import asyncio
import json
from typing import Dict


BASE_URL = "http://localhost:8000"


async def test_truck_specs():
    """Test GET /api/truck/specs"""
    print("\n" + "="*80)
    print("TEST 1: Récupération des spécifications du camion")
    print("="*80)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/truck/specs")
        
    print(f"Status: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    assert response.status_code == 200
    assert data["fuel_type"] == "BIOGAZ"
    print("✓ Test réussi")


async def test_truck_calculate():
    """Test POST /api/truck/calculate"""
    print("\n" + "="*80)
    print("TEST 2: Calcul de coût Paris -> Lyon")
    print("="*80)
    
    payload = {
        "origin": "Paris",
        "destination": "Lyon"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/truck/calculate",
            json=payload
        )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    # Affichage formaté
    print(f"\n📍 Trajet: {data['trip']['origin']} → {data['trip']['destination']}")
    print(f"📏 Distance: {data['trip']['distance_km']} km")
    print(f"\n💰 Coûts:")
    print(f"   • Péage: {data['toll']['price_euros']}€ ({data['toll']['description']})")
    print(f"   • Carburant: {data['fuel']['fuel_consumption_kg']} kg × {data['fuel']['fuel_price_per_kg']}€/kg = {data['fuel']['fuel_cost_euros']}€")
    print(f"   • TOTAL: {data['total_cost_euros']}€ ({data['cost_per_km_euros']}€/km)")
    print(f"\n🌱 Environnement:")
    print(f"   • CO2 émis: {data['environmental']['co2_total_kg']} kg ({data['environmental']['co2_per_km_g']} g/km)")
    print(f"   • Carburant: {data['environmental']['fuel_type']}")
    print(f"\n📊 Comparaison diesel:")
    print(f"   • Coût équivalent diesel: {data['comparison']['diesel_equivalent_cost']}€")
    print(f"   • Économie: {data['comparison']['savings_vs_diesel_euros']}€ ({data['comparison']['savings_percent']}%)")
    
    assert response.status_code == 200
    assert data["total_cost_euros"] > 0
    print("\n✓ Test réussi")


async def test_truck_calculate_custom():
    """Test POST /api/truck/calculate avec paramètres personnalisés"""
    print("\n" + "="*80)
    print("TEST 3: Calcul avec consommation et prix personnalisés")
    print("="*80)
    
    payload = {
        "origin": "Lyon",
        "destination": "Marseille",
        "custom_consumption": 22.0,  # Meilleure consommation
        "custom_biogaz_price": 0.75   # Prix réduit
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/truck/calculate",
            json=payload
        )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"\n📍 Trajet: {data['trip']['origin']} → {data['trip']['destination']}")
    print(f"⚙️  Paramètres personnalisés:")
    print(f"   • Consommation: {data['fuel']['consumption_per_100km']} kg/100km")
    print(f"   • Prix biogaz: {data['fuel']['fuel_price_per_kg']}€/kg")
    print(f"\n💰 Total: {data['total_cost_euros']}€")
    
    assert response.status_code == 200
    assert data["fuel"]["consumption_per_100km"] == 22.0
    assert data["fuel"]["fuel_price_per_kg"] == 0.75
    print("\n✓ Test réussi")


async def test_toll_prices():
    """Test GET /api/truck/toll-prices"""
    print("\n" + "="*80)
    print("TEST 4: Liste des prix de péages")
    print("="*80)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/truck/toll-prices")
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"\nClasse de péage: {data['toll_class']}")
    print(f"Description: {data['description']}")
    print(f"Nombre de trajets: {data['count']}")
    print(f"\nExemples de tarifs:")
    
    for price_info in data['prices'][:5]:  # Afficher les 5 premiers
        print(f"   • {price_info['route']}: {price_info['price_euros']}€")
    
    assert response.status_code == 200
    assert data["count"] > 0
    print("\n✓ Test réussi")


async def test_compare_truck_car():
    """Test POST /api/truck/compare-with-car"""
    print("\n" + "="*80)
    print("TEST 5: Comparaison Camion vs Voiture")
    print("="*80)
    
    payload = {
        "trip": {
            "origin": "Paris",
            "destination": "Lyon"
        },
        "vehicle": {
            "brand": "RENAULT",
            "model": "CLIO",
            "energy": "ESSENCE"
        },
        "passengers": 4
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/api/truck/compare-with-car",
            json=payload
        )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"\n🚛 CAMION SYMONE:")
    print(f"   • Coût total: {data['truck']['total_cost']}€")
    print(f"   • Coût/km: {data['truck']['cost_per_km']}€")
    print(f"   • CO2: {data['truck']['co2_kg']} kg")
    print(f"   • Carburant: {data['truck']['fuel_type']}")
    
    print(f"\n🚗 VOITURE ({payload['passengers']} passagers):")
    print(f"   • Prix/passager: {data['car']['price_per_passenger']}€")
    print(f"   • Coût total: {data['car']['total_cost']}€")
    print(f"   • Catégorie CO2: {data['car']['co2_category']}")
    print(f"   • Carburant: {data['car']['fuel_type']}")
    
    print(f"\n📊 COMPARAISON:")
    print(f"   • Moins cher: {'Camion' if data['comparison']['truck_cheaper'] else 'Voiture'}")
    print(f"   • Différence: {data['comparison']['cost_difference_euros']}€")
    print(f"   • Ratio: {data['comparison']['truck_vs_car_ratio']}")
    
    assert response.status_code == 200
    print("\n✓ Test réussi")


async def test_multiple_routes():
    """Test plusieurs trajets différents"""
    print("\n" + "="*80)
    print("TEST 6: Calcul de plusieurs trajets")
    print("="*80)
    
    routes = [
        ("Paris", "Marseille"),
        ("Lyon", "Nice"),
        ("Grenoble", "Montpellier"),
        ("Dijon", "Lyon"),
        ("Paris", "Dijon")
    ]
    
    results = []
    
    async with httpx.AsyncClient() as client:
        for origin, destination in routes:
            response = await client.post(
                f"{BASE_URL}/api/truck/calculate",
                json={"origin": origin, "destination": destination}
            )
            
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "route": f"{origin} → {destination}",
                    "distance": data["trip"]["distance_km"],
                    "cost": data["total_cost_euros"],
                    "cost_per_km": data["cost_per_km_euros"]
                })
    
    print(f"\n{'Route':<30} {'Distance':<12} {'Coût total':<12} {'€/km'}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['route']:<30} {r['distance']:>6.0f} km    {r['cost']:>7.2f} €     {r['cost_per_km']:.3f}")
    
    assert len(results) == len(routes)
    print(f"\n✓ Test réussi - {len(results)} trajets calculés")


async def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "="*80)
    print("🚀 DÉMARRAGE DES TESTS API CAMION SYMONE")
    print("="*80)
    print(f"Base URL: {BASE_URL}")
    print("Assurez-vous que l'API est lancée (uvicorn main:app)")
    
    try:
        # Vérifier que l'API est accessible
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{BASE_URL}/api/health")
            assert health.status_code == 200
            print("✓ API accessible")
    except Exception as e:
        print(f"✗ Erreur: L'API n'est pas accessible")
        print(f"  Lancez l'API avec: uvicorn backend.main:app --reload")
        return
    
    # Exécuter les tests
    tests = [
        test_truck_specs,
        test_truck_calculate,
        test_truck_calculate_custom,
        test_toll_prices,
        test_compare_truck_car,
        test_multiple_routes
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test échoué: {str(e)}")
            failed += 1
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    print(f"✓ Réussis: {passed}")
    print(f"✗ Échoués: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 Tous les tests sont passés avec succès!")
    else:
        print(f"\n⚠️  {failed} test(s) ont échoué")


if __name__ == "__main__":
    # Note: Ce script nécessite que l'API soit lancée
    # Lancez d'abord: uvicorn backend.main:app --reload
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║            TESTS API CAMION SYMONE - BILLIE GREEN                ║
║                                                                  ║
║  Ce script teste les nouveaux endpoints pour le calculateur     ║
║  de coûts des camions au biogaz de Symone                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  PRÉREQUIS:")
    print("   L'API doit être lancée sur http://localhost:8000")
    print("   Commande: uvicorn backend.main:app --reload")
    print("\nAppuyez sur Entrée pour continuer ou Ctrl+C pour annuler...")
    
    try:
        input()
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n\n✋ Tests annulés par l'utilisateur")
