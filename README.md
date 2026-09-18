# MLOps Observability Dashboard

Un outil de visualisation branché au cœur d'une chaîne MLOps : historique des runs d'entraînement, qualité et dérive des données, traçabilité donnée → modèle. Construit comme projet portfolio ciblant le stage **Thales** (Ingénieur Développement et intégration d'un outil IA dans une chaîne MLOps, CortAIx Labs) — la mission de l'offre est quasiment reprise mot pour mot : *"développer et intégrer un outil moderne de visualisation et d'analyse de données au cœur d'une chaîne MLOps"*.

## Ce que fait le projet

Une chaîne MLOps polyglotte-persistance complète, avec un dashboard qui l'observe :

1. **Pipeline d'entraînement** (`src/train.py`) : récupère des données météo réelles (Open-Meteo), fait tourner des **contrôles qualité et de dérive** avant d'entraîner, entraîne un XGBoost, et trace tout.
2. **MLflow** (backend Postgres) : suivi d'expériences — paramètres, métriques, modèle versionné à chaque run.
3. **MongoDB** : événements non structurés — rapports qualité (schéma, valeurs manquantes, dérive) et traçabilité (quel run vient de quelles données, avec quel résultat qualité).
4. **Dashboard Streamlit** (lecture seule) : historique des runs, évolution des métriques, alertes qualité/dérive, traçabilité donnée → modèle.
5. **Docker Compose** pour tout faire tourner localement ; **manifestes Kubernetes** pour la même architecture (voir `k8s/README.md` pour ce qui a été testé vs. seulement écrit).

## Pourquoi SQL *et* NoSQL (polyglot persistence, pas un choix arbitraire)

- **Postgres** (via MLflow) pour les runs/métriques : données structurées, un schéma stable, on veut des requêtes/agrégations fiables dessus.
- **MongoDB** pour les événements qualité/traçabilité : le contenu d'un rapport de qualité change de forme d'une vérification à l'autre (colonnes manquantes, violations de plage, dérive détectée...) — le forcer dans des tables relationnelles imposerait des migrations de schéma à chaque nouvelle vérification ajoutée. Un document JSON schema-less colle mieux à ce que c'est réellement.

## Un vrai bug rencontré et corrigé pendant le développement

MLflow 3.x rejette par défaut toute requête dont l'en-tête `Host` n'est pas dans une liste blanche (protection anti *DNS rebinding*). Le nom de service Docker Compose `mlflow` n'y est pas par défaut → toutes les requêtes du pipeline d'entraînement échouaient avec `403 Invalid Host header`, jusqu'à ajouter explicitement `--allowed-hosts mlflow,mlflow:5000` au démarrage du serveur (voir `Dockerfile.mlflow`). Pas un problème qu'on trouve en lisant la doc en diagonale — un bon exemple à raconter en entretien sur "debugger une brique MLOps en conditions réelles".

## Résultats obtenus (deux runs réels, données réelles)

| Run | Période | Accuracy | ROC-AUC | Qualité des données |
|---|---|---|---|---|
| 1 | 2026-08-01 → 2026-09-17 | 0.80 | 0.84 | ✅ passed |
| 2 | 2026-06-01 → 2026-08-01 | 0.84 | 0.85 | ⚠️ dérive détectée sur `precipitation` (+26%) |

Le run 2 illustre le système de dérive en action avec un vrai signal météo (les précipitations diffèrent réellement entre juin-août et août-septembre) — pas une donnée simulée pour la démo.

## Structure du projet

```
mlops-observability-dashboard/
├── src/
│   ├── data.py          # Données météo réelles (Open-Meteo) + label proxy (même méthodologie que flight-disruption-risk)
│   ├── quality.py       # Contrôles qualité/dérive (fonctions pures, testées unitairement)
│   ├── events.py        # Accès MongoDB (rapports qualité + traçabilité)
│   ├── train.py         # Pipeline complet : data -> qualité -> entraînement -> MLflow -> traçabilité
│   └── dashboard.py     # Dashboard Streamlit (lecture seule)
├── k8s/                  # Manifestes Kubernetes (voir k8s/README.md : testé vs. écrit)
├── tests/                 # 21 tests pytest (quality, data, events via mongomock)
├── docker-compose.yml
├── Dockerfile.mlflow
├── Dockerfile.dashboard
└── requirements.txt
```

## Utilisation

### Avec Docker Compose (testé et fonctionnel)
```bash
docker compose up -d postgres mongo mlflow dashboard

# Lancer un entraînement (peut se relancer autant de fois que voulu)
docker compose run --rm trainer src/train.py --start 2026-07-01 --end 2026-09-17

# Dashboard : http://localhost:8501
# MLflow UI : http://localhost:5000
```

### En local sans Docker
```bash
pip install -r requirements.txt
# Nécessite un Postgres et un Mongo accessibles (ou docker compose up -d postgres mongo)
mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow --host 0.0.0.0 --allowed-hosts localhost &
python3 src/train.py
streamlit run src/dashboard.py
```

### Tests
```bash
pytest tests/ -v
```

## Ce qui a été vérifié en conditions réelles dans cet environnement
- `docker compose build` + `docker compose up` : les 4 services démarrent et passent healthy
- Deux runs d'entraînement réels exécutés via `docker compose run trainer`, avec de vraies données Open-Meteo
- MLflow : runs, paramètres et métriques bien persistés dans Postgres (vérifié via l'API MLflow)
- MongoDB : rapports qualité et événements de traçabilité bien écrits et relus (vérifié directement)
- **Détection de dérive fonctionnelle sur un vrai signal** (précipitations, +26% entre les deux périodes)
- Le module `dashboard.py` a été exécuté directement contre la stack réelle (hors interface Streamlit) pour confirmer que les fonctions de chargement de données retournent bien les 2 runs et les 2 événements attendus
- 21 tests pytest, tous passants
- Un bug réel (MLflow / DNS rebinding protection) trouvé et corrigé pendant le développement, documenté ci-dessus

## Limite assumée
Les manifestes Kubernetes (`k8s/`) sont écrits pour reproduire fidèlement l'architecture Docker Compose testée, mais n'ont pas été appliqués sur un cluster réel dans cet environnement (aucun `kind`/`minikube` disponible). Voir `k8s/README.md` pour comment les tester réellement.
