# Manifestes Kubernetes

**Statut honnête** : ces manifestes sont écrits pour reproduire exactement l'architecture validée avec `docker-compose.yml` (mêmes images, mêmes variables d'environnement, mêmes dépendances entre services), mais n'ont **pas** été appliqués sur un cluster réel dans cet environnement — aucun cluster Kubernetes local (`kind`/`minikube`) n'était disponible. Toute la chaîne a en revanche été testée bout en bout avec Docker Compose : postgres + mongo + mlflow + dashboard + un vrai run d'entraînement, deux fois, avec détection de dérive vérifiée sur de vraies données.

## Pour tester réellement ces manifestes

```bash
# Installer un cluster local (une seule fois)
brew install kind
kind create cluster --name mlops-demo

# Construire et charger les images locales dans le cluster kind
docker build -t mlops-dashboard:local -f ../Dockerfile.dashboard ..
docker build -t mlops-mlflow:local -f ../Dockerfile.mlflow ..
kind load docker-image mlops-dashboard:local --name mlops-demo
kind load docker-image mlops-mlflow:local --name mlops-demo

# Appliquer les manifestes
kubectl apply -f .

# Vérifier
kubectl get pods
kubectl port-forward svc/dashboard 8501:8501
```

## Contenu

| Fichier | Rôle |
|---|---|
| `postgres.yaml` | Deployment + Service + PVC pour le backend store MLflow |
| `mongo.yaml` | Deployment + Service + PVC pour les événements qualité/traçabilité |
| `mlflow.yaml` | Deployment + Service pour le serveur de tracking MLflow |
| `dashboard.yaml` | Deployment + Service pour le dashboard Streamlit |
| `secrets.yaml` | Identifiants Postgres (à remplacer avant tout usage réel — valeurs de démo ici) |

Différence volontaire avec docker-compose : les identifiants Postgres passent par un `Secret` Kubernetes plutôt que d'être en clair dans le manifeste, ce qui est la pratique attendue en production (contrairement à docker-compose.yml qui les met en clair pour la simplicité d'un environnement de démo local).
