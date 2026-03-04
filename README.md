<p align="center">
  <img src="docs/images/banner.png" alt="MLOps Iris Project Banner" width="800"/>
</p>

<h1 align="center">🌿 MLOps Final Project — Iris Classifier</h1>

<p align="center">
  <strong>End-to-end Machine Learning Operations pipeline with full traceability, automated CI/CD, and multi-environment deployment.</strong>
</p>

<p align="center">
  <a href="https://github.com/camm22/MLOps_final_project/actions"><img src="https://img.shields.io/github/actions/workflow/status/camm22/MLOps_final_project/ci-pr-dev.yml?label=CI%20Pipeline&style=for-the-badge&logo=githubactions" alt="CI"></a>
  <a href="https://dagshub.com/camm22/MLOps_final_project"><img src="https://img.shields.io/badge/MLflow-DagsHub-ff6f00?style=for-the-badge&logo=mlflow" alt="MLflow"></a>
  <a href="https://iris-frontend-staging-lrrp.onrender.com"><img src="https://img.shields.io/badge/Staging-Live-blue?style=for-the-badge&logo=render" alt="Staging"></a>
  <a href="https://iris-frontend-production.onrender.com"><img src="https://img.shields.io/badge/Production-Live-green?style=for-the-badge&logo=render" alt="Production"></a>
</p>

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture Diagram](#-architecture-diagram)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [How We Built It — Step by Step](#-how-we-built-it--step-by-step)
6. [CI/CD Pipeline — Detailed Explanation](#-cicd-pipeline--detailed-explanation)
7. [Model Promotion — From Training to Production](#-model-promotion--from-training-to-production)
8. [Quality Gate](#-quality-gate)
9. [Data Versioning with DVC](#-data-versioning-with-dvc)
10. [Test Pyramid](#-test-pyramid)
11. [Deployment Architecture on Render](#-deployment-architecture-on-render)
12. [Reproducibility Instructions](#-reproducibility-instructions)
13. [Demonstration Plan](#-demonstration-plan)
14. [GitHub Secrets Reference](#-github-secrets-reference)
15. [Live URLs](#-live-urls)

---

## 🎯 Project Overview

This project implements a **complete MLOps pipeline** for an Iris flower classification model. It demonstrates industry best practices for putting Machine Learning models into production, including:

- **ML Model Training** with full experiment tracking (MLflow on DagsHub)
- **Data Versioning** with DVC (Data Version Control)
- **REST API** serving predictions via FastAPI
- **React Frontend** for user-friendly interaction
- **Dockerized** backend and frontend
- **3 CI/CD Pipelines** with GitHub Actions (PR validation → Staging deploy → Production promotion)
- **Quality Gate** ensuring only high-performing models reach production
- **Multi-environment deployment** on Render (Staging + Production)
- **Git Flow** branching strategy (feature → dev → staging → main)

> **Screenshot placeholder:**
> ![Project Overview](docs/images/project_overview.png)
> *Caption: The running application — frontend making predictions through the API*

---

## 🏗 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DEVELOPMENT                                     │
│                                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐                    │
│  │  Feature     │───▶│   Dev        │───▶│  Staging     │───▶ Main          │
│  │  Branch      │ PR │   Branch     │    │  Branch      │     Branch        │
│  └─────────────┘    └──────────────┘    └──────────────┘                    │
│         │                  │                    │              │              │
│         ▼                  ▼                    ▼              ▼              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  ┌───────────┐    │
│  │ Git + DVC   │    │  CI Pipeline │    │ Deploy Stag. │  │ Promote   │    │
│  │ Commit      │    │  (Tests +    │    │ Pipeline     │  │ Pipeline  │    │
│  └─────────────┘    │  Docker)     │    └──────────────┘  └───────────┘    │
│                      └──────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────────┘
         │                                        │                │
         ▼                                        ▼                ▼
┌──────────────────┐                   ┌────────────────────────────────────┐
│    DagsHub       │                   │           Render.com               │
│  ┌────────────┐  │                   │  ┌──────────────┬───────────────┐  │
│  │  MLflow    │  │                   │  │  STAGING     │  PRODUCTION   │  │
│  │  Tracking  │  │                   │  │              │               │  │
│  │  Server    │  │                   │  │ Backend API  │ Backend API   │  │
│  ├────────────┤  │                   │  │ (@staging)   │ (@production) │  │
│  │  Model     │  │◀──── loads ──────▶│  │              │               │  │
│  │  Registry  │  │     model         │  │ Frontend     │ Frontend      │  │
│  ├────────────┤  │                   │  │ (React)      │ (React)       │  │
│  │  DVC       │  │                   │  └──────────────┴───────────────┘  │
│  │  Remote    │  │                   └────────────────────────────────────┘
│  │  Storage   │  │
│  └────────────┘  │
└──────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATA & MODEL FLOW                                    │
│                                                                              │
│  iris.csv ──▶ DVC track ──▶ train.py ──▶ MLflow log ──▶ Model Registry     │
│                                              │                │              │
│                                         Experiments      @staging            │
│                                         (params,         @production         │
│                                          metrics,           │                │
│                                          artifacts)    Quality Gate          │
│                                                        (accuracy ≥ 0.90)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

> **Screenshot placeholder:**
> ![Architecture](docs/images/architecture_diagram.png)
> *Caption: High-level architecture of the MLOps pipeline (you can recreate this as a clean diagram in draw.io or Mermaid)*

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Training** | scikit-learn, pandas | Logistic Regression on Iris dataset |
| **Experiment Tracking** | MLflow (on DagsHub) | Log params, metrics, artifacts, model registry |
| **Data Versioning** | DVC + DagsHub Remote | Version control for datasets |
| **Backend API** | FastAPI + Uvicorn | REST API serving model predictions |
| **Frontend** | React 19 + Vite 7 | User interface for predictions |
| **Containerization** | Docker | Backend & frontend Dockerfiles |
| **CI/CD** | GitHub Actions (3 workflows) | Automated testing, deployment, promotion |
| **Deployment** | Render (4 services) | Cloud hosting for staging + production |
| **Quality Gate** | Custom Python script | Accuracy threshold validation |
| **Branching** | Git Flow | feature → dev → staging → main |

---

## 📂 Project Structure

```
MLOps_final_project/
│
├── 📁 .github/workflows/          # CI/CD Pipelines
│   ├── ci-pr-dev.yml               # Pipeline 1: PR validation to dev
│   ├── deploy-staging.yml          # Pipeline 2: Auto-deploy on staging push
│   └── promote-to-production.yml   # Pipeline 3: Manual promotion to production
│
├── 📁 backend/                     # FastAPI Backend
│   ├── main.py                     # API endpoints (/predict, /)
│   └── Dockerfile                  # Backend Docker image
│
├── 📁 frontend/                    # React Frontend
│   ├── src/
│   │   ├── App.jsx                 # Main React component
│   │   ├── App.css                 # Dark theme styling
│   │   ├── index.css               # Global styles
│   │   └── main.jsx                # Entry point
│   ├── Dockerfile                  # Multi-stage build (Node → Nginx)
│   ├── nginx.conf                  # SPA routing config
│   ├── vite.config.js              # Vite + dev proxy config
│   └── package.json                # Dependencies
│
├── 📁 data/                        # Dataset (DVC tracked)
│   ├── iris.csv                    # ← NOT in Git, pulled via DVC
│   ├── iris.csv.dvc                # DVC metadata file
│   └── .gitignore                  # Ignores iris.csv (tracked by DVC)
│
├── 📁 scripts/
│   └── train.py                    # ML training script with MLflow logging
│
├── 📁 tests/                       # Test Pyramid
│   ├── test_unit.py                # Unit tests (data validation, git hash)
│   ├── test_integration.py         # Integration tests (API endpoints)
│   └── test_e2e.py                 # End-to-end tests (full flow)
│
├── check_quality.py                # Quality Gate script (accuracy ≥ 0.90)
├── requirements.txt                # Python dependencies
├── .dvcignore                      # DVC ignore patterns
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

---

## 🔨 How We Built It — Step by Step

### Step 1: Project Initialization & Data Setup

We started by creating the project structure and initializing all necessary tools:

```bash
# Initialize Git repository
git init
git remote add origin https://github.com/camm22/MLOps_final_project.git

# Create branch structure
git checkout -b dev
git checkout -b staging
git checkout -b main

# Initialize DVC for data versioning
pip install dvc dvc-s3 dagshub
dvc init

# Configure DagsHub as DVC remote
dvc remote add origin https://dagshub.com/camm22/MLOps_final_project.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user camm22
dvc remote modify origin --local password <DAGSHUB_TOKEN>
```

We then generated the Iris dataset using `sklearn.datasets.load_iris()` and saved it as `data/iris.csv`. The file was tracked with DVC:

```bash
dvc add data/iris.csv    # Creates data/iris.csv.dvc + data/.gitignore
dvc push -r origin       # Pushes the actual data to DagsHub storage
git add data/iris.csv.dvc data/.gitignore
git commit -m "Track iris dataset with DVC"
```

> **Screenshot placeholder:**
> ![DVC Init](docs/images/step1_dvc_init.png)
> *Caption: DVC initialized with DagsHub as remote storage*

### Step 2: ML Training Script with Full Traceability

We created `scripts/train.py` which trains a **Logistic Regression** model on the Iris dataset with complete MLOps traceability:

- **MLflow experiment tracking** via DagsHub integration
- **Parameters logged**: `C`, `max_iter`
- **Metrics logged**: `accuracy`
- **Traceability tags**: `git_commit` (current commit hash), `dvc_data_version` (DVC URL of the dataset)
- **Model registered** in MLflow Model Registry as `iris_logistic_model`

```python
# Key traceability: every training run records which code AND data version was used
commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
data_url = dvc.api.get_url('data/iris.csv')
mlflow.log_param("git_commit", commit_hash)
mlflow.log_param("dvc_data_version", data_url)
```

After running the training script:
```bash
python scripts/train.py
# Output: Entraînement réussi ! Accuracy: 1.0000
```

> **Screenshot placeholder:**
> ![MLflow Run](docs/images/step2_mlflow_run.png)
> *Caption: MLflow experiment run on DagsHub showing params, metrics, and artifacts*

> **Screenshot placeholder:**
> ![Model Registry](docs/images/step2_model_registry.png)
> *Caption: Model registered in MLflow Model Registry with version number*

### Step 3: Backend API with FastAPI

We built a REST API using **FastAPI** that loads the trained model directly from the MLflow Model Registry:

```python
# The model is loaded at startup from the registry
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}{MODEL_STAGE}")
```

The API has two endpoints:
- `GET /` — Health check (returns model status and stage)
- `POST /predict` — Takes feature values, returns prediction

The `MODEL_STAGE` environment variable controls which model version is loaded:
- **Staging backend** → loads `@staging` alias
- **Production backend** → loads `@production` alias (set via `MODEL_STAGE=@production`)

CORS middleware is added to allow the frontend to call the API from a different domain.

> **Screenshot placeholder:**
> ![API Health](docs/images/step3_api_health.png)
> *Caption: FastAPI health endpoint showing model is loaded and ready*

### Step 4: React Frontend

We built a modern React frontend using **Vite** that provides:
- 4 input fields for Iris flower measurements (sepal/petal length/width)
- A "Predict" button that calls the backend API
- An "Example" button that fills sample values
- Beautiful result display with species name and emoji (🌸 Setosa, 🌺 Versicolor, 🌹 Virginica)
- Dark theme modern UI

The frontend uses `VITE_API_URL` environment variable to point to the correct backend:
- **Local dev**: `http://localhost:8000`
- **Staging**: `https://iris-backend-staging-vosw.onrender.com`
- **Production**: `https://iris-backend-production.onrender.com`

> **Screenshot placeholder:**
> ![Frontend UI](docs/images/step4_frontend_ui.png)
> *Caption: React frontend with dark theme — Iris Classifier interface*

> **Screenshot placeholder:**
> ![Frontend Prediction](docs/images/step4_frontend_prediction.png)
> *Caption: Successful prediction showing Setosa with emoji*

### Step 5: Docker Containerization

**Backend Dockerfile** — Simple Python 3.12-slim image:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note: The Dockerfile is built from the **project root** with `-f` flag:
```bash
docker build -t backend-image -f backend/Dockerfile .
```

**Frontend Dockerfile** — Multi-stage build (Node.js build → Nginx serve):
```dockerfile
FROM node:20-alpine AS build     # Stage 1: Build React app
FROM nginx:alpine                 # Stage 2: Serve with Nginx
```

> **Screenshot placeholder:**
> ![Docker Build](docs/images/step5_docker_build.png)
> *Caption: Successful Docker build of both images*

### Step 6: Test Pyramid

We implemented a **3-level test pyramid**:

| Level | File | Tests | What it validates |
|-------|------|-------|-------------------|
| **Unit** | `test_unit.py` | 3 tests | Data columns, data not empty, git hash retrieval |
| **Integration** | `test_integration.py` | 2 tests | API health endpoint, prediction format |
| **E2E** | `test_e2e.py` | 1 test | Complete prediction input validation |

```bash
pytest tests/ -v
# ✅ 6 tests passed
```

> **Screenshot placeholder:**
> ![Pytest Results](docs/images/step6_pytest_results.png)
> *Caption: All 6 tests passing — unit, integration, and e2e*

### Step 7: CI/CD Pipelines (3 Workflows)

We created 3 GitHub Actions workflows implementing a complete CI/CD chain:

1. **ci-pr-dev.yml** — Triggered on PR to `dev` → runs tests + Docker build
2. **deploy-staging.yml** — Triggered on push to `staging` → tests + triggers Render deploy hooks
3. **promote-to-production.yml** — Manual trigger → quality gate + model promotion + merge to main + Render deploy

Full details in the [CI/CD section below](#-cicd-pipeline--detailed-explanation).

### Step 8: Quality Gate

We created `check_quality.py` that validates the staging model meets an accuracy threshold of **≥ 0.90** before allowing promotion to production. This script:
1. Fetches the model version with `@staging` alias from the MLflow registry
2. Retrieves the accuracy metric from the associated training run
3. Passes if accuracy ≥ 0.90, fails otherwise (exits with code 1)

### Step 9: Deployment on Render

We deployed **4 services** on Render:

| Service | Type | Branch | URL |
|---------|------|--------|-----|
| Backend Staging | Web Service (Docker) | staging | `iris-backend-staging-vosw.onrender.com` |
| Backend Production | Web Service (Docker) | main | `iris-backend-production.onrender.com` |
| Frontend Staging | Static Site | staging | `iris-frontend-staging-lrrp.onrender.com` |
| Frontend Production | Static Site | main | `iris-frontend-production.onrender.com` |

> **Screenshot placeholder:**
> ![Render Dashboard](docs/images/step9_render_dashboard.png)
> *Caption: Render dashboard showing all 4 services running*

### Step 10: MLflow Alias Setup on DagsHub

On the DagsHub Model Registry, we manually assigned:
- **@staging** alias → to the trained model version (used by staging backend)
- **@production** alias → assigned automatically by the promote workflow

> **Screenshot placeholder:**
> ![DagsHub Aliases](docs/images/step10_dagshub_aliases.png)
> *Caption: Model Registry on DagsHub showing @staging and @production aliases*

---

## 🔄 CI/CD Pipeline — Detailed Explanation

Our CI/CD strategy uses **3 separate GitHub Actions workflows**, each triggered at a different stage of the Git Flow:

### Pipeline 1: CI on Pull Request to Dev (`ci-pr-dev.yml`)

```
Trigger: Pull Request opened/updated targeting 'dev' branch
```

**Purpose:** Validate code quality before merging into the development branch.

**Steps:**
1. **Checkout code** — Gets the PR branch code
2. **Setup Python 3.12** — Installs Python runtime
3. **Install dependencies** — `pip install -r requirements.txt` + `dvc[http]`
4. **Configure DVC remote** — Sets up DagsHub credentials for data access
5. **Pull data from DVC** — Downloads `iris.csv` (needed for tests)
6. **Run tests** — `pytest tests/` (all unit + integration + e2e tests)
7. **Build Docker image** — `docker build -t backend-image -f backend/Dockerfile .`

**If any step fails** → the PR is marked as failed ❌ and cannot be merged.

**Secrets used:** `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `DAGSHUB_TOKEN`

> **Screenshot placeholder:**
> ![CI PR Dev](docs/images/cicd_pipeline1_pr_dev.png)
> *Caption: GitHub Actions — CI pipeline running on a PR to dev*

> **Screenshot placeholder:**
> ![CI PR Dev Success](docs/images/cicd_pipeline1_success.png)
> *Caption: All CI checks passed — PR is ready to merge*

---

### Pipeline 2: Deploy to Staging (`deploy-staging.yml`)

```
Trigger: Push to 'staging' branch (typically via merge from dev)
```

**Purpose:** Run full tests and automatically deploy to the staging environment.

**Steps:**
1. **Checkout + Python + Dependencies** — Same setup as Pipeline 1
2. **DVC Pull** — Downloads data for tests
3. **Run Full Test Suite** — `pytest tests/`
4. **Trigger Render Deploy Hooks** — Sends POST requests to:
   - `RENDER_DEPLOY_HOOK_STAGING` → Redeploys backend staging
   - `RENDER_DEPLOY_HOOK_FRONTEND_STAGING` → Redeploys frontend staging

**Result:** Staging environment is updated with the latest code. The staging backend loads the model with `@staging` alias.

> **Screenshot placeholder:**
> ![Deploy Staging](docs/images/cicd_pipeline2_deploy_staging.png)
> *Caption: Deploy to Staging workflow executing — tests + deploy hooks*

> **Screenshot placeholder:**
> ![Staging Live](docs/images/cicd_pipeline2_staging_live.png)
> *Caption: Staging services updated and live on Render*

---

### Pipeline 3: Promote to Production (`promote-to-production.yml`)

```
Trigger: Manual (workflow_dispatch) — initiated by clicking "Run workflow" in GitHub Actions
```

**Purpose:** Validate model quality, promote model alias, merge code to main, and deploy to production.

**Steps:**
1. **Checkout** — With `fetch-depth: 0` (full history needed for merge) + write permissions
2. **Install MLflow + DagsHub** — Minimal dependencies for quality check
3. **Run Quality Gate** — Executes `check_quality.py`:
   - Fetches model with `@staging` alias
   - Checks accuracy ≥ 0.90
   - **Fails the entire workflow if accuracy is too low** ❌
4. **Promote Model** — If quality gate passes:
   - Gets the staging model version
   - Assigns `@Production` alias to that version
   - Now production backend will load this model version
5. **Merge Staging → Main** — Automated git merge:
   ```bash
   git checkout main
   git merge origin/staging --no-ff -m "Promote staging to production [skip ci]"
   git push origin main
   ```
6. **Trigger Render Production Deploy**:
   - `RENDER_DEPLOY_HOOK_PRODUCTION` → Redeploys backend production
   - `RENDER_DEPLOY_HOOK_FRONTEND_PRODUCTION` → Redeploys frontend production

> **Screenshot placeholder:**
> ![Promote Workflow](docs/images/cicd_pipeline3_promote.png)
> *Caption: Promote to Production workflow — manual trigger*

> **Screenshot placeholder:**
> ![Quality Gate Pass](docs/images/cicd_pipeline3_quality_gate.png)
> *Caption: Quality Gate passed — accuracy above threshold*

> **Screenshot placeholder:**
> ![Production Deploy](docs/images/cicd_pipeline3_production_deploy.png)
> *Caption: Production services redeployed after successful promotion*

---

### CI/CD Flow Summary

```
 feature/xxx ──PR──▶ dev ──merge──▶ staging ──manual──▶ main
      │                │                │                  │
      │          ┌─────▼─────┐   ┌─────▼─────┐    ┌──────▼──────┐
      │          │ Pipeline 1│   │ Pipeline 2│    │  Pipeline 3 │
      │          │           │   │           │    │             │
      │          │ ✅ Tests  │   │ ✅ Tests  │    │ ✅ Quality  │
      │          │ ✅ Docker │   │ ✅ Deploy │    │    Gate     │
      │          │   Build   │   │   Staging │    │ ✅ Promote  │
      │          │           │   │           │    │    Model    │
      │          └───────────┘   └───────────┘    │ ✅ Merge    │
      │                                           │ ✅ Deploy   │
      │                                           │   Production│
      │                                           └─────────────┘
```

---

## 🏆 Model Promotion — From Training to Production

The model goes through a carefully controlled lifecycle:

### Stage 1: Training & Registration

```python
# scripts/train.py
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="iris_logistic_model"
)
```

When `train.py` runs, it:
1. Trains a Logistic Regression model
2. Logs all parameters (`C`, `max_iter`), metrics (`accuracy`), and traceability info (`git_commit`, `dvc_data_version`)
3. Registers the model in the **MLflow Model Registry** as `iris_logistic_model` with an auto-incremented version number

### Stage 2: Staging Alias Assignment

On **DagsHub Model Registry**, the `@staging` alias is assigned to the model version that should be tested:

> **Screenshot placeholder:**
> ![Assign Staging](docs/images/promotion_assign_staging.png)
> *Caption: Assigning @staging alias to model version on DagsHub*

The **staging backend** on Render automatically loads this version:
```python
MODEL_STAGE = os.getenv("MODEL_STAGE", "@staging")  # Default
model = mlflow.pyfunc.load_model(f"models:/iris_logistic_model@staging")
```

### Stage 3: Quality Gate Validation

Before promotion, `check_quality.py` verifies:

```python
model_version = client.get_model_version_by_alias("iris_logistic_model", "staging")
run_data = client.get_run(model_version.run_id).data
accuracy = run_data.metrics.get("accuracy", 0)

if accuracy >= 0.90:  # THRESHOLD
    print("✅ Quality Gate passed!")
else:
    exit(1)  # BLOCKS promotion
```

### Stage 4: Production Promotion

If the quality gate passes, the **Promote to Production** workflow:

1. **Assigns `@Production` alias** to the staging model version:
```python
client.set_registered_model_alias('iris_logistic_model', 'Production', staging_version.version)
```

2. The **production backend** loads this version:
```python
MODEL_STAGE = "@production"  # Set via environment variable on Render
model = mlflow.pyfunc.load_model(f"models:/iris_logistic_model@production")
```

### Model Lifecycle Summary

```
 Train ──▶ Register ──▶ @staging ──▶ Quality Gate ──▶ @production
   │          │            │              │                │
   │     MLflow Model   DagsHub      check_quality.py   DagsHub
   │     Registry       Manual       accuracy ≥ 0.90    Automatic
   │                    Assignment                      Assignment
   │
   ▼
 MLflow Experiment
 (params, metrics,
  git_commit,
  dvc_data_version)
```

> **Screenshot placeholder:**
> ![Model Lifecycle](docs/images/promotion_lifecycle.png)
> *Caption: Complete model lifecycle from training to production*

---

## 🛡 Quality Gate

The quality gate (`check_quality.py`) is the **safety barrier** between staging and production.

**What it checks:**
- Retrieves the model version with `@staging` alias from the MLflow Model Registry
- Fetches the `accuracy` metric from the training run that produced this model
- **Threshold: accuracy ≥ 0.90** (90%)

**When it runs:**
- Automatically during the "Promote to Production" workflow
- If it fails → **the entire promotion is blocked** (model stays in staging, code is NOT merged to main, production is NOT redeployed)

**Example output (success):**
```
Vérification du modèle Candidat (v1)
Accuracy trouvée : 1.0000 (Seuil requis : 0.9)
✅ Quality Gate passée avec succès !
```

**Example output (failure):**
```
Vérification du modèle Candidat (v2)
Accuracy trouvée : 0.7500 (Seuil requis : 0.9)
❌ Échec : L'accuracy (0.75) est trop faible.
```

---

## 📦 Data Versioning with DVC

We use **DVC (Data Version Control)** to version our dataset, with **DagsHub** as the remote storage backend.

### Why DVC?
- `data/iris.csv` is **NOT stored in Git** (it's in `data/.gitignore`)
- Instead, `data/iris.csv.dvc` (a small metadata file) is committed to Git
- The actual data is stored on DagsHub's DVC remote
- This ensures reproducibility: any commit can retrieve the exact dataset used

### DVC Configuration

```bash
# Remote setup
dvc remote add origin https://dagshub.com/camm22/MLOps_final_project.dvc
dvc remote list
# origin  https://dagshub.com/camm22/MLOps_final_project.dvc
```

### DVC File (`data/iris.csv.dvc`)

```yaml
outs:
- md5: 21d441a28bce4417276097df955afc50
  size: 2928
  hash: md5
  path: iris.csv
```

This file contains the **MD5 hash** of the dataset. DVC uses this hash to fetch the correct version from the remote.

### DVC in CI/CD

Every CI/CD pipeline pulls the data before running tests:
```yaml
- name: Pull data from DVC
  run: dvc pull -r origin
```

> **Screenshot placeholder:**
> ![DVC Remote](docs/images/dvc_remote_list.png)
> *Caption: DVC remote configured with DagsHub*

> **Screenshot placeholder:**
> ![DVC Pull](docs/images/dvc_pull.png)
> *Caption: DVC pull downloading the dataset in CI*

---

## 🧪 Test Pyramid

### Unit Tests (`tests/test_unit.py`)

| Test | Purpose |
|------|---------|
| `test_data_columns` | Verifies iris.csv has all 5 expected columns |
| `test_data_not_empty` | Verifies the dataset has at least 1 row |
| `test_git_hash_format` | Verifies git commit hash retrieval works (40 chars) |

### Integration Tests (`tests/test_integration.py`)

| Test | Purpose |
|------|---------|
| `test_api_health` | Calls `GET /` and checks response contains "API Iris en ligne" |
| `test_api_prediction_format` | Calls `POST /predict` with sample data, checks status code |

### End-to-End Tests (`tests/test_e2e.py`)

| Test | Purpose |
|------|---------|
| `test_complete_prediction_flow` | Validates that prediction input is properly formatted as a list |

### Running Tests

```bash
pytest tests/ -v
```

> **Screenshot placeholder:**
> ![Test Results](docs/images/test_pyramid_results.png)
> *Caption: pytest output showing all 6 tests passing*

---

## 🌐 Deployment Architecture on Render

### Backend Services (Docker Web Services)

Both backend services use the same `backend/Dockerfile`, built from the project root. The only difference is the `MODEL_STAGE` environment variable:

| Service | Branch | MODEL_STAGE | Docker Context |
|---------|--------|-------------|---------------|
| `iris-backend-staging` | staging | `@staging` (default) | Root + `backend/Dockerfile` |
| `iris-backend-production` | main | `@production` | Root + `backend/Dockerfile` |

**Render settings for backend:**
- **Type:** Web Service
- **Environment:** Docker
- **Dockerfile Path:** `./backend/Dockerfile`
- **Docker Context:** `.` (root)
- **Environment Variables:**
  - `MLFLOW_TRACKING_USERNAME` = camm22
  - `MLFLOW_TRACKING_PASSWORD` = (DagsHub token)
  - `MODEL_STAGE` = `@staging` or `@production`

### Frontend Services (Static Sites)

Both frontend services build the React app with Vite, pointing to their respective backends:

| Service | Branch | VITE_API_URL |
|---------|--------|-------------|
| `iris-frontend-staging` | staging | `https://iris-backend-staging-vosw.onrender.com` |
| `iris-frontend-production` | main | `https://iris-backend-production.onrender.com` |

**Render settings for frontend:**
- **Type:** Static Site
- **Root Directory:** `frontend`
- **Build Command:** `npm install && npm run build`
- **Publish Directory:** `dist`
- **Environment Variables:**
  - `VITE_API_URL` = (backend URL)

> **Screenshot placeholder:**
> ![Render Backend Config](docs/images/render_backend_config.png)
> *Caption: Render backend service configuration*

> **Screenshot placeholder:**
> ![Render Frontend Config](docs/images/render_frontend_config.png)
> *Caption: Render frontend static site configuration*

---

## 🔁 Reproducibility Instructions

### Prerequisites

- Python 3.12+
- Node.js 20+
- Git
- DVC (`pip install dvc dvc-s3`)
- A DagsHub account (or access to the DagsHub repo)

### 1. Clone the Repository

```bash
git clone https://github.com/camm22/MLOps_final_project.git
cd MLOps_final_project
```

### 2. Set Up Python Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Pull Data with DVC

```bash
# Configure DVC credentials (one-time)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <YOUR_DAGSHUB_USERNAME>
dvc remote modify origin --local password <YOUR_DAGSHUB_TOKEN>

# Pull the dataset
dvc pull -r origin
```

This will download `data/iris.csv` to your local machine.

### 4. Train the Model (Optional)

```bash
# Set DagsHub credentials
export MLFLOW_TRACKING_USERNAME=<your_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>

python scripts/train.py
```

### 5. Run the Backend Locally

```bash
export MLFLOW_TRACKING_USERNAME=<your_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>

uvicorn backend.main:app --reload
# API available at http://localhost:8000
```

### 6. Run the Frontend Locally

```bash
cd frontend
npm install
npm run dev
# Frontend available at http://localhost:5173
```

### 7. Run Tests

```bash
pytest tests/ -v
```

### 8. Build Docker Images

```bash
# Backend
docker build -t iris-backend -f backend/Dockerfile .

# Frontend
docker build -t iris-frontend --build-arg VITE_API_URL=http://localhost:8000 frontend/
```

> **Screenshot placeholder:**
> ![Local Setup](docs/images/reproducibility_local.png)
> *Caption: Backend + Frontend running locally with successful prediction*

---

## 🎬 Demonstration Plan

This section provides a **step-by-step demonstration** of the complete MLOps pipeline.

---

### Demo Part 1: Show the Project Structure & Codebase

**Goal:** Walk through the project organization and explain each component.

1. Open the project in VS Code / your IDE
2. Show the **folder structure** — explain each directory's purpose
3. Open `scripts/train.py` — highlight MLflow logging, git commit tracking, DVC data version tracking
4. Open `backend/main.py` — show model loading from registry, CORS setup, predict endpoint
5. Open `frontend/src/App.jsx` — show the React UI, API call, species mapping
6. Open `check_quality.py` — explain the quality gate threshold

> **Screenshot placeholder:**
> ![Demo Codebase](docs/images/demo_part1_codebase.png)
> *Caption: VS Code showing the project structure*

---

### Demo Part 2: Show MLflow Experiment Tracking on DagsHub

**Goal:** Demonstrate that every training run is fully tracked.

1. Open DagsHub: `https://dagshub.com/camm22/MLOps_final_project`
2. Navigate to the **Experiments** tab
3. Show a training run — click on it to expand:
   - **Parameters**: `C=1.0`, `max_iter=200`, `git_commit=<hash>`, `dvc_data_version=<url>`
   - **Metrics**: `accuracy=1.0`
   - **Artifacts**: the serialized model
4. Explain: *"Every run records which code version (git commit) and which data version (DVC URL) was used, ensuring full traceability."*

> **Screenshot placeholder:**
> ![DagsHub Experiments](docs/images/demo_part2_experiments_list.png)
> *Caption: DagsHub Experiments tab showing training runs*

> **Screenshot placeholder:**
> ![DagsHub Run Detail](docs/images/demo_part2_run_detail.png)
> *Caption: Detailed view of a training run — params, metrics, artifacts*

> **Screenshot placeholder:**
> ![DagsHub Params](docs/images/demo_part2_params.png)
> *Caption: Parameters showing git_commit and dvc_data_version for traceability*

---

### Demo Part 3: Show the Model Registry & Aliases on DagsHub

**Goal:** Demonstrate the model registry with staging/production aliases.

1. On DagsHub, navigate to **Models** or **Model Registry**
2. Click on `iris_logistic_model`
3. Show the **registered versions** (v1, v2, etc.)
4. Show the **aliases**: `@staging` and `@production` assigned to specific versions
5. Explain: *"The @staging alias points to the model version that the staging backend loads. The @production alias is assigned automatically by the promotion workflow after the quality gate passes."*

> **Screenshot placeholder:**
> ![Model Registry](docs/images/demo_part3_model_registry.png)
> *Caption: MLflow Model Registry showing iris_logistic_model*

> **Screenshot placeholder:**
> ![Model Aliases](docs/images/demo_part3_aliases.png)
> *Caption: Model versions with @staging and @production aliases*

---

### Demo Part 4: Show DVC Data Versioning

**Goal:** Prove that data is versioned and reproducible.

1. Open a terminal and run:
   ```bash
   dvc remote list
   # Shows: origin  https://dagshub.com/camm22/MLOps_final_project.dvc
   ```
2. Show `data/iris.csv.dvc` — explain the MD5 hash and how DVC tracks versions
3. Show `data/.gitignore` — iris.csv is NOT in Git
4. Run `dvc pull -r origin` — show that it downloads the exact dataset version
5. Explain: *"The actual data is stored on DagsHub's DVC remote. Git only tracks a small metadata file (.dvc) with the hash. This ensures any commit can retrieve the exact dataset used."*

> **Screenshot placeholder:**
> ![DVC Remote List](docs/images/demo_part4_dvc_remote.png)
> *Caption: DVC remote pointing to DagsHub*

> **Screenshot placeholder:**
> ![DVC File](docs/images/demo_part4_dvc_file.png)
> *Caption: Content of iris.csv.dvc showing the MD5 hash*

> **Screenshot placeholder:**
> ![DVC Pull](docs/images/demo_part4_dvc_pull.png)
> *Caption: DVC pull downloading the dataset*

---

### Demo Part 5: Run Tests Locally

**Goal:** Show the test pyramid in action.

1. Run the full test suite:
   ```bash
   pytest tests/ -v
   ```
2. Show output: 6 tests passing (3 unit + 2 integration + 1 e2e)
3. Explain each test level:
   - **Unit**: Data validation, git hash format
   - **Integration**: API health check, prediction endpoint
   - **End-to-end**: Full prediction flow validation

> **Screenshot placeholder:**
> ![Pytest Local](docs/images/demo_part5_pytest.png)
> *Caption: pytest -v output showing all 6 tests passing*

---

### Demo Part 6: CI/CD Pipeline 1 — PR to Dev

**Goal:** Demonstrate the automated CI on Pull Request.

1. Create a feature branch from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/demo-readme
   ```

2. Make a change (e.g., update this README) and commit:
   ```bash
   git add .
   git commit -m "feat: add comprehensive README"
   git push origin feature/demo-readme
   ```

3. On GitHub, **open a Pull Request** from `feature/demo-readme` → `dev`

4. **Show the CI pipeline running** in the "Checks" tab:
   - ✅ Checkout code
   - ✅ Setup Python
   - ✅ Install dependencies
   - ✅ DVC pull (downloads data)
   - ✅ Run tests (6 passed)
   - ✅ Docker build

5. Once all checks pass, **merge the PR**

> **Screenshot placeholder:**
> ![PR Created](docs/images/demo_part6_pr_created.png)
> *Caption: Pull Request from feature/demo-readme to dev*

> **Screenshot placeholder:**
> ![CI Running](docs/images/demo_part6_ci_running.png)
> *Caption: CI pipeline running — tests and Docker build in progress*

> **Screenshot placeholder:**
> ![CI Passed](docs/images/demo_part6_ci_passed.png)
> *Caption: All CI checks passed — ready to merge*

> **Screenshot placeholder:**
> ![PR Merged](docs/images/demo_part6_pr_merged.png)
> *Caption: PR merged into dev*

---

### Demo Part 7: CI/CD Pipeline 2 — Deploy to Staging

**Goal:** Show automatic deployment to staging environment.

1. Merge `dev` into `staging`:
   ```bash
   git checkout staging
   git pull origin staging
   git merge dev
   git push origin staging
   ```

2. **Show the Deploy Staging workflow** running in GitHub Actions:
   - ✅ Tests pass
   - ✅ Render deploy hooks triggered (backend + frontend)

3. **Wait for Render to redeploy** (check Render dashboard — both staging services show "Live")

4. **Open the staging frontend** in the browser: `https://iris-frontend-staging-lrrp.onrender.com`

5. **Make a prediction**:
   - Enter values: Sepal Length=5.1, Sepal Width=3.5, Petal Length=1.4, Petal Width=0.2
   - Click "Predict"
   - Show result: 🌸 Setosa (class 0)

6. Explain: *"The staging backend loads the model with @staging alias. This is where we validate everything works before promoting to production."*

> **Screenshot placeholder:**
> ![Deploy Staging Running](docs/images/demo_part7_deploy_running.png)
> *Caption: Deploy to Staging workflow running in GitHub Actions*

> **Screenshot placeholder:**
> ![Deploy Staging Success](docs/images/demo_part7_deploy_success.png)
> *Caption: Staging deployment successful — hooks triggered*

> **Screenshot placeholder:**
> ![Render Staging Live](docs/images/demo_part7_render_staging.png)
> *Caption: Render dashboard showing staging services are Live*

> **Screenshot placeholder:**
> ![Staging Prediction](docs/images/demo_part7_staging_prediction.png)
> *Caption: Frontend staging — successful prediction on Setosa*

---

### Demo Part 8: CI/CD Pipeline 3 — Promote to Production

**Goal:** Demonstrate the manual promotion workflow with quality gate.

1. Go to **GitHub → Actions → "Promote to Production"**

2. Click **"Run workflow"** (dropdown → select branch → Run)

3. **Watch the workflow execute**:
   - ✅ Checkout code (full history)
   - ✅ Install MLflow + DagsHub
   - ✅ **Quality Gate** — `check_quality.py` runs:
     ```
     Accuracy trouvée : 1.0000 (Seuil requis : 0.9)
     ✅ Quality Gate passée avec succès !
     ```
   - ✅ **Model Promotion** — `@Production` alias assigned
   - ✅ **Git Merge** — staging merged into main automatically
   - ✅ **Render Deploy** — production hooks triggered

4. **Check DagsHub** — Model Registry now shows `@Production` alias on the promoted version

5. **Open the production frontend**: `https://iris-frontend-production.onrender.com`

6. **Make the same prediction**:
   - Enter: 5.1, 3.5, 1.4, 0.2
   - Result: 🌸 Setosa — same result as staging ✅

7. Explain: *"The quality gate verified the model accuracy is above 90%. The model @staging alias was promoted to @production. Code from staging was automatically merged to main. Both production services were redeployed."*

> **Screenshot placeholder:**
> ![Promote Trigger](docs/images/demo_part8_promote_trigger.png)
> *Caption: Manually triggering the Promote to Production workflow*

> **Screenshot placeholder:**
> ![Quality Gate Log](docs/images/demo_part8_quality_gate.png)
> *Caption: Quality Gate step — accuracy 1.0 ≥ 0.90 threshold*

> **Screenshot placeholder:**
> ![Model Promoted](docs/images/demo_part8_model_promoted.png)
> *Caption: DagsHub showing @Production alias assigned after promotion*

> **Screenshot placeholder:**
> ![Merge to Main](docs/images/demo_part8_merge_main.png)
> *Caption: Automatic merge from staging to main*

> **Screenshot placeholder:**
> ![Production Prediction](docs/images/demo_part8_production_prediction.png)
> *Caption: Production frontend — successful prediction confirming end-to-end pipeline works*

---

### Demo Part 9: Show All 4 Services Running on Render

**Goal:** Visual proof that the full infrastructure is live.

1. Open the **Render dashboard** — show all 4 services with "Live" status
2. Open each URL in the browser:
   - Backend Staging: `https://iris-backend-staging-vosw.onrender.com/`
   - Backend Production: `https://iris-backend-production.onrender.com/`
   - Frontend Staging: `https://iris-frontend-staging-lrrp.onrender.com/`
   - Frontend Production: `https://iris-frontend-production.onrender.com/`

> **Screenshot placeholder:**
> ![All Services](docs/images/demo_part9_all_services.png)
> *Caption: Render dashboard — 4 services all showing "Live" status*

> **Screenshot placeholder:**
> ![Backend Health Staging](docs/images/demo_part9_backend_staging.png)
> *Caption: Backend staging health check — model loaded with @staging*

> **Screenshot placeholder:**
> ![Backend Health Prod](docs/images/demo_part9_backend_production.png)
> *Caption: Backend production health check — model loaded with @production*

---

### Demo Part 10: Recap — Everything Connected

**Final summary slide / explanation:**

```
1. DATA        → iris.csv versioned with DVC on DagsHub
2. TRAINING    → train.py logs to MLflow (params, metrics, git commit, DVC version)
3. REGISTRY    → Model registered as iris_logistic_model with aliases
4. API         → FastAPI loads model from registry based on alias
5. FRONTEND    → React app calls the API for predictions
6. TESTS       → 6 tests (unit + integration + e2e) run in every pipeline
7. CI/CD       → 3 GitHub Actions workflows automate the entire flow
8. QUALITY     → check_quality.py blocks bad models from reaching production
9. DEPLOYMENT  → 4 Render services (2 backends + 2 frontends)
10. GIT FLOW   → feature → dev → staging → main (controlled progression)
```

> **Screenshot placeholder:**
> ![Final Summary](docs/images/demo_part10_summary.png)
> *Caption: Complete MLOps pipeline — all components connected*

---

## 🔑 GitHub Secrets Reference

These secrets must be configured in **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Purpose | Example |
|--------|---------|---------|
| `MLFLOW_TRACKING_USERNAME` | DagsHub username | `camm22` |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub password/token | `***` |
| `DAGSHUB_TOKEN` | DagsHub API token (for DVC) | `***` |
| `RENDER_DEPLOY_HOOK_STAGING` | Render deploy hook URL for backend staging | `https://api.render.com/deploy/...` |
| `RENDER_DEPLOY_HOOK_PRODUCTION` | Render deploy hook URL for backend production | `https://api.render.com/deploy/...` |
| `RENDER_DEPLOY_HOOK_FRONTEND_STAGING` | Render deploy hook URL for frontend staging | `https://api.render.com/deploy/...` |
| `RENDER_DEPLOY_HOOK_FRONTEND_PRODUCTION` | Render deploy hook URL for frontend production | `https://api.render.com/deploy/...` |

> **Screenshot placeholder:**
> ![GitHub Secrets](docs/images/github_secrets.png)
> *Caption: GitHub repository secrets configured for CI/CD*

---

## 🌍 Live URLs

| Environment | Service | URL |
|-------------|---------|-----|
| **Staging** | Backend API | https://iris-backend-staging-vosw.onrender.com |
| **Staging** | Frontend | https://iris-frontend-staging-lrrp.onrender.com |
| **Production** | Backend API | https://iris-backend-production.onrender.com |
| **Production** | Frontend | https://iris-frontend-production.onrender.com |
| **MLflow** | Tracking Server | https://dagshub.com/camm22/MLOps_final_project.mlflow |
| **DagsHub** | Repository | https://dagshub.com/camm22/MLOps_final_project |
| **GitHub** | Repository | https://github.com/camm22/MLOps_final_project |

---

<p align="center">
  <strong>Built with ❤️ as part of the Machine Learning in Production course — 2026</strong>
</p>
