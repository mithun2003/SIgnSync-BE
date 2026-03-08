<h1 align="center">🤟 SignSync Backend</h1>
<p align="center" markdown=1>
  <i><b>Real-time Sign Language Recognition API</b> — FastAPI backend with CNN + MediaPipe ML inference, JWT auth, WebSocket streaming, and full admin management.</i>
</p>

<p align="center">
  <a href="https://fastapi.tiangolo.com">
      <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  </a>
  <a href="https://www.postgresql.org">
      <img src="https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL">
  </a>
  <a href="https://redis.io">
      <img src="https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=fff&style=for-the-badge" alt="Redis">
  </a>
  <a href="https://www.tensorflow.org">
      <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  </a>
  <a href="https://www.docker.com">
      <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

______________________________________________________________________

## 📖 About SignSync

**SignSync** is a production-ready backend for **sign language recognition** that enables users to detect and learn American Sign Language (ASL) hand gestures in real time. It uses a **CNN model (MobileNet)** combined with **MediaPipe** for skeletal landmark detection to classify hand signs A–Z plus special gestures (`space`, `del`, `nothing`).

### What it does

- 🤖 **ML Prediction** — Accepts hand-skeleton images and returns the detected ASL letter with confidence score
- 📡 **WebSocket Streaming** — Real-time frame-by-frame sign detection via WebSocket
- 🧑‍💻 **User Management** — Full auth (JWT), profiles, profile images (local/Cloudinary)
- 📊 **Detection Logging** — Persists every detected sign for learning analytics and progress tracking
- 📈 **Dashboard Analytics** — User activity trends, accuracy stats, streaks, mastered letters
- 🖼️ **Sign Image Library** — Admin-managed reference images per ASL character, served via Cloudinary CDN
- 🔑 **Role-based Access** — Tier-based rate limiting (free/pro), admin-only management routes
- ⚙️ **Background Jobs** — Async task processing via ARQ + Redis queue

______________________________________________________________________

## ✨ Features

| Feature                 | Details                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| ⚡ **Async FastAPI**    | Fully non-blocking with SQLAlchemy 2.0 async sessions               |
| 🤖 **ML Inference**     | TensorFlow CNN + MediaPipe skeletal detection                       |
| 📡 **WebSocket**        | Real-time prediction streaming endpoint                             |
| 🔐 **JWT Auth**         | Access + refresh tokens, HttpOnly cookie, token blacklist on logout |
| 👮 **Rate Limiting**    | Per-tier, per-path configurable limits                              |
| 📦 **FastCRUD**         | Efficient CRUD with pagination for all resources                    |
| 🧑‍💼 **Admin Panel**      | CRUDAdmin interface for user, sign, and tier management             |
| 🚦 **Background Jobs**  | ARQ worker with Redis for async task processing                     |
| 🧊 **Redis Caching**    | Server-side cache + client-side cache-control headers               |
| 🖼️ **Cloudinary CDN**   | Sign image hosting with version management                          |
| 🐳 **Docker**           | One-command Docker Compose for local and production                 |
| 🚀 **NGINX + Gunicorn** | Production-grade reverse proxy and worker management                |
| 🧪 **Test Suite**       | pytest + pytest-asyncio with fixtures, mocks, factories             |

______________________________________________________________________

## 🛠️ Tech Stack

| Layer                | Technology                                                       |
| -------------------- | ---------------------------------------------------------------- |
| **Backend**          | FastAPI 0.109.1, Uvicorn, Python 3.11+                           |
| **Database**         | PostgreSQL, SQLAlchemy 2.0 (async), Alembic                      |
| **ML**               | TensorFlow (CPU) 2.18.0, MediaPipe 0.10.31, OpenCV, scikit-learn |
| **Caching / Queues** | Redis 5.0.1, ARQ 0.25.0                                          |
| **Auth**             | JWT (python-jose), bcrypt password hashing                       |
| **Storage**          | Cloudinary (sign images), local filesystem (profile images)      |
| **Admin**            | CRUDAdmin 0.4.2, FastCRUD 0.19.2                                 |
| **Testing**          | pytest, pytest-asyncio, faker                                    |
| **DevOps**           | Docker, docker-compose, Gunicorn, NGINX                          |

______________________________________________________________________

## 📁 Project Structure

```
backend/
├── src/
│   ├── app/
│   │   ├── main.py                         # Application entry point
│   │   ├── admin/                          # CRUDAdmin panel setup
│   │   │   ├── initialize.py               # Admin app factory
│   │   │   └── views.py                    # Admin model views
│   │   ├── api/                            # API layer
│   │   │   ├── dependencies.py             # Auth, rate-limit, user dependencies
│   │   │   └── v1/                         # Version 1 routes
│   │   │       ├── auth.py                 # POST /register
│   │   │       ├── login.py                # POST /login, /refresh
│   │   │       ├── logout.py               # POST /logout
│   │   │       ├── health.py               # GET /health, /ready
│   │   │       ├── users.py                # User profile CRUD + image upload
│   │   │       ├── signs.py                # GET sign image by character
│   │   │       ├── predict.py              # POST /predict, WS /predict/ws
│   │   │       ├── sign_detections.py      # Detection log CRUD
│   │   │       ├── tasks.py                # Background job endpoints
│   │   │       ├── tiers.py                # Tier management
│   │   │       ├── rate_limits.py          # Rate limit config
│   │   │       ├── dashboard.py            # User analytics dashboard
│   │   │       └── admin/                  # Admin-only routes
│   │   │           ├── dashboard.py        # Admin analytics + CSV export
│   │   │           ├── signs.py            # Admin sign image management
│   │   │           └── users.py            # Admin user management
│   │   ├── core/                           # Core infrastructure
│   │   │   ├── config.py                   # Settings loaded from .env
│   │   │   ├── schemas.py                  # Shared response models
│   │   │   ├── security.py                 # JWT creation, token blacklist
│   │   │   ├── logger.py                   # Logging configuration
│   │   │   ├── health.py                   # Health check utilities
│   │   │   ├── setup.py                    # App factory + lifespan handler
│   │   │   ├── db/                         # Database layer
│   │   │   │   ├── database.py             # Async engine, session, Base
│   │   │   │   ├── token_blacklist.py      # Token blacklist model
│   │   │   │   └── crud_token_blacklist.py # CRUD for blacklisted tokens
│   │   │   ├── ml/                         # Machine learning
│   │   │   │   ├── predict.py              # Predictor loader + dispatcher
│   │   │   │   ├── predict_cnn.py          # CNN-only prediction
│   │   │   │   ├── predict_cnn_mediapipe.py# MediaPipe skeleton + CNN
│   │   │   │   ├── predict_svm.py          # SVM predictor (alternative)
│   │   │   │   ├── schema.py               # ML request/response schemas
│   │   │   │   ├── train.py                # Model training script
│   │   │   │   └── train_local.py          # Local training helper
│   │   │   ├── utils/
│   │   │   │   ├── cache.py                # Redis async cache helpers
│   │   │   │   ├── cloudinary.py           # Cloudinary upload/list utils
│   │   │   │   ├── queue.py                # ARQ background job pool
│   │   │   │   └── rate_limit.py           # Rate limiting logic
│   │   │   ├── exceptions/
│   │   │   │   ├── http_exceptions.py      # Custom HTTP error classes
│   │   │   │   └── cache_exceptions.py     # Cache error classes
│   │   │   └── worker/
│   │   │       └── functions.py            # Background job handler functions
│   │   ├── crud/                           # FastCRUD wrappers
│   │   │   ├── crud_users.py
│   │   │   ├── crud_signs.py
│   │   │   ├── crud_sign_detections.py
│   │   │   ├── crud_tier.py
│   │   │   ├── crud_rate_limit.py
│   │   │   └── crud_admin.py
│   │   ├── models/                         # SQLAlchemy ORM models
│   │   │   ├── user.py                     # User table
│   │   │   ├── signs.py                    # Sign images table
│   │   │   ├── sign_detection.py           # Detection log table
│   │   │   ├── tier.py                     # Subscription tier table
│   │   │   └── rate_limit.py               # Rate limit rules table
│   │   ├── schemas/                        # Pydantic request/response models
│   │   │   ├── user.py
│   │   │   ├── signs.py
│   │   │   ├── sign_detection.py
│   │   │   ├── tier.py
│   │   │   ├── rate_limit.py
│   │   │   ├── admin.py
│   │   │   └── job.py
│   │   ├── services/
│   │   │   └── dashboard_service.py        # Dashboard analytics business logic
│   │   └── middleware/
│   │       └── client_cache_middleware.py  # Client-side cache header injection
│   ├── migrations/                         # Alembic DB migrations
│   │   ├── versions/                       # Migration version files
│   │   └── env.py
│   ├── scripts/                            # One-time setup scripts
│   │   ├── create_first_superuser.py       # Bootstrap admin account
│   │   └── create_first_tier.py            # Bootstrap default tier
│   └── alembic.ini
├── tests/                                  # Test suite
│   ├── conftest.py                         # Fixtures and test DB setup
│   ├── test_auth.py                        # Auth flow tests
│   ├── test_user.py                        # User profile tests
│   ├── test_sign_detections.py             # Detection CRUD tests
│   ├── test_rate_limiter.py                # Rate limit tests
│   └── helpers/
│       ├── generators.py                   # Test data factories
│       └── mocks.py                        # Mock objects
├── Dockerfile                              # Docker image definition
├── docker-compose.yml                      # Multi-container orchestration
├── default.conf                            # NGINX configuration
├── pyproject.toml                          # Project metadata & dependencies
├── setup.py                                # Interactive setup script
└── README.md                               # This file
```

______________________________________________________________________

## 🚀 Quickstart

### 1. Clone & configure

```bash
git clone https://github.com/<your-org>/SignSync
cd SignSync/backend
cp src/.env.example src/.env   # then fill in your values
```

### 2. Start with Docker Compose

```bash
docker compose up
```

### 3. Access the app

| URL                           | Description            |
| ----------------------------- | ---------------------- |
| `http://127.0.0.1:8000`       | API root               |
| `http://127.0.0.1:8000/docs`  | Interactive Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc API reference    |
| `http://127.0.0.1:8000/admin` | CRUDAdmin panel        |

### 4. Bootstrap first admin & tier

```bash
docker compose run --rm create_superuser
docker compose run --rm create_first_tier
```

### Run locally without Docker

```bash
uv sync
uv run uvicorn src.app.main:app --reload
```

______________________________________________________________________

## ⚙️ Configuration

Create `src/.env` with at minimum:

```env
# App
APP_NAME="SignSync"
ENVIRONMENT=local        # local | staging | production

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=signsync
POSTGRES_HOST=db
POSTGRES_PORT=5432

# Redis
REDIS_URL=redis://redis:6379/0

# JWT
SECRET_KEY=your-very-secret-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# First superuser (created by bootstrap script)
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=changeme

# Cloudinary (for sign images)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

- `ENVIRONMENT=local` exposes `/docs` and `/redoc`
- `ENVIRONMENT=production` hides API docs

______________________________________________________________________

## 🧪 Running Tests

```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_auth.py -v
uv run pytest tests/test_sign_detections.py -v
```

______________________________________________________________________

## 🗄️ Database Migrations

```bash
# Generate a new migration after model changes
cd src && uv run alembic revision --autogenerate -m "describe your change"

# Apply migrations
cd src && uv run alembic upgrade head
```

______________________________________________________________________

## 📋 API Endpoints Overview

See **[API_ENDPOINTS.md](API_ENDPOINTS.md)** for full documentation of all endpoints including:

- HTTP method and path
- Description and purpose
- Sample request body / query parameters
- Sample response with all fields explained

### Quick reference

| Group           | Base Path                        | Endpoints                                    |
| --------------- | -------------------------------- | -------------------------------------------- |
| Auth            | `/api/v1/auth`                   | register, login, refresh, logout             |
| Health          | `/api/v1`                        | health, ready                                |
| User Profile    | `/api/v1/user`                   | profile CRUD, image upload                   |
| Signs           | `/api/v1/signs`                  | get sign reference image                     |
| Prediction      | `/api/v1/predict`                | HTTP predict, WebSocket stream, info, health |
| Detections      | `/api/v1/detection`              | log, batch log, list, get, update, delete    |
| Dashboard       | `/api/v1/dashboard`              | user analytics                               |
| Tasks           | `/api/v1/tasks`                  | enqueue job, get status                      |
| Tiers           | `/api/v1/tier`                   | CRUD (admin)                                 |
| Rate Limits     | `/api/v1/tier/{name}/rate_limit` | CRUD (admin)                                 |
| Admin Users     | `/api/v1/admin`                  | full user management, tier assignment        |
| Admin Signs     | `/api/v1/admin/signs`            | upload, bulk-upload, set active image        |
| Admin Dashboard | `/api/v1/admin/dashboard`        | system analytics, CSV export                 |

______________________________________________________________________

## 🏗️ Deployment

### Staging (Gunicorn + Uvicorn workers)

```bash
./setup.py staging
docker compose up
```

### Production (NGINX + Gunicorn + Uvicorn)

```bash
./setup.py production
# Edit src/.env — change SECRET_KEY, DB password, Cloudinary keys
docker compose up -d
```

> ⚠️ Always change `SECRET_KEY` and all credentials before any non-local deployment.

______________________________________________________________________

## 🤝 Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

[MIT](LICENSE.md)
