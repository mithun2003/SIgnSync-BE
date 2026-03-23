# SignSync Backend Report

## Methodology, Architecture, and System Implementation (FastAPI + ML)

## 1. Backend Scope

The backend provides:

- Authentication and authorization.
- User profile and emergency contact management.
- Real-time and single-shot sign prediction APIs.
- Sign detection persistence and dashboard analytics.
- Admin operations (users, signs, system health, settings, backups).
- Supporting infrastructure for rate limiting, caching, and background tasks.

Base API namespace: `/api/v1`.

______________________________________________________________________

## 2. Backend Development Methodology

### 2.1 Iterative API-First Delivery

Backend development followed an API-first incremental method:

1. Define endpoint groups and contracts.
2. Implement ORM models and CRUD operations.
3. Add dependencies for auth, authorization, and rate control.
4. Integrate ML prediction and streaming.
5. Add admin observability and operational endpoints.

### 2.2 Core Engineering Practices

- **Versioned routing** (`/api/v1`) for long-term compatibility.
- **Asynchronous execution model** for scalable I/O behavior.
- **Dependency-injected security** for consistent auth checks.
- **Layered organization**: routers, services, CRUD, models, schemas, core utilities.
- **Containerized runtime** for reproducible deployment.

______________________________________________________________________

## 3. Backend Architecture

### 3.1 Architectural Layers

1. **API layer** (`api/v1`, `api/v1/admin`)
   Handles HTTP/WebSocket interfaces, validation, and response composition.

2. **Dependency/security layer**  
   Resolves current user, superuser checks, and optional user context.

3. **Business/service layer**  
   Dashboard aggregation and email-alert orchestration.

4. **Persistence layer**  
   SQLAlchemy models + FastCRUD wrappers for data access.

5. **ML inference layer**  
   SVM model loading, feature normalization, and prediction functions.

6. **Infrastructure layer**  
   Redis cache/queue/rate-limiter clients, health checks, startup lifecycle.

### 3.2 API Domain Structure

Primary route groups include:

- `auth`, `login`, `logout`
- `user`
- `predict`
- `signs`
- `detection`
- `dashboard`
- `tasks`
- `tiers`, `rate_limits`
- `alerts`
- `admin` subdomains (`users`, `signs`, `dashboard`, `system`, `analytics`, `settings`)

### 3.3 Data Architecture

Core relational entities:

- `User`: account and profile information, role flags, emergency contacts.
- `SignDetection`: per-detection logs with confidence and session metadata.
- `Signs`: active sign image metadata and version tracking per character.
- `Tier`: subscription/permission bucket.
- `RateLimit`: per-tier, per-path throttling policy.

### 3.4 Runtime and Infrastructure Architecture

- **Database**: PostgreSQL with async SQLAlchemy engine/sessions.
- **Redis**:
  - application cache support,
  - ARQ queue backend,
  - per-path rate-limiter keying.
- **Web/worker split**:
  - FastAPI service,
  - optional ARQ worker for background jobs.
- **Deployment profiles**:
  - local Uvicorn,
  - production Gunicorn/Uvicorn behind NGINX.

______________________________________________________________________

## 4. Backend System Implementation

### 4.1 Application Startup and Lifespan

Startup responsibilities include:

- threadpool tuning,
- optional Redis pool initialization,
- database table initialization (if enabled),
- ML model preload (SVM),
- admin interface initialization.

Lifespan orchestration ensures resources are also closed gracefully on shutdown.

### 4.2 Authentication and Authorization

- Password hashing with bcrypt.
- JWT access + refresh token strategy.
- Refresh token delivered via secure HttpOnly cookie.
- Token blacklist support for logout invalidation.
- Role checks:
  - authenticated user for user APIs,
  - superuser dependency for admin APIs.

### 4.3 Prediction and Real-Time Inference

Two prediction paths are implemented:

1. **HTTP image prediction** (`POST /predict`)
   Accepts image uploads, extracts landmarks server-side, returns sign label/confidence.

2. **WebSocket landmark prediction** (`/predict/ws`)  
   Accepts pre-extracted landmark JSON and returns low-latency predictions frame-by-frame.

For successful predictions:

- results can be logged to `sign_detection`,
- HELP sign can trigger emergency mail workflow.

### 4.4 Detection Logging and Analytics

- Detection CRUD endpoints support single and batch logging.
- User dashboard endpoint aggregates:
  - total/today counts,
  - confidence metrics,
  - streaks,
  - frequent signs,
  - daily activity.

### 4.5 Sign Library Management

Public sign retrieval:

- `/signs` exposes active sign mapping for frontend translation.

Admin sign lifecycle:

- upload single image,
- bulk upload with filename-to-character mapping,
- list Cloudinary versions,
- set active version,
- delete image versions with fallback handling.

### 4.6 User Profile and Emergency Contacts

Implemented capabilities include:

- fetch/update own profile,
- upload/delete profile image,
- account deletion (soft-delete logic),
- password change,
- emergency contacts CRUD + batch operations.

### 4.7 Admin Operations and Observability

Admin APIs provide:

- system health (DB/Redis/ML/WebSocket status),
- active users windowing,
- cache clear operation,
- backup metadata and snapshot export,
- user analytics and growth trends,
- mutable settings persisted in JSON store.

### 4.8 Background Tasks

- ARQ-backed task endpoint can enqueue and inspect async jobs.
- Worker process executes registered functions and logs lifecycle events.

______________________________________________________________________

## 5. Non-Functional Implementation

### 5.1 Performance

- Async request handling and pooled DB access.
- Lightweight landmark transport in WebSocket path.
- Cached and reusable ML model state.

### 5.2 Security

- Credential hashing and tokenized auth.
- Route-level privilege separation.
- Configurable rate limiting by path and tier.
- Environment-aware documentation exposure.

### 5.3 Reliability

- Health and readiness endpoints for orchestration.
- Structured exception handling and logging.
- Containerized service composition with service health checks.

______________________________________________________________________

## 6. Backend Summary

The SignSync backend is implemented as a modular asynchronous FastAPI platform integrating ML inference, secure authentication, operational administration, and analytics. Its architecture supports real-time interaction requirements while preserving maintainability and deployment flexibility for both development and production environments.
