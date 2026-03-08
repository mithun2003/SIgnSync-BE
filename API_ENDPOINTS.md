# 📡 SignSync API Endpoints

Complete reference for all REST and WebSocket endpoints in the SignSync backend.

**Base URL:** `http://localhost:8000`
**API Version prefix:** `/api/v1`
**Auth:** All protected routes require `Authorization: Bearer <access_token>` header unless noted otherwise.

______________________________________________________________________

## Table of Contents

1. [Authentication](#1-authentication)
1. [Health Checks](#2-health-checks)
1. [User Profile](#3-user-profile)
1. [Signs (Reference Images)](#4-signs-reference-images)
1. [Prediction](#5-prediction)
1. [Sign Detections](#6-sign-detections)
1. [Dashboard](#7-dashboard)
1. [Background Tasks](#8-background-tasks)
1. [Tiers](#9-tiers)
1. [Rate Limits](#10-rate-limits)
1. [Admin — Users](#11-admin--users)
1. [Admin — Signs](#12-admin--signs)
1. [Admin — Dashboard](#13-admin--dashboard)

______________________________________________________________________

## 1. Authentication

### `POST /api/v1/auth/register`

**Description:** Register a new user account. The username and email must be unique. On success returns the created user profile.

**Auth required:** No

**Request body** (`application/json`):

```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe"
}
```

| Field        | Type   | Required | Description         |
| ------------ | ------ | -------- | ------------------- |
| `username`   | string | ✅       | Unique username     |
| `email`      | string | ✅       | Valid email address |
| `password`   | string | ✅       | Min 8 characters    |
| `first_name` | string | ❌       | Optional first name |
| `last_name`  | string | ❌       | Optional last name  |

**Response** `201 Created`:

```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "bio": null,
  "country": null,
  "language": "en",
  "profile_image_url": null,
  "is_superuser": false,
  "is_active": true,
  "created_at": "2024-01-15T10:30:00"
}
```

**Error responses:**

| Code  | Reason                                             |
| ----- | -------------------------------------------------- |
| `400` | Email already registered or username not available |
| `422` | Validation error (missing/invalid fields)          |

______________________________________________________________________

### `POST /api/v1/auth/login`

**Description:** Authenticate a user with their username/email and password. Returns a JWT access token and refresh token. The refresh token is also set as an HttpOnly cookie for secure refresh flows.

**Auth required:** No

**Request body** (`application/x-www-form-urlencoded`):

```
username=john_doe&password=SecurePass123!
```

> The `username` field accepts either the username **or** the email address.

**Response** `200 OK`:

```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "user_role": "user"
  }
}
```

Response headers include:

```
Set-Cookie: refresh_token=<token>; HttpOnly; Secure; SameSite=Lax; Path=/
```

**Error responses:**

| Code  | Reason                             |
| ----- | ---------------------------------- |
| `401` | Wrong username, email, or password |
| `422` | Missing form fields                |

______________________________________________________________________

### `POST /api/v1/auth/refresh`

**Description:** Issue a new short-lived access token using the refresh token. The refresh token is read from the HttpOnly cookie set during login. Use this when the access token expires to avoid asking the user to log in again.

**Auth required:** Cookie `refresh_token` (set by `/login`)

**Request:** No body needed — the refresh token is read from the cookie automatically.

**Response** `200 OK`:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Error responses:**

| Code  | Reason                                         |
| ----- | ---------------------------------------------- |
| `401` | Refresh token missing, expired, or blacklisted |

______________________________________________________________________

### `POST /api/v1/auth/logout`

**Description:** Invalidate the current session by blacklisting both the access token and the refresh token. After logout, both tokens are rejected on all subsequent requests.

**Auth required:** Yes (Bearer token) + Cookie `refresh_token`

**Request:** No body needed.

**Response** `200 OK`:

```json
{
  "message": "Logged out successfully"
}
```

**Error responses:**

| Code  | Reason                             |
| ----- | ---------------------------------- |
| `401` | Refresh token not found or invalid |

______________________________________________________________________

## 2. Health Checks

### `GET /api/v1/health`

**Description:** Simple liveness check. Returns app name, version, and current environment. Used by load balancers and uptime monitors to confirm the process is running.

**Auth required:** No

**Response** `200 OK`:

```json
{
  "app": "SignSync",
  "environment": "local",
  "version": "0.1.0"
}
```

______________________________________________________________________

### `GET /api/v1/ready`

**Description:** Readiness check. Verifies that the application can actually serve traffic by testing connectivity to PostgreSQL and Redis. Returns individual service statuses. Use this in Kubernetes readiness probes or deployment health gates.

**Auth required:** No

**Response** `200 OK` (all services healthy):

```json
{
  "status": "ready",
  "services": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

**Response** `503 Service Unavailable` (a service is down):

```json
{
  "status": "not ready",
  "services": {
    "database": "healthy",
    "redis": "unhealthy"
  }
}
```

______________________________________________________________________

## 3. User Profile

### `GET /api/v1/user/me`

**Description:** Retrieve the currently authenticated user's full profile including personal details, preferences, and profile image URL.

**Auth required:** Yes

**Response** `200 OK`:

```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "bio": "Learning ASL one sign at a time.",
  "country": "US",
  "language": "en",
  "profile_image_url": "/media/profile_images/abc123.jpg",
  "is_superuser": false,
  "is_active": true,
  "created_at": "2024-01-15T10:30:00"
}
```

______________________________________________________________________

### `PATCH /api/v1/user/me`

**Description:** Update one or more fields on the current user's profile. Only the provided fields are updated (partial update). You can update personal info, preferences, and two-factor authentication settings.

**Auth required:** Yes

**Request body** (`application/json`) — all fields optional:

```json
{
  "email": "new_email@example.com",
  "username": "new_username",
  "first_name": "Johnny",
  "last_name": "Doe",
  "bio": "Sign language enthusiast",
  "country": "US",
  "language": "en"
}
```

**Response** `200 OK`:

```json
{
  "id": 1,
  "username": "new_username",
  "email": "new_email@example.com",
  "first_name": "Johnny",
  "bio": "Sign language enthusiast",
  "country": "US",
  "language": "en",
  "profile_image_url": "/media/profile_images/abc123.jpg",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00"
}
```

**Error responses:**

| Code  | Reason                           |
| ----- | -------------------------------- |
| `401` | Not authenticated                |
| `409` | Email or username already in use |
| `422` | Validation error                 |

______________________________________________________________________

### `POST /api/v1/user/me/profile-image`

**Description:** Upload a profile picture for the current user. The image is stored locally in the `media/profile_images/` directory. Supported formats: JPEG, PNG, GIF, WebP. Maximum file size: 5 MB.

**Auth required:** Yes

**Request** (`multipart/form-data`):

```
file: <image file>
```

**Response** `200 OK`:

```json
{
  "data": {
    "id": 1,
    "username": "john_doe",
    "profile_image_url": "/media/profile_images/abc123def456.jpg",
    "email": "john@example.com"
  }
}
```

**Error responses:**

| Code  | Reason                                    |
| ----- | ----------------------------------------- |
| `400` | Unsupported file type or missing filename |
| `413` | File exceeds 5 MB                         |
| `401` | Not authenticated                         |

______________________________________________________________________

### `DELETE /api/v1/user/me/profile-image`

**Description:** Remove the current user's profile picture. The file is deleted from local storage and the `profile_image_url` field is set to `null`.

**Auth required:** Yes

**Response** `200 OK`:

```json
{
  "message": "Profile image deleted successfully"
}
```

______________________________________________________________________

### `DELETE /api/v1/user/me`

**Description:** Soft-delete the current user's account. The account is marked as inactive but the data is retained. The user cannot log in after this action.

**Auth required:** Yes

**Response** `200 OK`:

```json
{
  "message": "User deleted successfully"
}
```

______________________________________________________________________

## 4. Signs (Reference Images)

### `GET /api/v1/signs/{character}`

**Description:** Retrieve the currently active reference image URL for a given ASL character. This is the "ground truth" image used in the UI to show the user how a sign should look. Images are served from Cloudinary CDN.

**Auth required:** No

**Path parameters:**

| Parameter   | Type   | Description                           |
| ----------- | ------ | ------------------------------------- |
| `character` | string | Single letter A–Z or the word `space` |

**Example:** `GET /api/v1/signs/A`

**Response** `200 OK`:

```json
{
  "id": 1,
  "character": "A",
  "cloudinary_url": "https://res.cloudinary.com/demo/image/upload/v1/asl-signs/A/active.jpg",
  "is_active": true,
  "created_at": "2024-01-10T08:00:00"
}
```

**Error responses:**

| Code  | Reason                                 |
| ----- | -------------------------------------- |
| `404` | Character not found or no active image |

______________________________________________________________________

## 5. Prediction

### `POST /api/v1/predict/`

**Description:** Submit a hand-skeleton image (white landmarks on black background) to the ML model and receive the predicted ASL sign with a confidence score. The model uses MediaPipe to extract skeletal landmarks and a trained CNN (MobileNet) to classify the gesture. This is the core prediction endpoint for single-frame inference.

**Auth required:** Yes

**Request** (`multipart/form-data`):

```
file: <image file>   (jpg, png, gif, or webp — max 5 MB)
```

**Response** `200 OK`:

```json
{
  "success": true,
  "message": "Detected gesture: A (95%)",
  "data": {
    "label": "A",
    "confidence": 95.0
  },
  "query_generated_time": 0.1234
}
```

**Possible label values:**

| Label       | Meaning                               |
| ----------- | ------------------------------------- |
| `A`–`Z`     | Detected ASL letter                   |
| `space`     | Space gesture                         |
| `del`       | Delete/backspace gesture              |
| `nothing`   | No gesture detected but model ran     |
| `no_hand`   | No hand skeleton detected in image    |
| `uncertain` | Prediction confidence below threshold |
| `error`     | Model processing failed               |

**Error responses:**

| Code  | Reason                              |
| ----- | ----------------------------------- |
| `400` | Unsupported file type or file >5 MB |
| `401` | Not authenticated                   |
| `503` | ML model not loaded or unavailable  |

______________________________________________________________________

### `WebSocket /api/v1/predict/ws?token=<access_token>`

**Description:** Real-time sign prediction via WebSocket. Connect once and stream raw image frames as binary messages. The server responds with a JSON prediction result for each frame. Ideal for live camera feeds where low latency matters. Authentication is done via the `token` query parameter (your JWT access token).

**Auth required:** `?token=<access_token>` query parameter

**Connection URL:**

```
ws://localhost:8000/api/v1/predict/ws?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Client → Server:** Send raw binary image data (JPEG/PNG bytes) as a WebSocket binary frame.

**Server → Client** (JSON text frame after each image):

```json
{
  "success": true,
  "data": {
    "label": "A",
    "confidence": 92.5
  },
  "time": "0.1234s",
  "frame": 1
}
```

**On error (server → client):**

```json
{
  "success": false,
  "error": "Model processing failed",
  "frame": 2
}
```

**WebSocket close codes:**

| Code   | Reason                                      |
| ------ | ------------------------------------------- |
| `4001` | Invalid or missing token, or user not found |
| `1000` | Normal closure                              |

______________________________________________________________________

### `GET /api/v1/predict/info`

**Description:** Get metadata about the prediction service — which gestures are supported, the model type currently loaded, and service availability status. Use this to dynamically populate UI gesture lists.

**Auth required:** Yes

**Response** `200 OK`:

```json
{
  "status": "available",
  "model_type": "cnn_mediapipe",
  "supported_gestures": [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "space", "del", "nothing"
  ],
  "total_classes": 29
}
```

______________________________________________________________________

### `GET /api/v1/predict/health`

**Description:** Check whether the ML model is loaded and ready to serve predictions. Returns the model's load status and type. Use this before sending prediction requests to verify the model is available.

**Auth required:** Yes

**Response** `200 OK` (model ready):

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "cnn_mediapipe"
}
```

**Response** `503 Service Unavailable` (model not loaded):

```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_type": null
}
```

______________________________________________________________________

## 6. Sign Detections

Detection records store every sign the user predicted, enabling progress tracking, accuracy analytics, and learning history.

### `POST /api/v1/detection/log`

**Description:** Log a single sign detection result to the database. Call this after every successful prediction to persist the result for analytics. The `session_id` groups detections from the same practice session.

**Auth required:** Yes

**Request body** (`application/json`):

```json
{
  "detected_sign": "A",
  "confidence": 0.95,
  "is_correct": true,
  "session_id": "sess_abc123",
  "duration_seconds": 0.25
}
```

| Field              | Type    | Required | Description                                       |
| ------------------ | ------- | -------- | ------------------------------------------------- |
| `detected_sign`    | string  | ✅       | The predicted sign label (A–Z, space, del, etc.)  |
| `confidence`       | float   | ✅       | Model confidence between 0.0 and 1.0              |
| `is_correct`       | boolean | ❌       | Whether user confirmed the prediction was correct |
| `session_id`       | string  | ❌       | Groups detections by practice session             |
| `duration_seconds` | float   | ❌       | Time taken for the detection                      |

**Response** `201 Created`:

```json
{
  "id": 42,
  "user_id": 1,
  "detected_sign": "A",
  "confidence": 0.95,
  "is_correct": true,
  "session_id": "sess_abc123",
  "duration_seconds": 0.25,
  "created_at": "2024-01-15T10:35:00"
}
```

**Error responses:**

| Code  | Reason                                    |
| ----- | ----------------------------------------- |
| `401` | Not authenticated                         |
| `422` | Confidence out of range or invalid fields |
| `429` | Rate limit exceeded                       |

______________________________________________________________________

### `POST /api/v1/detection/log/batch`

**Description:** Log multiple detection results in a single request. More efficient than making one request per detection, especially when logging a full practice session at once. Accepts 1 to 100 detections per request.

**Auth required:** Yes

**Request body** (`application/json`):

```json
{
  "detections": [
    {
      "detected_sign": "A",
      "confidence": 0.95,
      "is_correct": true,
      "session_id": "sess_abc123",
      "duration_seconds": 0.25
    },
    {
      "detected_sign": "B",
      "confidence": 0.87,
      "is_correct": true,
      "session_id": "sess_abc123",
      "duration_seconds": 0.30
    },
    {
      "detected_sign": "C",
      "confidence": 0.72,
      "is_correct": false,
      "session_id": "sess_abc123",
      "duration_seconds": 0.45
    }
  ]
}
```

**Response** `201 Created`:

```json
{
  "message": "3 detections logged",
  "count": 3
}
```

**Error responses:**

| Code  | Reason                            |
| ----- | --------------------------------- |
| `401` | Not authenticated                 |
| `422` | Empty list or more than 100 items |

______________________________________________________________________

### `GET /api/v1/detection/list`

**Description:** Retrieve a paginated list of all sign detections logged by the current user, ordered by most recent first. Use this to show detection history or feed analytics charts.

**Auth required:** Yes

**Query parameters:**

| Parameter        | Type | Default | Description              |
| ---------------- | ---- | ------- | ------------------------ |
| `page`           | int  | `1`     | Page number              |
| `items_per_page` | int  | `20`    | Items per page (max 100) |

**Example:** `GET /api/v1/detection/list?page=1&items_per_page=20`

**Response** `200 OK`:

```json
{
  "data": [
    {
      "id": 42,
      "user_id": 1,
      "detected_sign": "A",
      "confidence": 0.95,
      "is_correct": true,
      "session_id": "sess_abc123",
      "duration_seconds": 0.25,
      "created_at": "2024-01-15T10:35:00"
    },
    {
      "id": 41,
      "user_id": 1,
      "detected_sign": "B",
      "confidence": 0.87,
      "is_correct": true,
      "session_id": "sess_abc123",
      "duration_seconds": 0.30,
      "created_at": "2024-01-15T10:34:50"
    }
  ],
  "total_count": 150,
  "page": 1,
  "items_per_page": 20
}
```

______________________________________________________________________

### `GET /api/v1/detection/{id}`

**Description:** Retrieve a single detection record by its ID. Only the owner of the detection can access it.

**Auth required:** Yes

**Path parameters:**

| Parameter | Type | Description         |
| --------- | ---- | ------------------- |
| `id`      | int  | Detection record ID |

**Response** `200 OK`:

```json
{
  "id": 42,
  "user_id": 1,
  "detected_sign": "A",
  "confidence": 0.95,
  "is_correct": true,
  "session_id": "sess_abc123",
  "duration_seconds": 0.25,
  "created_at": "2024-01-15T10:35:00"
}
```

**Error responses:**

| Code  | Reason                                         |
| ----- | ---------------------------------------------- |
| `404` | Detection not found or belongs to another user |

______________________________________________________________________

### `PATCH /api/v1/detection/{id}`

**Description:** Update a detection record — typically to mark whether the prediction was actually correct or to adjust the confidence value. Only the owner can update their own detections.

**Auth required:** Yes

**Path parameters:**

| Parameter | Type | Description         |
| --------- | ---- | ------------------- |
| `id`      | int  | Detection record ID |

**Request body** (`application/json`) — all fields optional:

```json
{
  "is_correct": false,
  "confidence": 0.70
}
```

**Response** `200 OK`:

```json
{
  "id": 42,
  "user_id": 1,
  "detected_sign": "A",
  "confidence": 0.70,
  "is_correct": false,
  "session_id": "sess_abc123",
  "duration_seconds": 0.25,
  "created_at": "2024-01-15T10:35:00"
}
```

______________________________________________________________________

### `DELETE /api/v1/detection/{id}`

**Description:** Soft-delete a detection record. The record is marked as deleted but remains in the database for audit purposes. It will no longer appear in list results or analytics.

**Auth required:** Yes

**Path parameters:**

| Parameter | Type | Description         |
| --------- | ---- | ------------------- |
| `id`      | int  | Detection record ID |

**Response** `200 OK`:

```json
{
  "message": "Detection deleted successfully"
}
```

______________________________________________________________________

## 7. Dashboard

### `GET /api/v1/dashboard/`

**Description:** Retrieve a comprehensive analytics summary for the current user's sign language practice. Includes total detections, accuracy rate, daily activity trend, practice time, most-frequent signs, accuracy distribution, and recommended letters to practice. Use this to power the user's learning progress dashboard in the frontend.

**Auth required:** Yes

**Query parameters:**

| Parameter | Type | Default | Description                                         |
| --------- | ---- | ------- | --------------------------------------------------- |
| `days`    | int  | `7`     | Number of past days to include in trend data (1–90) |

**Example:** `GET /api/v1/dashboard/?days=30`

**Response** `200 OK`:

```json
{
  "status_code": 200,
  "message": "Dashboard fetched successfully",
  "data": {
    "stats": {
      "total_signs_detected": 1500,
      "today_signs_count": 42,
      "total_practice_hours": 24.5,
      "today_minutes": 45.0,
      "average_accuracy": 0.88,
      "accuracy_change": 0.05,
      "current_streak": 7
    },
    "frequent_signs": [
      { "sign": "A", "count": 220 },
      { "sign": "B", "count": 185 },
      { "sign": "C", "count": 142 }
    ],
    "daily_activity": [
      { "date": "2024-01-15", "count": 42 },
      { "date": "2024-01-14", "count": 55 },
      { "date": "2024-01-13", "count": 30 }
    ],
    "accuracy_distribution": {
      "high": 950,
      "medium": 400,
      "low": 150
    },
    "mastered_letters": 18,
    "recent_activities": [
      {
        "id": 42,
        "detected_sign": "A",
        "confidence": 0.95,
        "is_correct": true,
        "created_at": "2024-01-15T10:35:00"
      }
    ],
    "recommended_letters": ["J", "Z", "X"]
  }
}
```

| Field                        | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `stats.total_signs_detected` | All-time detection count                          |
| `stats.today_signs_count`    | Detections logged today                           |
| `stats.total_practice_hours` | Total practice duration in hours                  |
| `stats.average_accuracy`     | Overall correct-prediction ratio (0–1)            |
| `stats.accuracy_change`      | Change vs previous period                         |
| `stats.current_streak`       | Consecutive days with at least one detection      |
| `frequent_signs`             | Top detected signs by count                       |
| `daily_activity`             | Per-day detection counts for the requested period |
| `accuracy_distribution`      | Breakdown into high/medium/low confidence buckets |
| `mastered_letters`           | Count of letters with consistently high accuracy  |
| `recommended_letters`        | Letters the user should practice more             |

______________________________________________________________________

## 8. Background Tasks

### `POST /api/v1/tasks/task`

**Description:** Enqueue an asynchronous background job on the ARQ/Redis worker queue. This is a general-purpose endpoint for running jobs that should not block the HTTP response. Rate-limited per user.

**Auth required:** Yes

**Query parameters:**

| Parameter | Type   | Required | Description                             |
| --------- | ------ | -------- | --------------------------------------- |
| `message` | string | ✅       | Payload message for the background task |

**Example:** `POST /api/v1/tasks/task?message=process_report`

**Response** `200 OK`:

```json
{
  "id": "task_7f3e9a12b4c5"
}
```

**Error responses:**

| Code  | Reason              |
| ----- | ------------------- |
| `429` | Rate limit exceeded |
| `401` | Not authenticated   |

______________________________________________________________________

### `GET /api/v1/tasks/task/{task_id}`

**Description:** Poll the status of a previously enqueued background task using its task ID. Returns the current status and result once the job completes.

**Auth required:** Yes

**Path parameters:**

| Parameter | Type   | Description                              |
| --------- | ------ | ---------------------------------------- |
| `task_id` | string | Task ID returned by the enqueue endpoint |

**Response** `200 OK` (task completed):

```json
{
  "id": "task_7f3e9a12b4c5",
  "status": "complete",
  "result": "Task completed successfully"
}
```

**Response** `200 OK` (task in progress):

```json
{
  "id": "task_7f3e9a12b4c5",
  "status": "in_progress",
  "result": null
}
```

______________________________________________________________________

## 9. Tiers

Tiers define subscription levels (e.g., Free, Pro) and are used to enforce rate limits. Admin-only write operations.

### `POST /api/v1/tier`

**Description:** Create a new subscription tier. Admin only. Tiers are later referenced when assigning users and configuring rate limits.

**Auth required:** Yes (superuser)

**Request body** (`application/json`):

```json
{
  "name": "pro",
  "description": "Pro tier with higher limits"
}
```

**Response** `201 Created`:

```json
{
  "id": 2,
  "name": "pro",
  "description": "Pro tier with higher limits",
  "created_at": "2024-01-15T10:00:00"
}
```

______________________________________________________________________

### `GET /api/v1/tiers`

**Description:** List all available subscription tiers with pagination.

**Auth required:** Yes

**Query parameters:**

| Parameter        | Type | Default | Description    |
| ---------------- | ---- | ------- | -------------- |
| `page`           | int  | `1`     | Page number    |
| `items_per_page` | int  | `10`    | Items per page |

**Response** `200 OK`:

```json
{
  "data": [
    { "id": 1, "name": "free", "description": "Free tier", "created_at": "2024-01-01T00:00:00" },
    { "id": 2, "name": "pro", "description": "Pro tier", "created_at": "2024-01-15T10:00:00" }
  ],
  "total_count": 2,
  "page": 1,
  "items_per_page": 10
}
```

______________________________________________________________________

### `GET /api/v1/tier/{name}`

**Description:** Get a single tier by its name.

**Auth required:** Yes

**Response** `200 OK`:

```json
{
  "id": 1,
  "name": "free",
  "description": "Free tier",
  "created_at": "2024-01-01T00:00:00"
}
```

______________________________________________________________________

### `PATCH /api/v1/tier/{name}`

**Description:** Update an existing tier's details. Admin only.

**Auth required:** Yes (superuser)

**Request body** (`application/json`):

```json
{
  "description": "Updated free tier description"
}
```

**Response** `200 OK`:

```json
{
  "id": 1,
  "name": "free",
  "description": "Updated free tier description",
  "created_at": "2024-01-01T00:00:00"
}
```

______________________________________________________________________

### `DELETE /api/v1/tier/{name}`

**Description:** Delete a tier by name. Admin only. Will fail if users are still assigned to this tier.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "message": "Tier deleted successfully"
}
```

______________________________________________________________________

## 10. Rate Limits

Rate limit rules are tied to a tier and a specific API path. Admin-only write operations.

### `POST /api/v1/tier/{tier_name}/rate_limit`

**Description:** Create a rate limit rule for a specific tier and API path. For example, limit the `free` tier to 10 prediction requests per minute. Admin only.

**Auth required:** Yes (superuser)

**Path parameters:**

| Parameter   | Type   | Description                    |
| ----------- | ------ | ------------------------------ |
| `tier_name` | string | The tier to apply this rule to |

**Request body** (`application/json`):

```json
{
  "path": "/api/v1/predict/",
  "limit": 10,
  "period": 60
}
```

| Field    | Type   | Description                        |
| -------- | ------ | ---------------------------------- |
| `path`   | string | API path this rule applies to      |
| `limit`  | int    | Max requests allowed in the period |
| `period` | int    | Time window in seconds             |

**Response** `201 Created`:

```json
{
  "id": 1,
  "tier_name": "free",
  "path": "/api/v1/predict/",
  "limit": 10,
  "period": 60,
  "created_at": "2024-01-15T10:00:00"
}
```

______________________________________________________________________

### `GET /api/v1/tier/{tier_name}/rate_limits`

**Description:** List all rate limit rules configured for a specific tier, paginated.

**Auth required:** Yes

**Response** `200 OK`:

```json
{
  "data": [
    {
      "id": 1,
      "tier_name": "free",
      "path": "/api/v1/predict/",
      "limit": 10,
      "period": 60
    }
  ],
  "total_count": 1,
  "page": 1,
  "items_per_page": 10
}
```

______________________________________________________________________

### `GET /api/v1/tier/{tier_name}/rate_limit/{id}`

**Description:** Get a specific rate limit rule by ID.

**Auth required:** Yes

**Response** `200 OK`: Same structure as a single item in the list above.

______________________________________________________________________

### `PATCH /api/v1/tier/{tier_name}/rate_limit/{id}`

**Description:** Update an existing rate limit rule (limit count or time period). Admin only.

**Auth required:** Yes (superuser)

**Request body** (`application/json`):

```json
{
  "limit": 20,
  "period": 60
}
```

**Response** `200 OK`: Updated rule object.

______________________________________________________________________

### `DELETE /api/v1/tier/{tier_name}/rate_limit/{id}`

**Description:** Delete a rate limit rule. Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "message": "Rate limit deleted successfully"
}
```

______________________________________________________________________

## 11. Admin — Users

All routes in this section require the user to be a **superuser**. Provides full user lifecycle management beyond what users can do on their own profiles.

### `POST /api/v1/admin/auth/user`

**Description:** Create a user account directly (admin bypass — no email verification required). Used to programmatically create test accounts or onboard users.

**Auth required:** Yes (superuser)

**Request body** (`application/json`):

```json
{
  "username": "new_user",
  "email": "new@example.com",
  "password": "SecurePass123!",
  "first_name": "New",
  "last_name": "User",
  "is_superuser": false
}
```

**Response** `201 Created`: Full user profile object.

______________________________________________________________________

### `GET /api/v1/admin/users`

**Description:** List all registered users with pagination. Supports filtering. Admin only.

**Auth required:** Yes (superuser)

**Query parameters:**

| Parameter        | Type | Default | Description    |
| ---------------- | ---- | ------- | -------------- |
| `page`           | int  | `1`     | Page number    |
| `items_per_page` | int  | `20`    | Items per page |

**Response** `200 OK`:

```json
{
  "data": [
    {
      "id": 1,
      "username": "john_doe",
      "email": "john@example.com",
      "is_superuser": false,
      "is_active": true,
      "created_at": "2024-01-15T10:30:00"
    }
  ],
  "total_count": 250,
  "page": 1,
  "items_per_page": 20
}
```

______________________________________________________________________

### `GET /api/v1/admin/user/{username}`

**Description:** Get full profile of any user by username. Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`: Full user profile object including tier assignment.

______________________________________________________________________

### `PATCH /api/v1/admin/user/{username}`

**Description:** Update any field on any user's profile. Admin only. Can also toggle `is_active` and `is_superuser` flags.

**Auth required:** Yes (superuser)

**Request body** (`application/json`):

```json
{
  "is_active": false,
  "is_superuser": false,
  "email": "updated@example.com"
}
```

**Response** `200 OK`: Updated user profile object.

______________________________________________________________________

### `DELETE /api/v1/admin/user/{username}`

**Description:** Soft-delete a user account. Marks the user as inactive but retains all their data. The user cannot log in after this action.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "message": "User deleted successfully"
}
```

______________________________________________________________________

### `DELETE /api/v1/admin/db_user/{username}`

**Description:** Hard-delete a user and all their associated data permanently from the database. **This is irreversible.** Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "message": "User permanently deleted"
}
```

______________________________________________________________________

### `GET /api/v1/admin/user/{username}/tier`

**Description:** Get the subscription tier currently assigned to a user. Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "user_id": 1,
  "username": "john_doe",
  "tier_name": "free"
}
```

______________________________________________________________________

### `PATCH /api/v1/admin/user/{username}/tier`

**Description:** Assign or update a user's subscription tier. Admin only. Changing tiers updates the rate limits that apply to the user.

**Auth required:** Yes (superuser)

**Request body** (`application/json`):

```json
{
  "tier_name": "pro"
}
```

**Response** `200 OK`:

```json
{
  "user_id": 1,
  "username": "john_doe",
  "tier_name": "pro"
}
```

______________________________________________________________________

### `GET /api/v1/admin/user/{username}/rate_limits`

**Description:** View all effective rate limits that apply to a specific user (based on their tier). Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "username": "john_doe",
  "tier_name": "free",
  "rate_limits": [
    {
      "path": "/api/v1/predict/",
      "limit": 10,
      "period": 60
    }
  ]
}
```

______________________________________________________________________

## 12. Admin — Signs

Manage the ASL reference sign image library. Each character (A–Z + space) can have multiple Cloudinary image versions with one marked as "active".

### `GET /api/v1/admin/signs`

**Description:** List all sign image records in the database along with which ASL characters are still missing an active image. Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "signs": [
    {
      "id": 1,
      "character": "A",
      "cloudinary_url": "https://res.cloudinary.com/.../A/v2.jpg",
      "is_active": true,
      "version": 2,
      "created_at": "2024-01-10T08:00:00"
    }
  ],
  "missing_characters": ["J", "Z"]
}
```

______________________________________________________________________

### `GET /api/v1/admin/signs/stats`

**Description:** Get sign image completion statistics — how many of the 27 characters (A–Z + space) have an active image. Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "total_characters": 27,
  "characters_with_images": 25,
  "completion_percent": 92.6,
  "missing": ["J", "Z"]
}
```

______________________________________________________________________

### `GET /api/v1/admin/signs/{character}/images`

**Description:** List all Cloudinary image versions ever uploaded for a specific character. Used in the admin UI to select which version to set as active.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "character": "A",
  "images": [
    {
      "id": 1,
      "cloudinary_url": "https://res.cloudinary.com/.../A/v1.jpg",
      "cloudinary_public_id": "asl-signs/A/v1",
      "is_active": false,
      "version": 1,
      "file_size": 48000
    },
    {
      "id": 2,
      "cloudinary_url": "https://res.cloudinary.com/.../A/v2.jpg",
      "cloudinary_public_id": "asl-signs/A/v2",
      "is_active": true,
      "version": 2,
      "file_size": 52000
    }
  ]
}
```

______________________________________________________________________

### `POST /api/v1/admin/signs/{character}/upload`

**Description:** Upload a new reference sign image for a character. The uploaded image is sent to Cloudinary and automatically set as the new active image for that character. Admin only.

**Auth required:** Yes (superuser)

**Path parameters:**

| Parameter   | Type   | Description    |
| ----------- | ------ | -------------- |
| `character` | string | A–Z or `space` |

**Request** (`multipart/form-data`):

```
file: <image file>
```

**Response** `200 OK`:

```json
{
  "id": 3,
  "character": "A",
  "cloudinary_url": "https://res.cloudinary.com/.../A/v3.jpg",
  "cloudinary_public_id": "asl-signs/A/v3",
  "file_size": 55000,
  "width": 640,
  "height": 480,
  "mime_type": "image/jpeg",
  "version": 3,
  "is_active": true,
  "created_at": "2024-01-15T12:00:00"
}
```

**Error responses:**

| Code  | Reason                   |
| ----- | ------------------------ |
| `400` | Invalid character value  |
| `502` | Cloudinary upload failed |

______________________________________________________________________

### `POST /api/v1/admin/signs/bulk-upload`

**Description:** Upload multiple sign images at once. The character is determined automatically from the filename (e.g., `A.jpg` → character `A`, `space.jpg` → `space`). Each image is uploaded to Cloudinary and set as active for its character. Admin only.

**Auth required:** Yes (superuser)

**Request** (`multipart/form-data`):

```
files: [A.jpg, B.png, C.jpg, ...]
```

**Response** `200 OK`:

```json
{
  "uploaded": ["A", "B", "C"],
  "failed": [],
  "count": 3
}
```

______________________________________________________________________

### `PUT /api/v1/admin/signs/{character}/set-active`

**Description:** Switch the active image for a character to a different previously-uploaded Cloudinary version. Use this to roll back to a previous image without re-uploading. Admin only.

**Auth required:** Yes (superuser)

**Request body** (`application/json`):

```json
{
  "cloudinary_public_id": "asl-signs/A/v1"
}
```

**Response** `200 OK`:

```json
{
  "character": "A",
  "active_image_id": 1,
  "cloudinary_url": "https://res.cloudinary.com/.../A/v1.jpg"
}
```

______________________________________________________________________

## 13. Admin — Dashboard

### `GET /api/v1/admin/dashboard/`

**Description:** Retrieve the admin analytics dashboard — system-wide statistics including total users, daily active users, total detections, growth percentages, recent user activity, top active users, and the health status of all system services (database, Redis, ML model, WebSocket, file storage). Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

```json
{
  "data": {
    "stats": {
      "total_users": 250,
      "active_users_today": 45,
      "total_detections": 54200,
      "system_health": 100,
      "user_growth_percent": 12.5,
      "detection_growth_percent": 8.3
    },
    "recent_activities": [
      {
        "id": 42,
        "type": "detection",
        "emoji": "🤟",
        "title": "john_doe detected 'A'",
        "description": "Confidence: 95%",
        "time_ago": "2m ago"
      }
    ],
    "system_services": [
      { "name": "Web Server (Uvicorn)", "status": "online" },
      { "name": "PostgreSQL Database", "status": "online" },
      { "name": "ML Model", "status": "online" },
      { "name": "WebSocket Server", "status": "online" },
      { "name": "File Storage (Cloudinary)", "status": "online" }
    ],
    "top_users": [
      { "username": "john_doe", "detections": 450 },
      { "username": "jane_smith", "detections": 320 }
    ]
  }
}
```

______________________________________________________________________

### `GET /api/v1/admin/dashboard/export`

**Description:** Export a full user activity report as a CSV file. The CSV includes user details, detection counts, accuracy stats, and last activity. Useful for external reporting, billing, or audit purposes. Admin only.

**Auth required:** Yes (superuser)

**Response** `200 OK`:

- Content-Type: `text/csv`
- Content-Disposition: `attachment; filename="signsync_report.csv"`

```csv
username,email,total_detections,average_accuracy,last_activity,tier,joined_at
john_doe,john@example.com,450,0.92,2024-01-15T10:35:00,pro,2024-01-01T00:00:00
jane_smith,jane@example.com,320,0.88,2024-01-14T18:20:00,free,2024-01-05T00:00:00
```

______________________________________________________________________

*For interactive API exploration, visit `/docs` (Swagger UI) or `/redoc` when running with `ENVIRONMENT=local`.*
