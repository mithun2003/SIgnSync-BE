from datetime import UTC, datetime, timedelta

from sqlalchemy import Date, case, cast, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..crud.crud_sign_detections import crud_sign_detections
from ..schemas.sign_detection import (
    AccuracyDistribution,
    DailyActivity,
    DashboardResponse,
    DashboardStats,
    FrequentSign,
    RecentActivity,
    RecommendedLetter,
)


class DashboardService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_dashboard(self, user_id: int, days: int) -> DashboardResponse:
        model = crud_sign_detections.model
        now = datetime.now(UTC)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        since = now - timedelta(days=days)

        # ─────────────────────────
        # 1️⃣ STATS
        # ─────────────────────────
        stats_raw = await crud_sign_detections.aggregate(
            self.db,
            func.count(model.id).label("total_signs"),
            func.count(
                case(
                    (model.created_at >= today_start, model.id),
                )
            ).label("today_signs"),
            func.coalesce(func.avg(model.confidence) * 100, 0).label("avg_accuracy"),
            func.coalesce(
                func.avg(
                    case(
                        (
                            model.created_at.between(yesterday_start, today_start),
                            model.confidence,
                        ),
                    )
                )
                * 100,
                0,
            ).label("yesterday_accuracy"),
            func.coalesce(func.sum(model.duration_seconds), 0).label("total_seconds"),
            func.coalesce(
                func.sum(
                    case(
                        (model.created_at >= today_start, model.duration_seconds),
                        else_=0,
                    )
                ),
                0,
            ).label("today_seconds"),
            one=True,
            user_id=user_id,
            is_deleted=False,
        )

        total_hours = round(float(stats_raw["total_seconds"]) / 3600, 1)
        today_minutes = round(float(stats_raw["today_seconds"]) / 60, 1)
        avg_accuracy = round(float(stats_raw["avg_accuracy"]), 1)
        yesterday_acc = float(stats_raw["yesterday_accuracy"])
        accuracy_change = round(avg_accuracy - yesterday_acc, 1) if yesterday_acc > 0 else 0.0

        # ─────────────────────────
        # 2️⃣ FREQUENT SIGNS
        # ─────────────────────────
        frequent_raw = await crud_sign_detections.aggregate(
            self.db,
            model.detected_sign.label("sign"),
            func.count(model.id).label("count"),
            group_by=[model.detected_sign],
            order_by=[desc("count")],
            limit=10,
            user_id=user_id,
            is_deleted=False,
            created_at__gte=since,
        )

        frequent_signs = [FrequentSign(**f) for f in frequent_raw]

        # ─────────────────────────
        # 3️⃣ DAILY ACTIVITY
        # ─────────────────────────
        activity_rows = await crud_sign_detections.aggregate(
            self.db,
            cast(model.created_at, Date).label("date"),
            func.count(model.id).label("count"),
            group_by=[cast(model.created_at, Date)],
            order_by=[cast(model.created_at, Date)],
            user_id=user_id,
            is_deleted=False,
            created_at__gte=since,
        )

        activity_map = {str(r["date"]): r["count"] for r in activity_rows}

        daily_activity = []
        for i in range(days):
            date = (since + timedelta(days=i + 1)).date()
            daily_activity.append(
                DailyActivity(
                    date=str(date),
                    count=activity_map.get(str(date), 0),
                )
            )

        # ─────────────────────────
        # 4️⃣ ACCURACY BY SIGN
        # ─────────────────────────
        accuracy_raw = await crud_sign_detections.aggregate(
            self.db,
            model.detected_sign.label("sign"),
            func.count(model.id).label("count"),
            (func.avg(model.confidence) * 100).label("accuracy"),
            group_by=[model.detected_sign],
            user_id=user_id,
            is_deleted=False,
        )

        for r in accuracy_raw:
            r["accuracy"] = round(float(r["accuracy"]), 1)

        high = sum(1 for a in accuracy_raw if a["accuracy"] >= 90)
        medium = sum(1 for a in accuracy_raw if 70 <= a["accuracy"] < 90)
        low = sum(1 for a in accuracy_raw if a["accuracy"] < 70)

        accuracy_dist = AccuracyDistribution(high=high, medium=medium, low=low)

        mastered = high

        recommended = [
            RecommendedLetter(
                char=a["sign"],
                count=a["count"],
                accuracy=a["accuracy"],
            )
            for a in sorted(
                [a for a in accuracy_raw if a["accuracy"] < 70],
                key=lambda x: x["accuracy"],
            )[:3]
        ]

        # ─────────────────────────
        # 5️⃣ STREAK
        # ─────────────────────────
        streak_rows = await crud_sign_detections.aggregate(
            self.db,
            cast(model.created_at, Date).label("date"),
            group_by=[cast(model.created_at, Date)],
            order_by=[desc(cast(model.created_at, Date))],
            user_id=user_id,
            is_deleted=False,
        )

        dates = [r["date"] for r in streak_rows]
        streak = 0

        if dates:
            today = now.date()
            if dates[0] == today or dates[0] == today - timedelta(days=1):
                expected = dates[0]
                for d in dates:
                    if d == expected:
                        streak += 1
                        expected -= timedelta(days=1)
                    else:
                        break

        # ─────────────────────────
        # 6️⃣ RECENT ACTIVITIES
        # ─────────────────────────
        recent_result = await crud_sign_detections.get_multi(
            self.db,
            limit=5,
            sort_columns="created_at",
            sort_orders="desc",
            user_id=user_id,
            is_deleted=False,
        )

        recent_rows = recent_result["data"]  # 🔥 THIS IS THE FIX

        recent_activities = []

        for det in recent_rows:
            confidence_pct = round(det["confidence"] * 100)

            recent_activities.append(
                RecentActivity(
                    id=det["id"],
                    emoji="🎯" if confidence_pct >= 90 else "📚" if confidence_pct >= 70 else "💪",
                    description=f'Letter "{det["detected_sign"]}" ({confidence_pct}%)',
                    time_ago="Recently",
                    badge="success" if confidence_pct >= 90 else None,
                    badge_text=f"{confidence_pct}%" if confidence_pct >= 90 else None,
                )
            )

        # ─────────────────────────
        # FINAL RESPONSE
        # ─────────────────────────
        stats = DashboardStats(
            total_signs_detected=int(stats_raw["total_signs"]),
            today_signs_count=int(stats_raw["today_signs"]),
            total_practice_hours=total_hours,
            today_minutes=today_minutes,
            average_accuracy=avg_accuracy,
            accuracy_change=accuracy_change,
            current_streak=streak,
        )

        return DashboardResponse(
            stats=stats,
            frequent_signs=frequent_signs,
            daily_activity=daily_activity,
            accuracy_distribution=accuracy_dist,
            mastered_letters=mastered,
            recent_activities=recent_activities,
            recommended_letters=recommended,
        )
