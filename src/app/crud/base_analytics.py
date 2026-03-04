from typing import Any, Generic, TypeVar

from fastcrud import FastCRUD
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")
UpdateInternalSchemaType = TypeVar("UpdateInternalSchemaType")
DeleteSchemaType = TypeVar("DeleteSchemaType")
ReadSchemaType = TypeVar("ReadSchemaType")


class AnalyticsCRUD(
    FastCRUD[
        ModelType,
        CreateSchemaType,
        UpdateSchemaType,
        UpdateInternalSchemaType,
        DeleteSchemaType,
        ReadSchemaType,
    ],
    Generic[
        ModelType,
        CreateSchemaType,
        UpdateSchemaType,
        UpdateInternalSchemaType,
        DeleteSchemaType,
        ReadSchemaType,
    ],
):
    """
    Base analytics CRUD that extends FastCRUD
    Add shared aggregate() and analytics helpers here
    """

    async def aggregate(
        self,
        db: AsyncSession,
        *expressions,
        group_by: list | None = None,
        order_by: list | None = None,
        limit: int | None = None,
        one: bool = False,
        **filters: Any,
    ):
        parsed_filters = self._filter_processor.parse_filters(**filters)

        stmt = select(*expressions).filter(*parsed_filters)

        if group_by:
            stmt = stmt.group_by(*group_by)

        if order_by:
            stmt = stmt.order_by(*order_by)

        if limit:
            stmt = stmt.limit(limit)

        result = await db.execute(stmt)

        if one:
            row = result.one_or_none()
            return dict(row._mapping) if row else None

        return [dict(r._mapping) for r in result.all()]
