from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class ClientCacheMiddleware(BaseHTTPMiddleware):
    """Middleware to set the `Cache-Control` header for client-side caching.

    GET responses are cached for `max_age` seconds.
    Admin routes and all non-GET methods always receive `no-store` so that
    clients always re-fetch data after a mutation or when viewing live admin data.

    Parameters
    ----------
    app: FastAPI
        The FastAPI application instance.
    max_age: int, optional
        Duration (in seconds) for which GET responses should be cached. Defaults to 60 seconds.

    Attributes
    ----------
    max_age: int
        Duration (in seconds) for which GET responses should be cached.
    """

    def __init__(self, app: FastAPI, max_age: int = 60) -> None:
        super().__init__(app)
        self.max_age = max_age

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request and set the `Cache-Control` header in the response.

        Parameters
        ----------
        request: Request
            The incoming request.
        call_next: RequestResponseEndpoint
            The next middleware or route handler in the processing chain.

        Returns
        -------
        Response
            The response object with the `Cache-Control` header set.

        Note
        ----
            - This method is automatically called by Starlette for processing the request-response cycle.
        """
        response: Response = await call_next(request)
        if request.method == "GET" and "/admin/" not in request.url.path:
            response.headers["Cache-Control"] = f"public, max-age={self.max_age}"
        else:
            response.headers["Cache-Control"] = "no-store"
        return response
