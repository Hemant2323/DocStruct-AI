"""Health and readiness endpoints.

Two endpoints, because they answer two different questions:

* ``/health`` — *is the process alive?* Checks nothing external. Used by a process
  supervisor to decide whether to restart.
* ``/ready``  — *can it serve traffic?* Checks dependencies. Used by a load balancer to
  decide whether to route requests here.

Conflating them causes a database blip to trigger pointless application restarts.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import SettingsDep
from app.api.schemas.common import HealthResponse, ReadinessResponse
from app.core.database import check_database_connection

router = APIRouter(tags=["health"])

API_VERSION = "0.1.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    response_description="The application process is running.",
)
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the application process is up.

    Checks nothing external and always returns 200 if the process can answer at all.
    That is the point: a liveness probe that also checked the database would report the
    *application* as dead during a database blip, and a supervisor would restart a
    perfectly healthy process, losing any in-flight work.

    Use `/ready` to find out whether dependencies are actually reachable.
    """
    return HealthResponse(
        app=settings.app_name,
        version=API_VERSION,
        environment=settings.app_env.value,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    response_description="Per-dependency health. 200 when ready, 503 when not.",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessResponse,
            "description": (
                "At least one dependency is unreachable. The body names which check "
                "failed. This is an expected operational state, not a crash."
            ),
        }
    },
)
async def ready(response: Response) -> ReadinessResponse:
    """Report whether the application's dependencies are reachable.

    Returns 200 with `status: "ready"` when every check passes, and 503 with
    `status: "not_ready"` when any fails, so infrastructure can act on the status code
    without parsing the body. The `checks` map names each dependency individually.
    """
    checks = {"database": await check_database_connection()}
    all_healthy = all(checks.values())

    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if all_healthy else "not_ready",
        checks=checks,
    )
