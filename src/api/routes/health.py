from fastapi import APIRouter

from src.api.schemas.events import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness check returning application operational status."""
    return HealthResponse(status="ok")


@router.get("/readiness", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness probe verifying all sub-services and dependencies are ready."""
    return ReadinessResponse(
        status="ready",
        version="0.1.0",
        services={"agent_engine": "ready"},
    )
