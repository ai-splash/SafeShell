"""
SafeShell routes.

    GET  /api/safeshell/scenarios        -> the 7 predefined demo scenarios
    POST /api/safeshell/analyze          -> intent + risk analysis (Analyze stage)
    POST /api/safeshell/simulate         -> impact simulation + undo plan; creates a transaction
    POST /api/safeshell/execute          -> confirm -> execute -> verify -> commit (or block/fail)
    POST /api/safeshell/rollback         -> roll a COMMITTED transaction back
    GET  /api/safeshell/transactions     -> full transaction history (seeded + created)
    GET  /api/safeshell/transactions/{id}-> a single transaction
    GET  /api/safeshell/status           -> Security Status panel counters

This is a self-contained, deterministic layer (see
`app.services.safeshell_engine`): every scenario is predefined demo data,
nothing here performs a real destructive Linux operation, and it does not
depend on Ollama, the internet, or any external service. It is additive -
it does not modify the existing Safe Command Execution pipeline in
`app/routes/commands.py`.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.logger import get_logger
from app.services import safeshell_engine

router = APIRouter(prefix="/api/safeshell", tags=["SafeShell"])
logger = get_logger(__name__)


class AnalyzeRequest(BaseModel):
    scenario_id: str | None = None
    command: str | None = None


class SimulateRequest(BaseModel):
    scenario_id: str | None = None
    command: str | None = None


class ExecuteRequest(BaseModel):
    transaction_id: str
    confirm: bool = True


class RollbackRequest(BaseModel):
    transaction_id: str


@router.get("/scenarios")
def scenarios():
    """The 7 predefined demo scenarios for the Demo Scenario Selector."""
    return {"scenarios": safeshell_engine.list_scenarios()}


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Intent Understanding + Command Risk Analysis (no transaction created yet)."""
    try:
        return safeshell_engine.analyze(request.scenario_id, request.command)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/simulate")
def simulate(request: SimulateRequest):
    """Impact Simulation + Undo Plan Generation. Creates and returns a
    transaction in AWAITING_CONFIRMATION or BLOCKED state."""
    try:
        return safeshell_engine.simulate(request.scenario_id, request.command)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/execute")
def execute(request: ExecuteRequest):
    """User Confirmation -> Safe Execution -> Verification -> Commit."""
    try:
        return safeshell_engine.execute_transaction(request.transaction_id, request.confirm)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/rollback")
def rollback(request: RollbackRequest):
    """Rollback Transaction -> Previous State Restored -> Verification Successful."""
    try:
        return safeshell_engine.rollback_transaction(request.transaction_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/transactions")
def transactions():
    return {"transactions": safeshell_engine.list_transactions()}


@router.get("/transactions/{transaction_id}")
def transaction(transaction_id: str):
    txn = safeshell_engine.get_transaction(transaction_id)
    if not txn:
        raise HTTPException(status_code=404, detail=f"No transaction found with id {transaction_id}")
    return txn


@router.get("/status")
def status():
    return safeshell_engine.security_status()
