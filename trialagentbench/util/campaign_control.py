"""Durable request reservations shared by native campaign workers and retries."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue

from trialagentbench.contracts.campaign import CampaignConfigurationV1, CampaignModelV1, CampaignPlanV1
from trialagentbench.io import canonical_payload_sha256, read_json_model
from trialagentbench.io.json import loads_json
from trialagentbench.ports.llm_provider import JsonObject, ProviderAccountingError
from trialagentbench.util.provider_redaction import redact_provider_text
from trialagentbench.util.reported_cost import RunStopRequested

if TYPE_CHECKING:
    import httpx2


def request_control_for_run(
    *, campaign_root: Path | None, job_id: str | None, run_root: Path
) -> CampaignRequestControl | None:
    """Bind native runtime control to one frozen campaign invocation."""
    from trialagentbench_verifier.io import sha256_file

    if campaign_root is None and job_id is None:
        return None
    if campaign_root is None or job_id is None:
        raise ValueError("Campaign execution requires both root and job identity")
    plan = read_json_model(CampaignPlanV1, campaign_root / "campaign.json")
    job = next((run for run in plan.runs if run.job_id == job_id), None)
    if job is None or (campaign_root / job.relative_run_root).resolve(strict=True) != run_root.resolve(strict=True):
        raise ValueError("Native run does not belong to this campaign job")
    custody = "run_config.json" if job.benchmark == "trialeval" else "invocation.json"
    if sha256_file(run_root / custody) != job.invocation_sha256:
        raise ValueError("Campaign spending control cannot bind a changed invocation")
    return CampaignRequestControl(
        database=campaign_root / "requests.sqlite", configuration=plan.configuration, model_name=job.model_name
    )


class CampaignCapacityStop(RunStopRequested):
    """Suspend pending work without converting infrastructure limits into model failures."""

    def __init__(self, message: str, *, campaign_wide: bool = True) -> None:
        super().__init__(message)
        self.campaign_wide = campaign_wide
        self.raw_response: JsonObject | None = None


class _Route(BaseModel):
    model_config = ConfigDict(extra="forbid")
    only: tuple[str, ...] = Field(min_length=1, max_length=1)
    allow_fallbacks: Literal[False]


class _FunctionTool(BaseModel):
    """Only participant-local function tools; server tools such as web search are rejected."""

    model_config = ConfigDict(extra="ignore")
    type: Literal["function"]


class _Request(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    provider: _Route
    completion_limit: int = Field(gt=0, validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    service_tier: Literal["default"] | None = None
    stream: Literal[False] = False
    plugins: tuple[JsonValue, ...] = ()
    web_search_options: None = None
    tools: tuple[_FunctionTool, ...] = ()


class _Usage(BaseModel):
    cost: Decimal | None = None
    prompt_tokens: int | None = Field(default=None, ge=0, strict=True)
    completion_tokens: int | None = Field(default=None, ge=0, strict=True)
    total_tokens: int | None = Field(default=None, ge=0, strict=True)


class _Response(BaseModel):
    error: JsonValue = None
    choices: tuple[JsonValue, ...] = ()
    usage: _Usage = Field(default_factory=_Usage)


class _Timeouts(BaseModel):
    connect: float | None = None
    read: float | None = None
    write: float | None = None
    pool: float | None = None


# Monotonic values are comparable between workers only within the same boot.
# Platforms without a kernel boot ID share an epoch through the ledger and
# rotate it only when the actual monotonic clock moves backwards across a reboot.
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_CLOCK_DOMAIN = _BOOT_ID_PATH.read_text(encoding="ascii").strip() if _BOOT_ID_PATH.exists() else None


def _request_clock_domain(connection: sqlite3.Connection) -> str:
    """Share a monotonic epoch between processes, including on non-Linux hosts."""

    if _CLOCK_DOMAIN is not None:
        return _CLOCK_DOMAIN
    connection.execute(
        "CREATE TABLE IF NOT EXISTS request_clock ("
        "id INTEGER PRIMARY KEY CHECK(id=1), epoch TEXT NOT NULL, observed REAL NOT NULL)"
    )
    record = connection.execute("SELECT epoch,observed FROM request_clock WHERE id=1").fetchone()
    # Sample inside BEGIN IMMEDIATE: an older caller timestamp can be stale
    # after lock contention, whereas this observation is serialized with writes.
    observed = time.monotonic()
    epoch = uuid4().hex if record is None or observed < record[1] else record[0]
    connection.execute(
        "INSERT INTO request_clock VALUES (1,?,?) ON CONFLICT(id) DO UPDATE SET epoch=excluded.epoch,observed=excluded.observed",
        (epoch, observed),
    )
    return str(epoch)


# Ledger scope recording exhaustion of the campaign-wide limit; model names are identifiers.
CAMPAIGN_BUDGET_SCOPE = "*"


def _read_only_ledger(database: Path, configuration: CampaignConfigurationV1) -> sqlite3.Connection:
    """Open an existing ledger read-only after binding it to the frozen campaign."""
    if database.is_symlink():
        raise ValueError("Campaign spending ledger must not be a symbolic link")
    connection = sqlite3.connect(database.resolve(strict=True).as_uri() + "?mode=ro", uri=True)
    identity = canonical_payload_sha256(configuration.model_dump(mode="json"))
    if connection.execute("SELECT digest FROM identity").fetchall() != [(identity,)]:
        connection.close()
        raise ValueError("Spending ledger belongs to a different frozen campaign")
    return connection


def rebind_ledger(database: Path, previous: CampaignConfigurationV1, current: CampaignConfigurationV1) -> None:
    """Move a copied ledger from its frozen campaign to that campaign rebound to another release."""
    if database.is_symlink():
        raise ValueError("Campaign spending ledger must not be a symbolic link")
    with closing(sqlite3.connect(database.resolve(strict=True), isolation_level=None)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        updated = connection.execute(
            "UPDATE identity SET digest=? WHERE digest=?",
            (
                canonical_payload_sha256(current.model_dump(mode="json")),
                canonical_payload_sha256(previous.model_dump(mode="json")),
            ),
        ).rowcount
        if updated != 1 or connection.execute("SELECT COUNT(*) FROM identity").fetchone() != (1,):
            connection.execute("ROLLBACK")
            raise ValueError("Spending ledger belongs to a different frozen campaign")
        connection.execute("COMMIT")


def exhausted_budgets(database: Path, configuration: CampaignConfigurationV1) -> dict[str, str]:
    """Return recorded permanent spending-limit exhaustion, keyed by model name or campaign scope.

    Budgets never refill, so a recorded exhaustion makes every pending task of
    that model (or, for the campaign scope, of every model) budget-censored.
    """
    if not database.exists():
        return {}
    with closing(_read_only_ledger(database, configuration)) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "exhausted" not in tables:
            return {}
        return dict(connection.execute("SELECT scope,detail FROM exhausted ORDER BY scope").fetchall())


def print_spending_status(database: Path, configuration: CampaignConfigurationV1) -> None:
    """Report actual attempt reservations without opening or changing a spending account."""
    if not database.exists():
        print("Provider attempts: 0; no spending ledger yet")
        return
    with closing(_read_only_ledger(database, configuration)) as connection:
        for name, attempts, charge, uncertain, excess in connection.execute(
            "SELECT model,COUNT(*),SUM(charge),SUM(1-settled),SUM(excess) FROM attempts GROUP BY model ORDER BY model"
        ):
            print(
                f"{name}: HTTP attempts={attempts}; charged-or-reserved microUSD={charge}; "
                f"unsettled attempts={uncertain}; reservation overruns={excess}"
            )
    for scope, detail in exhausted_budgets(database, configuration).items():
        label = "campaign" if scope == CAMPAIGN_BUDGET_SCOPE else f"model {scope}"
        print(f"Spending limit exhausted ({label}): {detail}. Its pending tasks are budget-censored at finish.")
    print("Unsettled attempts retain their full reservation; this is not a provider billing statement.")


class CampaignRequestControl:
    """Reserve each actual HTTP attempt, retaining uncertain charges across crashes."""

    def __init__(self, *, database: Path, configuration: CampaignConfigurationV1, model_name: str) -> None:
        self.database = database
        self._clock_domain = ""
        self._opened_monotonic = time.monotonic()
        self.configuration = configuration
        self.model = next((model for model in configuration.models if model.name == model_name), None)
        if self.model is None:
            raise ValueError("Unknown campaign model")
        database.parent.mkdir(parents=True, exist_ok=True)
        identity = canonical_payload_sha256(configuration.model_dump(mode="json"))
        with sqlite3.connect(database, isolation_level=None, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._clock_domain = _request_clock_domain(connection)
            connection.execute("CREATE TABLE IF NOT EXISTS identity (digest TEXT PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS control (id INTEGER PRIMARY KEY CHECK(id=1), cancelled INTEGER)"
            )
            connection.execute("INSERT OR IGNORE INTO control VALUES (1,0)")
            connection.execute("CREATE TABLE IF NOT EXISTS exhausted (scope TEXT PRIMARY KEY, detail TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS attempts ("
                "id INTEGER PRIMARY KEY, model TEXT, route TEXT, started REAL, "
                "tokens INTEGER, charge INTEGER, settled INTEGER, excess INTEGER DEFAULT 0, "
                "active_until REAL NOT NULL DEFAULT 0)"
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(attempts)")}
            if "clock_domain" not in columns:
                connection.execute("ALTER TABLE attempts ADD COLUMN clock_domain TEXT NOT NULL DEFAULT ''")
            identities = connection.execute("SELECT digest FROM identity").fetchall()
            if not identities:
                connection.execute("INSERT INTO identity VALUES (?)", (identity,))
            elif identities != [(identity,)]:
                raise ValueError("Spending ledger belongs to a different frozen campaign")
            connection.commit()

    def set_cancelled(self, cancelled: bool) -> None:
        """Stop new requests during drain, or explicitly reopen them on campaign resume."""
        with sqlite3.connect(self.database, timeout=30) as connection:
            connection.execute("UPDATE control SET cancelled=? WHERE id=1", (int(cancelled),))

    def reserve(self, content: bytes, *, timeout_seconds: float | None = None, active_transport: bool = False) -> int:
        """Validate the actual route and reserve funds and a conservative token window."""
        model = self.model
        if model is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Campaign model is absent")
        request = _Request.model_validate_json(content)
        if (
            request.model != model.model
            or request.provider.only != (model.upstream_provider,)
            or request.completion_limit > model.maximum_completion_tokens
            or request.plugins
        ):
            raise ValueError("Actual provider request differs from the frozen campaign route or limits")
        token_bound = len(content) + request.completion_limit
        if token_bound > model.reservation_tokens:
            raise CampaignCapacityStop(
                "Request exceeds the declared conservative prompt/completion token reservation", campaign_wide=False
            )
        deadline = time.monotonic() + min(
            model.request_timeout_seconds,
            model.request_timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
        charge = (
            model.reservation_micro_usd
            if model.request_pricing is None
            else model.request_pricing.reserve_micro_usd(
                prompt_tokens=len(content), completion_tokens=request.completion_limit
            )
        )
        if charge > model.reservation_micro_usd:
            raise CampaignCapacityStop(
                "Request cost bound exceeds the configured budget; increase the budget or reduce request size",
                campaign_wide=False,
            )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CampaignCapacityStop(
                    "Provider capacity window unavailable; retain pending work and resume", campaign_wide=False
                )
            now = time.monotonic()
            attempt = self.try_reserve(
                model=model,
                now=now,
                tokens=token_bound,
                micro_usd=charge,
                active_until=now + remaining if active_transport else 0,
                deadline=deadline,
            )
            if attempt is not None:
                return attempt
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.25, remaining))

    def try_reserve(
        self,
        *,
        model: CampaignModelV1,
        now: float,
        tokens: int | None = None,
        micro_usd: int | None = None,
        active_until: float = 0,
        deadline: float | None = None,
    ) -> int | None:
        """Wait for active transport or rate capacity while retaining all unresolved charges."""
        requested_tokens = model.reservation_tokens if tokens is None else tokens
        charge = model.reservation_micro_usd if micro_usd is None else micro_usd
        if not 0 < charge <= model.reservation_micro_usd:
            raise ValueError("Request charge must fit the reservation ceiling")
        if not 0 < requested_tokens <= model.reservation_tokens:
            raise ValueError("Request token bound must fit the declared reservation ceiling")
        if deadline is not None and time.monotonic() >= deadline:
            return None
        try:
            with sqlite3.connect(
                self.database, isolation_level=None, timeout=30 if deadline is None else 0
            ) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._clock_domain = _request_clock_domain(connection)
                if connection.execute("SELECT cancelled FROM control WHERE id=1").fetchone() != (0,):
                    raise CampaignCapacityStop("Campaign stop requested; unfinished work remains pending")
                if connection.execute("SELECT 1 FROM attempts WHERE excess=1 LIMIT 1").fetchone():
                    raise CampaignCapacityStop(
                        "A previous charge or token usage exceeded its reservation; campaign requires usage review"
                    )
                for scope, detail in connection.execute(
                    "SELECT scope,detail FROM exhausted WHERE scope IN (?,?) ORDER BY scope",
                    (CAMPAIGN_BUDGET_SCOPE, model.name),
                ):
                    # Exhaustion is permanent: budgets never refill, so no later request may run.
                    raise CampaignCapacityStop(detail, campaign_wide=scope == CAMPAIGN_BUDGET_SCOPE)
                total, model_total, permanent, model_permanent = connection.execute(
                    "SELECT COALESCE(SUM(charge),0), "
                    "COALESCE(SUM(CASE WHEN model=? THEN charge ELSE 0 END),0), "
                    "COALESCE(SUM(CASE WHEN settled=1 OR clock_domain!=? OR active_until<=? THEN charge ELSE 0 END),0), "
                    "COALESCE(SUM(CASE WHEN model=? AND (settled=1 OR clock_domain!=? OR active_until<=?) THEN charge ELSE 0 END),0) "
                    "FROM attempts",
                    (model.name, self._clock_domain, now, model.name, self._clock_domain, now),
                ).fetchone()
                campaign_limit = self.configuration.operations.spend_limit_micro_usd
                exhausted = None
                if permanent + charge > campaign_limit:
                    exhausted = (
                        CAMPAIGN_BUDGET_SCOPE,
                        "Campaign spending reservations exhausted before transport: "
                        f"retained {permanent} + requested {charge} exceeds {campaign_limit} microUSD",
                    )
                elif model_permanent + charge > model.spend_limit_micro_usd:
                    exhausted = (
                        model.name,
                        "Model spending reservations exhausted before transport: "
                        f"retained {model_permanent} + requested {charge} exceeds {model.spend_limit_micro_usd} microUSD",
                    )
                if exhausted is not None:
                    connection.execute("INSERT INTO exhausted VALUES (?,?)", exhausted)
                    connection.commit()
                    raise CampaignCapacityStop(exhausted[1], campaign_wide=exhausted[0] == CAMPAIGN_BUDGET_SCOPE)
                if total + charge > campaign_limit or model_total + charge > model.spend_limit_micro_usd:
                    return None
                count, tokens = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(tokens),0) FROM attempts WHERE route=? AND "
                    "((clock_domain=? AND started>?) OR (clock_domain!=? AND ?<60))",
                    (
                        model.upstream_provider,
                        self._clock_domain,
                        now - 60,
                        self._clock_domain,
                        now - self._opened_monotonic,
                    ),
                ).fetchone()
                if count >= model.requests_per_minute or tokens + requested_tokens > model.tokens_per_minute:
                    return None
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                cursor = connection.execute(
                    "INSERT INTO attempts(model,route,started,tokens,charge,settled,active_until,clock_domain) VALUES(?,?,?,?,?,0,?,?)",
                    (
                        model.name,
                        model.upstream_provider,
                        now,
                        requested_tokens,
                        charge,
                        active_until,
                        self._clock_domain,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("Spending reservation did not obtain an identity")
                connection.commit()
                return cursor.lastrowid
        except sqlite3.OperationalError as error:
            if deadline is not None and error.sqlite_errorcode == sqlite3.SQLITE_BUSY:
                return None
            raise

    def finish_attempt(self, attempt: int) -> None:
        """End transport activity without releasing any unresolved monetary liability."""
        model = self.model
        if model is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Campaign model is absent")
        with sqlite3.connect(self.database, timeout=30) as connection:
            changed = connection.execute(
                "UPDATE attempts SET active_until=0 WHERE id=? AND model=?", (attempt, model.name)
            )
            if changed.rowcount != 1:
                raise ValueError("Unknown or foreign spending reservation")

    def settle(self, attempt: int, content: bytes) -> None:
        """Persist explicit successful cost; ambiguous or failed responses retain all funds."""
        response = _Response.model_validate_json(content)
        cost = response.usage.cost
        if response.error or not response.choices or cost is None:
            return
        if not cost.is_finite() or cost < 0:
            raise ValueError("Invalid provider cost")
        micro = int((cost * 1_000_000).to_integral_value(rounding=ROUND_CEILING))
        model = self.model
        if model is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Campaign model is absent")
        usage = response.usage
        actual_tokens = None
        if usage.prompt_tokens is not None and usage.completion_tokens is not None and usage.total_tokens is not None:
            if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
                raise ValueError("Provider total tokens differ from prompt and completion usage")
            actual_tokens = usage.total_tokens
        with sqlite3.connect(self.database, isolation_level=None, timeout=30) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT model,settled FROM attempts WHERE id=?", (attempt,)).fetchone() != (
                model.name,
                0,
            ):
                raise ValueError("Unknown, foreign or already settled spending reservation")
            reserved_tokens, reserved_charge = connection.execute(
                "SELECT tokens,charge FROM attempts WHERE id=?", (attempt,)
            ).fetchone()
            retained_tokens = reserved_tokens if actual_tokens is None else actual_tokens
            excess = micro > reserved_charge or retained_tokens > reserved_tokens
            connection.execute(
                "UPDATE attempts SET charge=?,tokens=?,settled=1,excess=?,active_until=0 WHERE id=?",
                (micro, retained_tokens, int(excess), attempt),
            )
            connection.commit()
        if excess:
            raise CampaignCapacityStop("Observed provider usage exceeded the reservation; usage retained")


def controlled_http_client(control: CampaignRequestControl) -> httpx2.Client:
    """Build a per-adapter transport boundary without process-wide SDK monkeypatches."""
    from openai import DefaultHttpxClient

    class ControlledClient(DefaultHttpxClient):
        def send(self, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
            if (
                request.url.host != "openrouter.ai"
                or request.url.path != "/api/v1/chat/completions"
                or request.method != "POST"
            ):
                raise ValueError("Unexpected campaign transport destination")
            timeouts = _Timeouts.model_validate(request.extensions.get("timeout", {}))
            values = tuple(value for value in timeouts.model_dump().values() if value is not None)
            from trialagentbench.adapters.http_transport import provider_request_deadline

            started = time.monotonic()
            absolute = provider_request_deadline.get()
            if absolute is not None:
                values += (max(0.0, absolute - started),)
                request.extensions["provider_deadline"] = absolute
            attempt = control.reserve(
                request.content, timeout_seconds=min(values) if values else None, active_transport=True
            )
            try:
                elapsed = time.monotonic() - started
                adjusted = {
                    key: None if value is None else value - elapsed for key, value in timeouts.model_dump().items()
                }
                if any(value is not None and value <= 0 for value in adjusted.values()):
                    raise CampaignCapacityStop(
                        "Request deadline exhausted while awaiting provider capacity", campaign_wide=False
                    )
                request.extensions["timeout"] = adjusted
                kwargs["follow_redirects"] = False
                response = super().send(request, **kwargs)
                if response.is_success:
                    try:
                        control.settle(attempt, response.content)
                    except (ValueError, TypeError, CampaignCapacityStop) as error:
                        try:
                            payload = loads_json(response.text)
                        except ValueError:
                            payload = {"unparsed_response_text": response.text}
                        retained = payload if isinstance(payload, dict) else {"response": payload}
                        if isinstance(error, CampaignCapacityStop):
                            error.raw_response = retained
                            raise
                        raise ProviderAccountingError(
                            redact_provider_text(str(error)), raw_response=retained
                        ) from None
                return response
            finally:
                control.finish_attempt(attempt)

    from trialagentbench.adapters.http_transport import DeadlineTransport

    return ControlledClient(transport=DeadlineTransport())
