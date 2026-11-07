"""Build generic observable trace tables from completed benchmark runs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel

from trialagentbench.contracts.trace.bundle import (
    OBSERVABLE_TRACE_TABLE_MODELS,
    ObservableTraceBundleManifestV2,
    ObservableTraceTableV1,
)
from trialagentbench.contracts.trace.observable import ModelActionTraceEventV1, TraceFeatureRowV1
from trialagentbench.io import sha256_file
from trialagentbench.io.csv_contracts import write_contract_csv
from trialagentbench.tools.validate.validate_trace_analysis_bundle import (
    MANIFEST_NAME,
    validate_trace_analysis_bundle,
)
from trialagentbench.trialdev.action_trace import (
    collect_trialdev_action_trace,
    discover_trialdev_run_dirs,
    trialdev_decision_rows,
)
from trialagentbench.trialeval.action_trace import collect_trialeval_action_trace, discover_trialeval_run_dirs


def _portable_trace_events(events: Iterable[ModelActionTraceEventV1], root: Path) -> Iterator[ModelActionTraceEventV1]:
    """Keep provenance relative to the supplied run root when exporting traces."""
    for event in events:
        updates = {}
        for field in ("source_path", "source_artifact_path"):
            path = Path(getattr(event, field))
            # Historical runtime locators remain unchanged; only paths into
            # the supplied copy are projected relative to that copy.
            if path.is_absolute() and path.is_relative_to(root.resolve()):
                updates[field] = path.relative_to(root.resolve()).as_posix()
        yield event.model_copy(update=updates)


class _TraceRowStore:
    """Disk-backed deterministic row accumulator for one bundle build."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("CREATE TABLE trace_rows (table_name TEXT NOT NULL, payload TEXT NOT NULL)")

    def add(self, table_name: str, rows: Iterable[BaseModel]) -> None:
        """Validate and persist one bounded batch of table rows."""

        model = OBSERVABLE_TRACE_TABLE_MODELS[table_name]
        payloads = (
            (
                table_name,
                json.dumps(
                    model.model_validate(row.model_dump(mode="json")).model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for row in rows
        )
        self._connection.executemany(
            "INSERT INTO trace_rows (table_name, payload) VALUES (?, ?)",
            payloads,
        )
        self._connection.commit()

    def rows(self, table_name: str) -> Iterator[BaseModel]:
        """Yield stable task order with numeric event/decision sequence within each task."""

        model = OBSERVABLE_TRACE_TABLE_MODELS[table_name]
        cursor = self._connection.execute(
            "SELECT payload FROM trace_rows WHERE table_name = ? ORDER BY "
            "json_extract(payload,'$.model_id'),json_extract(payload,'$.run_id'),"
            "COALESCE(json_extract(payload,'$.task_id'),json_extract(payload,'$.program_id')),"
            "json_extract(payload,'$.decision_index'),json_extract(payload,'$.event_index'),payload",
            (table_name,),
        )
        for (payload,) in cursor:
            yield model.model_validate_json(payload)

    def count(self, table_name: str) -> int:
        """Return the persisted row count for one table."""

        result = self._connection.execute(
            "SELECT COUNT(*) FROM trace_rows WHERE table_name = ?",
            (table_name,),
        ).fetchone()
        if result is None:
            raise RuntimeError(f"trace row count query returned no result: {table_name}")
        return int(result[0])

    def close(self) -> None:
        """Close the temporary row store."""

        self._connection.close()


def _record_feature_identities(
    features: Iterable[TraceFeatureRowV1],
    *,
    benchmark_suites: set[str],
    model_ids: set[str],
    run_ids: set[str],
) -> int:
    count = 0
    for feature in features:
        benchmark_suites.add(feature.benchmark)
        model_ids.add(feature.model_id)
        run_ids.add(feature.run_id)
        count += 1
    return count


def build_trace_analysis_bundle(
    *,
    out_dir: Path,
    trialeval_root: Path | None = None,
    trialdev_root: Path | None = None,
    trialdev_release_root: Path | None = None,
) -> Path:
    """Build and validate a generic observable trace bundle.

    Parameters
    ----------
    out_dir
        New destination directory for the exact bundle members.
    trialeval_root
        Optional root containing completed TrialEval runs.
    trialdev_root
        Optional root containing completed TrialDev runs.
    trialdev_release_root
        Optional participant release root used to classify TrialDev evidence
        paths. It is never copied into the output.

    Returns
    -------
    Path
        The validated bundle directory.

    Raises
    ------
    FileExistsError
        If ``out_dir`` already exists.
    FileNotFoundError
        If an input root is absent.
    ValueError
        If no suite is requested or a requested root contains no completed run.
    """

    if trialeval_root is None and trialdev_root is None:
        raise ValueError("at least one TrialEval or TrialDev run root is required")
    destination = Path(out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite observable trace bundle: {destination}")
    if trialdev_release_root is not None and not Path(trialdev_release_root).is_dir():
        raise FileNotFoundError(f"TrialDev participant release does not exist: {trialdev_release_root}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    benchmark_suites: set[str] = set()
    model_ids: set[str] = set()
    run_ids: set[str] = set()
    feature_count = 0
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.build-", dir=destination.parent) as temporary:
        work_root = Path(temporary)
        bundle_root = work_root / "bundle"
        bundle_root.mkdir()
        store = _TraceRowStore(work_root / "trace_rows.sqlite3")
        try:
            if trialeval_root is not None:
                root = Path(trialeval_root)
                if not root.is_dir():
                    raise FileNotFoundError(f"TrialEval run root does not exist: {root}")
                trialeval_runs = discover_trialeval_run_dirs(root)
                if not trialeval_runs:
                    raise ValueError(f"TrialEval run root contains no completed runs: {root}")
                for run_dir in trialeval_runs:
                    events, features, evidence, cascades = collect_trialeval_action_trace([run_dir])
                    feature_count += _record_feature_identities(
                        features,
                        benchmark_suites=benchmark_suites,
                        model_ids=model_ids,
                        run_ids=run_ids,
                    )
                    for table_name, trialeval_rows in (
                        ("action_events.csv", _portable_trace_events(events, root)),
                        ("unit_features.csv", features),
                        ("evidence_use.csv", evidence),
                    ):
                        store.add(table_name, trialeval_rows)

            if trialdev_root is not None:
                root = Path(trialdev_root)
                if not root.is_dir():
                    raise FileNotFoundError(f"TrialDev run root does not exist: {root}")
                trialdev_runs = discover_trialdev_run_dirs(root)
                if not trialdev_runs:
                    raise ValueError(f"TrialDev run root contains no completed runs: {root}")
                for run_dir in trialdev_runs:
                    events, features, evidence = collect_trialdev_action_trace(
                        [run_dir],
                        trialdev_release_root=trialdev_release_root,
                    )
                    feature_count += _record_feature_identities(
                        features,
                        benchmark_suites=benchmark_suites,
                        model_ids=model_ids,
                        run_ids=run_ids,
                    )
                    for table_name, trialdev_rows in (
                        ("action_events.csv", _portable_trace_events(events, root)),
                        ("unit_features.csv", features),
                        ("evidence_use.csv", evidence),
                        ("trialdev_decisions.csv", trialdev_decision_rows(run_dir, events)),
                    ):
                        store.add(table_name, trialdev_rows)

            if feature_count == 0:
                raise ValueError("requested runs produced no observable benchmark units")
            tables: list[ObservableTraceTableV1] = []
            for name, model in OBSERVABLE_TRACE_TABLE_MODELS.items():
                path = bundle_root / name
                write_contract_csv(path, store.rows(name), model=model)
                tables.append(
                    ObservableTraceTableV1(
                        path=name,
                        row_schema_id=str(model.model_fields["schema_id"].default),
                        row_count=store.count(name),
                        sha256=sha256_file(path),
                    )
                )
        finally:
            store.close()

        manifest = ObservableTraceBundleManifestV2(
            benchmark_suites=tuple(sorted(benchmark_suites)),
            model_ids=tuple(sorted(model_ids)),
            run_ids=tuple(sorted(run_ids)),
            tables=tuple(sorted(tables, key=lambda table: table.path)),
        )
        (bundle_root / MANIFEST_NAME).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_trace_analysis_bundle(bundle_root)
        bundle_root.rename(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    """Build one generic observable trace bundle from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--trialeval-root", type=Path)
    parser.add_argument("--trialdev-root", type=Path)
    parser.add_argument("--trialdev-release-root", type=Path)
    args = parser.parse_args(argv)
    output = build_trace_analysis_bundle(
        out_dir=args.out_dir,
        trialeval_root=args.trialeval_root,
        trialdev_root=args.trialdev_root,
        trialdev_release_root=args.trialdev_release_root,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_trace_analysis_bundle", "main"]
