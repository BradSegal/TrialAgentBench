"""Observed standards transfers retain column and parameter metadata."""

from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import pandas as pd
import pyarrow.parquet as pq
import pytest

from trialagentbench.clinical_tables import StudyBaselineFieldV1, write_observed_study_tables


@pytest.mark.parametrize("endpoint_kind", ("binary", "continuous"))
def test_observed_archive_has_complete_columns_and_separate_parameter_mappings(
    tmp_path: Path, endpoint_kind: Literal["binary", "continuous"]
) -> None:
    participants = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3"],
            "assigned_arm": ["active", "control", "active"],
            "sex": ["F", "M", "F"],
            "baseline_value": [2.5, 3.0, 4.0],
        }
    )
    analysis = participants.assign(response=[1.0, 0.0, float("nan")])
    path = tmp_path / "standards.zip"
    label = "Study-specific clinical response at the final scheduled assessment"
    write_observed_study_tables(
        path,
        study_id="study",
        participants=participants,
        analysis=analysis,
        endpoint_id="response",
        endpoint_label=label,
        endpoint_object="Clinical response",
        endpoint_unit="indicator",
        endpoint_kind=endpoint_kind,
        assessment_time=17.5,
        baseline_fields=(
            StudyBaselineFieldV1(field_id="baseline_value", label="Recorded baseline concentration", unit="mg/L"),
        ),
    )
    with ZipFile(path) as archive:
        metadata = pd.read_csv(BytesIO(archive.read("variables.csv")), keep_default_na=False)
        for name in archive.namelist():
            if not name.endswith(".parquet"):
                continue
            frame = pd.read_parquet(BytesIO(archive.read(name)))
            defined = metadata.loc[metadata.dataset.eq(Path(name).stem)]
            assert set(defined.variable) == set(frame.columns)
            assert defined.storage_type.ne("").all()
            assert defined.label.ne("").all()
            assert defined.missing_value.ne("").all()
            columns = defined.loc[defined.entry_kind.eq("column")].set_index("variable")
            assert columns.storage_type.to_dict() == {
                field.name: str(field.type) for field in pq.read_schema(BytesIO(archive.read(name)))
            }
            if "USUBJID" in columns.index:
                assert columns.loc["USUBJID", "controlled_terminology"] == "not applicable"
        baseline = metadata.loc[metadata.source_field.eq("baseline_value")].iloc[0]
        assert baseline.unit == "mg/L"
        assert baseline.label == "Recorded baseline concentration"
        assert not metadata.loc[metadata.variable.str.startswith("BASE")].source_field.eq("sex").any()
        mapping = metadata.loc[metadata.parameter.eq("PRIMARY")].iloc[0]
        assert mapping.variable == "AVAL" and mapping.source_field == "response"
        bds = pd.read_parquet(BytesIO(archive.read("ADBDS.parquet")))
        assert bds.ELAPDAY.eq(17.5).all()
        assert mapping.label == label
        assert bds.PARAM.eq(label[:40]).all()
        domain = "FA" if endpoint_kind == "binary" else "QS"
        sdtm = pd.read_parquet(BytesIO(archive.read(f"{domain}.parquet")))
        assert sdtm[f"{domain}TEST"].eq(label[:40]).all()
        assert bds.AVAL.isna().sum() == 1
