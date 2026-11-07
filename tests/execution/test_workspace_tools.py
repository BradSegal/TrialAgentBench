"""Contract tests for the persistent agent text workspace."""

from __future__ import annotations

from pathlib import Path

import pytest
from trialagentbench_verifier.executor import DockerCodeSession

from trialagentbench.ports import CodeExecutionResultV1
from trialagentbench.ports.tool_input import ToolInputError
from trialagentbench.tools.workspace import (
    WORKSPACE_TOOLS,
    handle_workspace_tool,
    read_workspace_submission_text,
)


class _Session:
    def execute(self, code: str) -> str:
        return code

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(status="success", output=code, elapsed_seconds=0.1)

    def close(self) -> None:
        return None


def test_workspace_tool_schemas_are_closed_and_bounded() -> None:
    by_name = {tool["function"]["name"]: tool for tool in WORKSPACE_TOOLS}

    assert set(by_name) == {"write_workspace_file", "read_workspace_file", "list_workspace_files"}
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in by_name.values())
    read_properties = by_name["read_workspace_file"]["function"]["parameters"]["properties"]
    assert read_properties["start_line"]["minimum"] == 1
    assert read_properties["end_line"]["minimum"] == 1
    assert read_properties["end_line"]["type"] == ["integer", "null"]


def _schema_nodes(schema: dict) -> list[dict]:
    """Return a parameter schema and every nested property and item schema."""
    nodes = [schema]
    for child in (*schema.get("properties", {}).values(), *([schema["items"]] if "items" in schema else [])):
        nodes.extend(_schema_nodes(child))
    return nodes


@pytest.mark.parametrize("mode", ["python_r", "python", "r"])
def test_every_tool_offered_to_a_provider_types_every_schema_node(mode: str) -> None:
    import json

    from trialagentbench.tools.analysis import analysis_tools
    from trialagentbench.trialdev.actions import trialdev_programme_tools
    from trialagentbench.trialdev.task_agent import ANALYSIS_TOOLS as TRIALDEV_TOOLS
    from trialagentbench.trialeval.agent import _get_submission_tools, _get_tools

    tools = [
        *_get_tools(mode),
        *_get_submission_tools(),
        *analysis_tools(mode),
        *TRIALDEV_TOOLS,
        *trialdev_programme_tools(),
    ]
    for tool in tools:
        parameters = tool["function"]["parameters"]
        text = json.dumps(parameters)
        assert not {"$ref", "$defs", "anyOf", "oneOf", "allOf"} & {
            key for node in _schema_nodes(parameters) for key in node
        }, text
        for node in _schema_nodes(parameters):
            assert "type" in node, (tool["function"]["name"], node)
            if node["type"] == "array":
                assert "type" in node["items"], (tool["function"]["name"], node)


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "../truth.json",
        "analysis/../../truth.json",
        Path("/", "tmp", "analysis.py").as_posix(),
        "windows\\analysis.py",
        "line\nbreak.py",
        "scratch",
        "scratch/../truth.json",
        "x" * 241,
    ],
)
def test_workspace_rejects_paths_outside_its_posix_boundary(path: str) -> None:
    with pytest.raises(ToolInputError):
        handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": path, "content": "print('safe')"},
            session=_Session(),
        )


def test_workspace_rejects_invalid_content_and_read_ranges() -> None:
    with pytest.raises(ToolInputError, match="Invalid write_workspace_file arguments"):
        handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": "analysis.py", "content": 1},
            session=_Session(),
        )


@pytest.mark.parametrize(
    ("name", "arguments"),
    (
        ("write_workspace_file", {"path": "analysis.py", "content": "pass", "unexpected": 1}),
        ("read_workspace_file", {"path": "analysis.py", "unexpected": 1}),
        ("list_workspace_files", {"unexpected": 1}),
    ),
)
def test_workspace_runtime_rejects_arguments_excluded_by_its_schema(
    name: str,
    arguments: dict[str, object],
) -> None:
    """Keep provider-facing JSON schemas and local execution exact."""

    with pytest.raises(ToolInputError, match=f"Invalid {name} arguments"):
        handle_workspace_tool(name=name, arguments=arguments, session=_Session())
    with pytest.raises(ToolInputError, match="greater than or equal"):
        handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "analysis.py", "start_line": 5, "end_line": 4},
            session=_Session(),
        )
    with pytest.raises(ToolInputError, match="at most 400 lines"):
        handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "analysis.py", "start_line": 1, "end_line": 401},
            session=_Session(),
        )


@pytest.mark.executor
def test_workspace_persists_files_for_review_and_code_execution(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path)
    try:
        written = handle_workspace_tool(
            name="write_workspace_file",
            arguments={"path": "analysis/main.py", "content": "estimate = 6 * 7\nprint(estimate)\n"},
            session=session,
        )
        listed = handle_workspace_tool(name="list_workspace_files", arguments={}, session=session)
        read = handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "scratch/analysis/main.py", "start_line": 1, "end_line": 2},
            session=session,
        )
        executed = session.execute_result("exec(open('scratch/analysis/main.py').read())")
    finally:
        session.close()

    assert written.status == "success"
    assert written.output == "Wrote scratch/analysis/main.py (33 characters)."
    assert "scratch/analysis/main.py" in listed.output  # listed paths are the paths the other tools accept
    assert read.output == "0001: estimate = 6 * 7\n0002: print(estimate)"
    assert executed.output == "42"


@pytest.mark.executor
def test_workspace_reads_supplied_task_materials(tmp_path: Path) -> None:
    """Listed participant materials are readable without copying them to scratch."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "protocol.md").write_text("# Protocol\nEstimand: risk difference\n", encoding="utf-8")
    session = DockerCodeSession(cwd=tmp_path)
    try:
        read = handle_workspace_tool(
            name="read_workspace_file",
            arguments={"path": "documents/protocol.md", "start_line": 1, "end_line": 2},
            session=session,
        )
        missing = handle_workspace_tool(name="read_workspace_file", arguments={"path": "protocol.md"}, session=session)
    finally:
        session.close()

    assert read.output == "0001: # Protocol\n0002: Estimand: risk difference"
    # A wrong path names the files with that name instead of a misleading scratch/ path.
    assert "No file at protocol.md; files with that name: documents/protocol.md" in missing.output


@pytest.mark.executor
def test_submission_file_boundary_reads_regular_utf8_and_rejects_symlink(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path)
    try:
        written = session.execute_result(
            "from pathlib import Path\n"
            "Path('scratch/submission.json').write_text('{\"task_id\":\"task-1\"}', encoding='utf-8')"
        )
        text = read_workspace_submission_text(path="submission.json", session=session)
        prefixed_text = read_workspace_submission_text(path="scratch/submission.json", session=session)
        linked = session.execute_result(
            "from pathlib import Path\nPath('scratch/submission-link.json').symlink_to('submission.json')"
        )
        with pytest.raises(ToolInputError, match="session has terminated"):
            read_workspace_submission_text(path="submission-link.json", session=session)
        session.close()
    finally:
        session.close()

    assert written.status == "success"
    assert linked.status == "session_terminated"
    assert text == '{"task_id":"task-1"}'
    assert prefixed_text == text


@pytest.mark.executor
def test_executor_supports_propensity_model_workflow(tmp_path: Path) -> None:
    session = DockerCodeSession(cwd=tmp_path)
    try:
        result = session.execute_result("""
import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[0.0], [0.2], [0.8], [1.0]])
treatment = np.array([0, 0, 1, 1])
model = LogisticRegression(random_state=0).fit(x, treatment)
propensity = model.predict_proba(x)[:, 1]
print(model.n_features_in_, len(propensity), bool(np.all((propensity > 0) & (propensity < 1))))
""")
    finally:
        session.close()

    assert result.status == "success"
    assert result.output == "1 4 True"


def test_provider_tool_rejects_a_union_that_cannot_be_one_typed_node() -> None:
    from pydantic import BaseModel

    from trialagentbench.ports.tool_input import provider_tool

    class Ambiguous(BaseModel):
        value: int | str

    with pytest.raises(ValueError, match="one type"):
        provider_tool("ambiguous", "Ambiguous argument.", Ambiguous)


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("programme", "4fdaa83cc310f6d895ec0c84ece7e9dd830a8fdf1d262298fd8bbe11571457c3"),
        ("terminal", "52ebb1d2b8e6ac67b80ed2cac027fba9e165ea633319f9b6ec4cd9497c39992b"),
    ],
)
def test_programme_tool_json_preserves_the_frozen_agent_surface(surface: str, expected: str) -> None:
    """Moving descriptions into schema code preserves the exact provider JSON bytes."""
    import hashlib
    import json

    from trialagentbench.trialdev.actions import trialdev_programme_tools, trialdev_terminal_tools

    tools = trialdev_programme_tools() if surface == "programme" else trialdev_terminal_tools()
    assert hashlib.sha256(json.dumps(tools, ensure_ascii=True).encode()).hexdigest() == expected


@pytest.fixture
def package_definition(monkeypatch, tmp_path):
    """Install synthetic definitions in an imported package module for census tests."""
    import sys
    from types import ModuleType

    import trialagentbench

    locations = [*trialagentbench.__path__, str(tmp_path)]
    monkeypatch.setattr(trialagentbench, "__path__", locations)
    monkeypatch.setattr(trialagentbench.__spec__, "submodule_search_locations", locations)
    module = ModuleType("trialagentbench._description_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    definitions = {}

    def register(definition):
        monkeypatch.setattr(definition, "__module__", module.__name__)
        setattr(module, definition.__name__, definition)
        definitions[definition.__name__] = definition
        (tmp_path / "_description_test.py").write_text(
            "\n".join(f"class {name}:\n    {value.__doc__!r}\n" for name, value in definitions.items()),
            encoding="utf-8",
        )
        return definition

    return register


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_provider_tool_refuses_internal_docstrings_as_schema_descriptions(
    nested: bool, explicit: bool, package_definition
) -> None:
    """A reachable model cannot silently send an unhashed internal docstring to the provider."""
    from pydantic import BaseModel, ConfigDict, create_model

    from trialagentbench.ports.tool_input import provider_tool

    class Documented(BaseModel):
        """Internal validation documentation."""

        value: int

    package_definition(Documented)
    if explicit:
        Documented.model_config = ConfigDict(json_schema_extra={"description": Documented.__doc__})
    arguments = create_model("OuterArguments", child=(Documented, ...)) if nested else Documented
    with pytest.raises(ValueError, match="descriptions must be explicit schema code"):
        provider_tool("sample", "Explicit tool instruction.", arguments)


@pytest.mark.parametrize("kind", ["enum", "dataclass", "typed_dict"])
def test_provider_tool_refuses_package_docstrings_from_every_class_kind(kind, monkeypatch, package_definition) -> None:
    """Nested schema descriptions cannot inherit class documentation, including after an edit."""
    from enum import Enum

    from pydantic import Field, create_model
    from pydantic.dataclasses import dataclass
    from typing_extensions import TypedDict

    from trialagentbench.ports.tool_input import provider_tool

    class Choice(str, Enum):
        """Internal choice documentation."""

        FIRST = "first"

    @dataclass
    class Record:
        """Internal record documentation."""

        value: int

    class Mapping(TypedDict):
        """Internal mapping documentation."""

        value: int

    document_type = package_definition({"enum": Choice, "dataclass": Record, "typed_dict": Mapping}[kind])
    for docstring in (document_type.__doc__, "Revised internal schema documentation."):
        monkeypatch.setattr(document_type, "__doc__", docstring)
        package_definition(document_type)
        arguments = create_model("DocumentedArguments", child=(document_type, ...))
        with pytest.raises(ValueError, match="descriptions must be explicit schema code"):
            provider_tool("sample", "Explicit tool instruction.", arguments)
        explicit = create_model("ExplicitArguments", child=(document_type, Field(description="Explicit parameter.")))
        assert provider_tool("sample", "Explicit tool instruction.", explicit)


def test_dose_documentation_does_not_change_programme_or_terminal_tools(monkeypatch) -> None:
    """Both dose parameters keep their explicit description when internal Enum documentation changes."""
    import json

    from trialagentbench.contracts.trialdev.release import TrialDevDose
    from trialagentbench.trialdev.actions import trialdev_programme_tools, trialdev_terminal_tools

    before = json.dumps([trialdev_programme_tools(), trialdev_terminal_tools()])
    monkeypatch.setattr(TrialDevDose, "__doc__", "Revised internal dose documentation.")
    assert json.dumps([trialdev_programme_tools(), trialdev_terminal_tools()]) == before


def test_provider_tool_refuses_custom_annotation_documentation(monkeypatch, package_definition) -> None:
    """A custom schema hook cannot hide its annotation class from the description guard."""
    from pydantic import Field, create_model

    from trialagentbench.ports.tool_input import provider_tool

    class CustomAnnotation:
        """Internal custom annotation documentation."""

        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            return handler.generate_schema(str)

        @classmethod
        def __get_pydantic_json_schema__(cls, schema, handler):
            return {"type": "string", "description": cls.__doc__}

    package_definition(CustomAnnotation)
    for docstring in (CustomAnnotation.__doc__, "Revised internal annotation documentation."):
        monkeypatch.setattr(CustomAnnotation, "__doc__", docstring)
        package_definition(CustomAnnotation)
        arguments = create_model("CustomArguments", value=(CustomAnnotation, ...))
        with pytest.raises(ValueError, match="descriptions must be explicit schema code"):
            provider_tool("sample", "Explicit tool instruction.", arguments)
        explicit = create_model(
            "ExplicitCustomArguments", value=(CustomAnnotation, Field(description="Explicit parameter."))
        )
        assert provider_tool("sample", "Explicit tool instruction.", explicit)


@pytest.mark.parametrize(
    "route",
    [
        "annotated_instance",
        "with_json_schema",
        "field_callable",
        "field_dict",
        "field_description",
        "function",
        "model_callable",
    ],
)
def test_provider_tool_refuses_documentation_from_schema_metadata(route, package_definition):
    """Final rendered text is checked independently of the schema's construction route."""
    from typing import Annotated

    from pydantic import ConfigDict, Field, WithJsonSchema, create_model

    from trialagentbench.ports.tool_input import provider_tool

    @package_definition
    class Marker:
        """Internal metadata documentation, distinct from public instructions."""

        def __get_pydantic_json_schema__(self, schema, handler):
            return {**handler(schema), "description": type(self).__doc__}

    @package_definition
    def validator(value):
        """Internal validation function documentation, distinct from public instructions."""
        return value

    # These reproduce custom hooks, annotation metadata and field/model extras without
    # relying on whether their source class is present in Pydantic's core schema.
    annotation = str
    field = Field()
    config = ConfigDict()
    if route == "annotated_instance":
        annotation = Annotated[str, Marker()]
    elif route == "with_json_schema":
        annotation = Annotated[str, WithJsonSchema({"type": "string", "description": Marker.__doc__})]
    elif route == "field_callable":
        field = Field(json_schema_extra=lambda schema: schema.update(description=Marker.__doc__))
    elif route == "field_dict":
        field = Field(json_schema_extra={"description": Marker.__doc__})
    elif route == "field_description":
        field = Field(description=Marker.__doc__)
    elif route == "function":
        field = Field(description=validator.__doc__)
    else:
        config = ConfigDict(json_schema_extra=lambda schema: schema.update(description=Marker.__doc__))
    arguments = create_model("MetadataArguments", __config__=config, value=(annotation, field))
    with pytest.raises(ValueError, match="descriptions must be explicit schema code"):
        provider_tool("sample", "Explicit tool instruction.", arguments)


@pytest.mark.parametrize("package", ["trialagentbench", "trialagentbench_verifier"])
@pytest.mark.parametrize("kind", ["class", "function", "method", "property"])
def test_provider_text_census_covers_both_packages_and_nested_definitions(package, kind, monkeypatch, tmp_path):
    """All imported package definitions participate, even when absent from the tool graph."""
    import importlib
    import sys
    from types import ModuleType

    from trialagentbench.ports.tool_input import validate_provider_text_v1

    package_module = importlib.import_module(package)
    locations = [*package_module.__path__, str(tmp_path)]
    monkeypatch.setattr(package_module, "__path__", locations)
    monkeypatch.setattr(package_module.__spec__, "submodule_search_locations", locations)
    module = ModuleType(package + "._description_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = (
        "class Internal:\n"
        '    """Class documentation with   irregular whitespace."""\n'
        "    def method(self):\n"
        '        """Method documentation on another line."""\n'
        "    @property\n"
        "    def value(self):\n"
        '        """Property accessor documentation for internal use."""\n'
        "def function():\n"
        '    """Function documentation with additional context."""\n'
    )
    (tmp_path / "_description_test.py").write_text(source, encoding="utf-8")
    exec(source, module.__dict__)
    docs = {
        "class": module.Internal.__doc__,
        "function": module.function.__doc__,
        "method": module.Internal.method.__doc__,
        "property": module.Internal.value.fget.__doc__,
    }
    text = "Prefix: " + "\n  ".join(docs[kind].split()) + " Suffix."
    for messages, tools in (
        ([{"role": "system", "content": text}], None),
        ([{"role": "user", "content": [{"type": "text", "text": text}]}], None),
        ([], [{"function": {"parameters": {"items": {"description": text}}}}]),
    ):
        with pytest.raises(ValueError, match="internal docstrings"):
            validate_provider_text_v1(messages=messages, tools=tools)
    # Changing the documentation after a previous census must not bypass the next request.
    monkeypatch.setattr(module.function, "__doc__", "New internal function documentation after import.")
    (tmp_path / "_description_test.py").write_text(
        source.replace("Function documentation with additional context.", module.function.__doc__), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="internal docstrings"):
        validate_provider_text_v1(messages=[{"role": "system", "content": module.function.__doc__}], tools=None)


@pytest.mark.parametrize("kind", ["model", "enum", "dynamic"])
def test_provider_tool_refuses_local_automatic_documentation_without_global_registration(kind):
    """Function-local and dynamically created schemas retain the automatic-description boundary."""
    from enum import Enum

    from pydantic import BaseModel, create_model

    from trialagentbench.ports.tool_input import HarnessConfigurationError, provider_tool

    class LocalModel(BaseModel):
        """Internal local argument documentation."""

        value: int

    class LocalEnum(str, Enum):
        """Internal local enum documentation."""

        ONE = "one"

    arguments = (
        LocalModel
        if kind == "model"
        else (
            create_model("LocalArguments", value=(LocalEnum, ...))
            if kind == "enum"
            else create_model("DynamicArguments", __doc__="Internal dynamic argument documentation.", value=(int, ...))
        )
    )
    with pytest.raises(HarnessConfigurationError):
        provider_tool("local", "Explicit tool instruction.", arguments)


def test_documentation_census_is_complete_before_cli_import():
    """The source-module census does not change when another entry point is imported."""
    import importlib

    from trialagentbench.ports.tool_input import _package_docstrings

    before = _package_docstrings()
    assert "Run TrialDev." in before
    importlib.import_module("trialagentbench.cli")
    assert _package_docstrings() == before
