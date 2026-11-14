"""Versioned executable content identity bound to logical owners and import routing."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

_NAMESPACE = "trialagentbench_verifier"
_MODULE = re.compile(r"trialagentbench_verifier(?:\.[A-Za-z_]\w*)*")


def _syntax(node: object) -> object:
    if isinstance(node, ast.AST):
        return [
            type(node).__name__,
            [
                [name, _syntax(value)]
                for name, value in ast.iter_fields(node)
                # Python3.12 adds this empty field to the shared3.11 grammar.
                if name != "type_params" or value
            ],
        ]
    if isinstance(node, list):
        return [_syntax(item) for item in node]
    return repr(node)


def _tree(source: str) -> ast.Module:
    tree = ast.parse(source, feature_version=(3, 11))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return tree


def _module(root: Path, path: Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    return ".".join((_NAMESPACE, *(parts[:-1] if parts[-1] == "__init__" else parts)))


class _LogicalImports(ast.NodeTransformer):
    def __init__(self, routes: dict[str, str], module: str, is_package: bool) -> None:
        self.routes = routes
        self.package = module if is_package else module.rpartition(".")[0]

    def _route(self, value: str) -> str:
        for prefix in sorted(self.routes, key=len, reverse=True):
            if value == prefix or value.startswith(prefix + "."):
                return self.routes[prefix] + value[len(prefix) :]
        return value

    def visit_ImportFrom(self, node: ast.ImportFrom) -> list[ast.ImportFrom]:
        base = node.module or ""
        if node.level:
            base = importlib.util.resolve_name("." * node.level + base, self.package)
        groups: dict[str, list[ast.alias]] = {}
        for item in node.names:
            child = base + "." + item.name
            if child in self.routes:
                owner, _, name = self.routes[child].rpartition(".")
                alias = ast.alias(name=name, asname=item.asname or item.name)
            else:
                owner, alias = self._route(base), item
            groups.setdefault(owner, []).append(alias)
        return [ast.ImportFrom(module=owner, names=names, level=0) for owner, names in groups.items()]

    def visit_Import(self, node: ast.Import) -> ast.Import:
        return ast.Import(names=[ast.alias(name=self._route(item.name), asname=item.asname) for item in node.names])

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        # Lazy public exports use exact module strings. Data strings and physical
        # filesystem locators are otherwise retained as executable content.
        if isinstance(node.value, str) and node.value in self.routes:
            return ast.Constant(value=self.routes[node.value])
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        names = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            names.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            names.append(value.id)
            old = ".".join(reversed(names))
            new = self._route(old)
            if new != old:
                return ast.parse(new, mode="eval").body
        return self.generic_visit(node)


def _transparent_binding(tree: ast.Module, target: str) -> bool:
    if len(tree.body) != 5 or not isinstance(tree.body[3], ast.If):
        return False
    guard = tree.body[3]
    if ast.dump(guard.test) != ast.dump(ast.Name(id="TYPE_CHECKING", ctx=ast.Load())) or guard.orelse:
        return False
    if not guard.body or not all(isinstance(item, ast.ImportFrom) and item.module == target for item in guard.body):
        return False
    exports = [alias for item in guard.body if isinstance(item, ast.ImportFrom) for alias in item.names]
    if not exports or any(item.name != "*" and item.asname != item.name for item in exports):
        return False
    owner, _, name = target.rpartition(".")
    expected = ast.parse(
        "import sys as _compat_sys\n"
        "from typing import TYPE_CHECKING\n"
        f"from {owner} import {name} as _compat_module\n"
        "if TYPE_CHECKING:\n    pass\n"
        "_compat_sys.modules[__name__] = _compat_module\n"
    )
    expected.body[3] = guard
    return ast.dump(tree) == ast.dump(expected)


def source_census_sha256(root: Path) -> str:
    """Hash logical executable units and published bindings independently of their paths.

    Parameters
    ----------
    root
        Installed verifier package root containing ``identity/modules.json``.

    Returns
    -------
    str
        SHA-256 over the v2 format, sorted logical module IDs, normalized ASTs
        and published alias routing. Comments and docstrings are excluded;
        executable literals and import destinations are included.

    Raises
    ------
    ValueError
        If the module map omits executable source, duplicates owners, crosses
        its package boundary or declares a nontransparent compatibility module.
    OSError
        If the installed source or its module map cannot be read.
    """
    record = json.loads((root / "identity/modules.json").read_text())
    if set(record) != {"units", "bindings"}:
        raise ValueError("Invalid scorer module map")
    units, bindings = record["units"], record["bindings"]
    if not all(
        isinstance(mapping, dict)
        and all(
            isinstance(k, str) and isinstance(v, str) and _MODULE.fullmatch(k) and _MODULE.fullmatch(v)
            for k, v in mapping.items()
        )
        for mapping in (units, bindings)
    ):
        raise ValueError("Scorer module owners must be package-local names")
    if len(set(units.values())) != len(units) or any(owner not in units for owner in bindings.values()):
        raise ValueError("Scorer module ownership is ambiguous")
    sources: dict[str, ast.Module] = {}
    packages: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if path.relative_to(root).parts[0] == "resources":
            continue
        if path.is_symlink():
            raise ValueError(f"Scorer source must not be a symbolic link: {path}")
        name = _module(root, path)
        if name in sources:
            raise ValueError(f"Conflicting package and module: {name}")
        sources[name] = _tree(path.read_text(encoding="utf-8"))
        if path.name == "__init__.py":
            packages.add(name)
    routes = {physical: logical for logical, physical in units.items()}
    for alias, owner in bindings.items():
        if alias in routes and routes[alias] != owner:
            raise ValueError(f"Published scorer binding conflicts with canonical owner: {alias}")
        routes[alias] = owner
        if alias != units[owner] and (alias not in sources or not _transparent_binding(sources[alias], units[owner])):
            raise ValueError(f"Nontransparent scorer compatibility binding: {alias}")
    if any(name not in routes and tree.body for name, tree in sources.items()) or not set(routes) <= sources.keys():
        raise ValueError("Scorer module map does not cover the installed executable source")
    digest = hashlib.sha256(b"trialagentbench-logical-python-ast-v2\0")
    for logical, physical in sorted(units.items()):
        tree = _LogicalImports(routes, physical, physical in packages).visit(sources[physical])
        content = json.dumps(_syntax(tree), ensure_ascii=True, separators=(",", ":")).encode()
        digest.update(logical.encode() + b"\0" + content + b"\0")
    digest.update(json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()
