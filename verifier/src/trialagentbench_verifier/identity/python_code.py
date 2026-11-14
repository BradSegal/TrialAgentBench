"""The historical Python3.11/3.12 executable-span serialization contract."""

from __future__ import annotations

import ast
import io
import tokenize

SOURCE_CODE_FORMAT = b"trialagentbench-python-code-spans-v1\0"


def python_code_bytes_v1(source: bytes) -> bytes:
    r"""Serialize executable syntax independently of comments and docstrings.

    Parameters
    ----------
    source
        Python source bytes, decoded according to their source encoding.

    Returns
    -------
    bytes
        Format prefix followed by UTF-8 source with leading module, class,
        function and async-function docstring spans and comment tokens removed.
        Blank lines and trailing whitespace outside surviving string literals
        are removed, with indentation retained and one final newline. Syntax
        and spans must be shared by Python 3.11 and 3.12.

    Raises
    ------
    ValueError
        If the source uses syntax not shared by Python 3.11 and 3.12, including
        version-dependent comment or docstring spans in PEP 701 f-strings.

    Examples
    --------
    >>> documented = b'"Description."\nanswer = 1 # Comment.\n'
    >>> python_code_bytes_v1(documented) == python_code_bytes_v1(b"answer = 1\n")
    True
    """
    encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
    text = source.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = ast.parse(text, mode="exec", type_comments=False, feature_version=(3, 11))
    except SyntaxError as error:
        raise ValueError(
            "Source identity requires syntax shared by Python 3.11 and 3.12; "
            "version-dependent constructs such as PEP 701 f-string comments are unsupported"
        ) from error
    lines = text.split("\n")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)

    def offset(row: int, column: int, *, utf8: bool = False) -> int:
        if utf8:
            column = len(lines[row - 1].encode("utf-8")[:column].decode("utf-8"))
        return offsets[row - 1] + column

    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                doc = node.body[0]
                assert doc.end_lineno is not None and doc.end_col_offset is not None
                spans.append(
                    (
                        offset(doc.lineno, doc.col_offset, utf8=True),
                        offset(doc.end_lineno, doc.end_col_offset, utf8=True),
                    )
                )
    # Literal whitespace is executable output, including blank lines in multi-line strings.
    doc_spans = tuple(spans)
    literal_lines: set[int] = set()
    fstrings: list[tokenize.TokenInfo] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        kind = tokenize.tok_name[token.type]
        if kind == "FSTRING_START":
            fstrings.append(token)
        elif kind == "FSTRING_END":
            start_token = fstrings.pop()
            literal_lines.update(range(start_token.start[0] - 1, token.end[0] - 1))
        elif token.type == tokenize.STRING:
            start = offset(*token.start)
            if not any(lower <= start < upper for lower, upper in doc_spans):
                literal_lines.update(range(token.start[0] - 1, token.end[0] - 1))
        elif token.type == tokenize.COMMENT:
            start, end = offset(*token.start), offset(*token.end)
            if fstrings:
                raise ValueError(
                    "Source identity requires syntax shared by Python 3.11 and 3.12; "
                    "PEP 701 f-string comments have version-dependent token spans"
                )
            spans.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    for start, end in reversed(merged):
        text = text[:start] + "\n" * text[start:end].count("\n") + text[end:]
    normalized = [
        line if index in literal_lines else line.rstrip()
        for index, line in enumerate(text.split("\n"))
        if index in literal_lines or line.strip()
    ]
    return SOURCE_CODE_FORMAT + ("\n".join(normalized) + "\n").encode("utf-8")
