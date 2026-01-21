from __future__ import annotations

import io
import re
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator


@dataclass(frozen=True)
class FieldSig:
    name: str
    desc: str
    access: int


@dataclass(frozen=True)
class MethodSig:
    name: str
    desc: str
    access: int
    param_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class JavaType:
    package: str
    nested_name: str
    kind: str
    file: str
    super_raw: str | None
    implements_raw: tuple[str, ...]
    imports: dict[str, str]
    wildcard_imports: tuple[str, ...]
    annotations: tuple[str, ...]
    javadoc: str | None

    @property
    def fq_name(self) -> str:
        return f"{self.package}.{self.nested_name}" if self.package else self.nested_name

    @property
    def simple_name(self) -> str:
        return self.nested_name.split(".")[-1]


@dataclass(frozen=True)
class TypeInfo:
    package: str
    nested_name: str
    kind: str
    file: str
    super_fq: str | None
    interfaces_fq: tuple[str, ...]
    annotations: tuple[str, ...]
    javadoc: str | None
    source: str
    fields: tuple[FieldSig, ...] = ()
    methods: tuple[MethodSig, ...] = ()

    @property
    def fq_name(self) -> str:
        return f"{self.package}.{self.nested_name}" if self.package else self.nested_name

    @property
    def simple_name(self) -> str:
        return self.nested_name.split(".")[-1]


JAVA_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(static\s+)?([\w.]+|\w+(?:\.\w+)*\.\*)\s*;", re.M)
BASE_EVENT_TYPES = ("net.neoforged.bus.api.Event", "net.minecraftforge.eventbus.api.Event")


def _mask_java(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    in_line = False
    in_block = False
    in_str = False
    in_chr = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line:
            if ch == "\n":
                in_line = False
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue
        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                out.extend([" ", " "])
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if in_str:
            if ch == "\\" and i + 1 < n:
                out.extend([" ", " "])
                i += 2
                continue
            if ch == '"':
                in_str = False
                out.append(" ")
            else:
                out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if in_chr:
            if ch == "\\" and i + 1 < n:
                out.extend([" ", " "])
                i += 2
                continue
            if ch == "'":
                in_chr = False
                out.append(" ")
            else:
                out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line = True
            out.extend([" ", " "])
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block = True
            out.extend([" ", " "])
            i += 2
            continue
        if ch == '"':
            in_str = True
            out.append(" ")
            i += 1
            continue
        if ch == "'":
            in_chr = True
            out.append(" ")
            i += 1
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _parse_package(text: str) -> str:
    m = JAVA_PACKAGE_RE.search(text)
    return m.group(1) if m else ""


def _parse_imports(text: str) -> tuple[dict[str, str], tuple[str, ...]]:
    imports: dict[str, str] = {}
    wildcards: list[str] = []
    for m in JAVA_IMPORT_RE.finditer(text):
        if m.group(1):
            continue
        target = m.group(2)
        if target.endswith(".*"):
            wildcards.append(target[:-2])
            continue
        simple = target.split(".")[-1]
        imports[simple] = target
    return imports, tuple(wildcards)


def _extract_javadoc_and_annotations(original: str, class_kw_pos: int) -> tuple[str | None, tuple[str, ...]]:
    prefix = original[:class_kw_pos]
    s = prefix.rstrip()
    annotations: list[str] = []

    modifier_words = {
        "public",
        "protected",
        "private",
        "abstract",
        "static",
        "final",
        "sealed",
        "non-sealed",
        "strictfp",
    }

    while True:
        before = s
        s = s.rstrip()

        lines = s.splitlines()
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].lstrip().startswith("@"):
            line = lines.pop()
            m = re.match(r"\s*@([A-Za-z_]\w*)\b", line)
            if m:
                annotations.append(m.group(1))
            s = "\n".join(lines)
            continue

        m = re.search(r"\b([A-Za-z_-]+)\s*$", s)
        if m and m.group(1) in modifier_words:
            s = s[: m.start()].rstrip()
            continue

        if s == before:
            break

    s = s.rstrip()
    if not s.endswith("*/"):
        return None, tuple(reversed(annotations))
    end = s.rfind("*/")
    start = s.rfind("/**", 0, end + 2)
    if start == -1:
        return None, tuple(reversed(annotations))
    return s[start : end + 2], tuple(reversed(annotations))


def _iter_java_types_from_source(original: str, file_label: str) -> Iterable[JavaType]:
    masked = _mask_java(original)
    package = _parse_package(original)
    imports, wildcard_imports = _parse_imports(original)

    depth = 0
    stack: list[tuple[str, int]] = []
    i = 0

    class_re = re.compile(r"\b(class|record)\s+([A-Za-z_]\w*)\b")
    while True:
        m = class_re.search(masked, i)
        if not m:
            break

        kw_pos = m.start()
        for ch in masked[i:kw_pos]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                while stack and stack[-1][1] > depth:
                    stack.pop()

        kind = m.group(1)
        name = m.group(2)

        brace_pos = masked.find("{", kw_pos)
        if brace_pos == -1:
            i = kw_pos + len(kind)
            continue

        header = masked[kw_pos:brace_pos]
        super_raw = None
        em = re.search(r"\bextends\s+([A-Za-z_][\w.]*)(?:\s*<[^>{}]*>)?", header)
        if em:
            super_raw = em.group(1)

        implements_raw: list[str] = []
        im = re.search(r"\bimplements\s+(.+)$", header, re.S)
        if im:
            chunk = im.group(1)
            chunk = re.sub(r"\bextends\b[\s\S]*$", "", chunk).strip()
            for part in chunk.split(","):
                it = part.strip()
                if not it:
                    continue
                it = it.split("<", 1)[0].strip()
                implements_raw.append(it)

        javadoc, annotations = _extract_javadoc_and_annotations(original, kw_pos)
        nested = ".".join([s[0] for s in stack] + [name]) if stack else name

        yield JavaType(
            package=package,
            nested_name=nested,
            kind=kind,
            file=file_label,
            super_raw=super_raw,
            implements_raw=tuple(implements_raw),
            imports=imports,
            wildcard_imports=wildcard_imports,
            annotations=annotations,
            javadoc=javadoc,
        )

        open_depth = depth + 1
        stack.append((name, open_depth))
        depth = open_depth
        i = brace_pos + 1


def _javadoc_summary(javadoc: str | None) -> str:
    if not javadoc:
        return ""
    body = re.sub(r"^/\*\*|\*/$", "", javadoc.strip(), flags=re.S).strip()
    lines = []
    for line in body.splitlines():
        line = re.sub(r"^\s*\*\s?", "", line).rstrip()
        lines.append(line)
    body = "\n".join(lines).strip()
    body = re.split(r"\n\s*@\w+", body, maxsplit=1)[0].strip()
    body = re.sub(r"\s+", " ", body)
    m = re.search(r"(.+?[。.!?])\s", body)
    return (m.group(1) if m else body).strip()


def _resolve_super(t: JavaType, known_fq: set[str], by_pkg_simple: dict[tuple[str, str], str]) -> str | None:
    if not t.super_raw:
        return None
    raw = t.super_raw.split("<", 1)[0].strip()
    if "." in raw:
        return raw
    if raw in t.imports:
        return t.imports[raw]
    cand_same = f"{t.package}.{raw}" if t.package else raw
    if cand_same in known_fq:
        return cand_same
    for pkg in t.wildcard_imports:
        cand = f"{pkg}.{raw}"
        if cand in known_fq:
            return cand
        key = (pkg, raw)
        if key in by_pkg_simple:
            return by_pkg_simple[key]
    return cand_same


def _resolve_type_ref(
    raw: str,
    t: JavaType,
    known_fq: set[str],
    by_pkg_simple: dict[tuple[str, str], str],
) -> str:
    base = raw.split("<", 1)[0].strip()
    if not base:
        return base
    if "." in base:
        return base
    if base in t.imports:
        return t.imports[base]
    cand_same = f"{t.package}.{base}" if t.package else base
    if cand_same in known_fq:
        return cand_same
    for pkg in t.wildcard_imports:
        cand = f"{pkg}.{base}"
        if cand in known_fq:
            return cand
        key = (pkg, base)
        if key in by_pkg_simple:
            return by_pkg_simple[key]
    return base


def _is_event_type(
    fq_name: str,
    super_map: dict[str, str | None],
    seen: set[str] | None = None,
) -> bool:
    if seen is None:
        seen = set()
    cur = fq_name
    while True:
        if cur in seen:
            return False
        seen.add(cur)
        sup = super_map.get(cur)
        if not sup:
            return False
        if sup in BASE_EVENT_TYPES:
            return True
        cur = sup


def _is_cancelable(annotations: Iterable[str], interfaces_fq: Iterable[str]) -> bool:
    ann_set = {a.split(".")[-1] for a in annotations}
    if "Cancelable" in ann_set:
        return True
    for it in interfaces_fq:
        if it.split(".")[-1] == "ICancellableEvent":
            return True
    return False


def _source_kind_from_fq(fq_name: str) -> str:
    if fq_name.startswith("net.neoforged.neoforge."):
        return "NeoForge"
    if fq_name.startswith("net.neoforged.fml."):
        return "FML"
    if fq_name.startswith("net.neoforged.bus."):
        return "Bus"
    if fq_name.startswith("net.minecraftforge."):
        return "Forge"
    if fq_name.startswith("net.minecraftforge.fml."):
        return "FML"
    if fq_name.startswith("net.minecraftforge.eventbus."):
        return "Bus"
    return "其他"


def _merge_typeinfo(old: TypeInfo, new: TypeInfo) -> TypeInfo:
    if old.javadoc and not new.javadoc:
        keep, other = old, new
    elif new.javadoc and not old.javadoc:
        keep, other = new, old
    else:
        keep, other = old, new

    anns = sorted({*keep.annotations, *other.annotations})
    ifaces = sorted({*keep.interfaces_fq, *other.interfaces_fq})
    super_fq = keep.super_fq or other.super_fq
    return TypeInfo(
        package=keep.package,
        nested_name=keep.nested_name,
        kind=keep.kind,
        file=keep.file,
        super_fq=super_fq,
        interfaces_fq=tuple(ifaces),
        annotations=tuple(anns),
        javadoc=keep.javadoc or other.javadoc,
        source=keep.source,
    )


def _typeinfo_from_java(t: JavaType, super_fq: str | None, interfaces_fq: tuple[str, ...], source: str) -> TypeInfo:
    return TypeInfo(
        package=t.package,
        nested_name=t.nested_name,
        kind=t.kind,
        file=t.file,
        super_fq=super_fq,
        interfaces_fq=interfaces_fq,
        annotations=tuple(t.annotations),
        javadoc=t.javadoc,
        source=source,
    )


def _cp_get_utf8(cp: list[object | None], idx: int) -> str:
    v = cp[idx]
    if isinstance(v, str):
        return v
    raise ValueError(f"无效常量池 Utf8 索引: {idx}")


def _cp_get_class_name(cp: list[object | None], idx: int) -> str:
    v = cp[idx]
    if isinstance(v, tuple) and v and v[0] == "Class":
        return _cp_get_utf8(cp, v[1])
    raise ValueError(f"无效常量池 Class 索引: {idx}")


def _read_u1(buf: io.BytesIO) -> int:
    b = buf.read(1)
    if len(b) != 1:
        raise EOFError
    return b[0]


def _read_u2(buf: io.BytesIO) -> int:
    b = buf.read(2)
    if len(b) != 2:
        raise EOFError
    return struct.unpack(">H", b)[0]


def _read_u4(buf: io.BytesIO) -> int:
    b = buf.read(4)
    if len(b) != 4:
        raise EOFError
    return struct.unpack(">I", b)[0]


def _skip(buf: io.BytesIO, n: int) -> None:
    if n <= 0:
        return
    cur = buf.tell()
    buf.seek(cur + n)


def _parse_element_value(buf: io.BytesIO) -> None:
    tag = chr(_read_u1(buf))
    if tag in "BCDFIJSZs":
        _skip(buf, 2)
        return
    if tag == "e":
        _skip(buf, 4)
        return
    if tag == "c":
        _skip(buf, 2)
        return
    if tag == "@":
        _parse_annotation(buf)
        return
    if tag == "[":
        n = _read_u2(buf)
        for _ in range(n):
            _parse_element_value(buf)
        return
    _skip(buf, 2)


def _parse_annotation(buf: io.BytesIO) -> None:
    _skip(buf, 2)
    pairs = _read_u2(buf)
    for _ in range(pairs):
        _skip(buf, 2)
        _parse_element_value(buf)


def _read_annotations_attribute(buf: io.BytesIO, cp: list[object | None]) -> list[str]:
    out: list[str] = []
    num = _read_u2(buf)
    for _ in range(num):
        type_index = _read_u2(buf)
        desc = _cp_get_utf8(cp, type_index)
        out.append(desc)
        pairs = _read_u2(buf)
        for _ in range(pairs):
            _skip(buf, 2)
            _parse_element_value(buf)
    return out


def _parse_classfile(data: bytes) -> tuple[str, str | None, list[str], list[str]]:
    buf = io.BytesIO(data)
    magic = _read_u4(buf)
    if magic != 0xCAFEBABE:
        raise ValueError("不是有效的 .class 文件")
    _skip(buf, 4)
    cp_count = _read_u2(buf)
    cp: list[object | None] = [None] * cp_count
    i = 1
    while i < cp_count:
        tag = _read_u1(buf)
        if tag == 1:
            ln = _read_u2(buf)
            cp[i] = buf.read(ln).decode("utf-8", errors="replace")
        elif tag == 7:
            cp[i] = ("Class", _read_u2(buf))
        elif tag == 8:
            _skip(buf, 2)
        elif tag in (3, 4):
            _skip(buf, 4)
        elif tag in (9, 10, 11, 12):
            _skip(buf, 4)
        elif tag in (17, 18):
            _skip(buf, 4)
        elif tag in (5, 6):
            _skip(buf, 8)
            i += 1
        elif tag == 15:
            _skip(buf, 3)
        elif tag == 16:
            _skip(buf, 2)
        elif tag in (19, 20):
            _skip(buf, 2)
        else:
            raise ValueError(f"未知常量池 tag={tag}")
        i += 1

    _skip(buf, 2)
    this_class = _read_u2(buf)
    super_class = _read_u2(buf)
    this_name = _cp_get_class_name(cp, this_class)
    super_name = None if super_class == 0 else _cp_get_class_name(cp, super_class)
    interfaces_count = _read_u2(buf)
    interfaces: list[str] = []
    for _ in range(interfaces_count):
        interfaces.append(_cp_get_class_name(cp, _read_u2(buf)))

    fields_count = _read_u2(buf)
    for _ in range(fields_count):
        _skip(buf, 6)
        ac = _read_u2(buf)
        for _ in range(ac):
            _skip(buf, 2)
            ln = _read_u4(buf)
            _skip(buf, ln)

    methods_count = _read_u2(buf)
    for _ in range(methods_count):
        _skip(buf, 6)
        ac = _read_u2(buf)
        for _ in range(ac):
            _skip(buf, 2)
            ln = _read_u4(buf)
            _skip(buf, ln)

    annotations: list[str] = []
    attrs_count = _read_u2(buf)
    for _ in range(attrs_count):
        name_index = _read_u2(buf)
        length = _read_u4(buf)
        name = _cp_get_utf8(cp, name_index)
        payload = buf.read(length)
        if name in ("RuntimeVisibleAnnotations", "RuntimeInvisibleAnnotations"):
            ann_buf = io.BytesIO(payload)
            annotations.extend(_read_annotations_attribute(ann_buf, cp))

    return this_name, super_name, interfaces, annotations


def _parse_classfile_members(data: bytes) -> tuple[list[FieldSig], list[MethodSig]]:
    buf = io.BytesIO(data)
    magic = _read_u4(buf)
    if magic != 0xCAFEBABE:
        raise ValueError("不是有效的 .class 文件")
    _skip(buf, 4)
    cp_count = _read_u2(buf)
    cp: list[object | None] = [None] * cp_count
    i = 1
    while i < cp_count:
        tag = _read_u1(buf)
        if tag == 1:
            ln = _read_u2(buf)
            cp[i] = buf.read(ln).decode("utf-8", errors="replace")
        elif tag == 7:
            cp[i] = ("Class", _read_u2(buf))
        elif tag == 8:
            _skip(buf, 2)
        elif tag in (3, 4):
            _skip(buf, 4)
        elif tag in (9, 10, 11, 12):
            _skip(buf, 4)
        elif tag in (17, 18):
            _skip(buf, 4)
        elif tag in (5, 6):
            _skip(buf, 8)
            i += 1
        elif tag == 15:
            _skip(buf, 3)
        elif tag == 16:
            _skip(buf, 2)
        elif tag in (19, 20):
            _skip(buf, 2)
        else:
            raise ValueError(f"未知常量池 tag={tag}")
        i += 1

    _skip(buf, 2)
    _skip(buf, 2)
    _skip(buf, 2)
    interfaces_count = _read_u2(buf)
    _skip(buf, interfaces_count * 2)

    fields: list[FieldSig] = []
    fields_count = _read_u2(buf)
    for _ in range(fields_count):
        access_flags = _read_u2(buf)
        name = _cp_get_utf8(cp, _read_u2(buf))
        desc = _cp_get_utf8(cp, _read_u2(buf))
        ac = _read_u2(buf)
        for _ in range(ac):
            _skip(buf, 2)
            ln = _read_u4(buf)
            _skip(buf, ln)
        fields.append(FieldSig(name=name, desc=desc, access=access_flags))

    methods: list[MethodSig] = []
    methods_count = _read_u2(buf)
    for _ in range(methods_count):
        access_flags = _read_u2(buf)
        name = _cp_get_utf8(cp, _read_u2(buf))
        desc = _cp_get_utf8(cp, _read_u2(buf))
        lvt: dict[int, str] = {}
        lvt_start: dict[int, int] = {}
        mp_names: list[str] | None = None
        ac = _read_u2(buf)
        for _ in range(ac):
            attr_name = _cp_get_utf8(cp, _read_u2(buf))
            ln = _read_u4(buf)
            if attr_name == "MethodParameters":
                payload = buf.read(ln)
                pbuf = io.BytesIO(payload)
                count = _read_u1(pbuf) if ln >= 1 else 0
                names: list[str] = []
                for _ in range(count):
                    name_index = _read_u2(pbuf)
                    _skip(pbuf, 2)
                    names.append(_cp_get_utf8(cp, name_index) if name_index else "")
                mp_names = names
                continue
            if attr_name == "Code":
                _skip(buf, 4)
                code_len = _read_u4(buf)
                _skip(buf, code_len)
                ex_len = _read_u2(buf)
                _skip(buf, ex_len * 8)
                sub_ac = _read_u2(buf)
                for _ in range(sub_ac):
                    sub_name = _cp_get_utf8(cp, _read_u2(buf))
                    sub_ln = _read_u4(buf)
                    if sub_name == "LocalVariableTable":
                        payload = buf.read(sub_ln)
                        sbuf = io.BytesIO(payload)
                        n = _read_u2(sbuf) if sub_ln >= 2 else 0
                        for _ in range(n):
                            start_pc = _read_u2(sbuf)
                            _skip(sbuf, 2)
                            name_index = _read_u2(sbuf)
                            _skip(sbuf, 2)
                            idx = _read_u2(sbuf)
                            if not name_index:
                                continue
                            if idx not in lvt_start or start_pc < lvt_start[idx]:
                                lvt_start[idx] = start_pc
                                lvt[idx] = _cp_get_utf8(cp, name_index)
                    else:
                        _skip(buf, sub_ln)
                continue
            _skip(buf, ln)

        arg_types, _ret = _method_desc_to_sig(desc)
        arg_count = len(arg_types)
        param_names = [f"p{i}" for i in range(arg_count)]
        if mp_names and len(mp_names) == arg_count:
            for i, n in enumerate(mp_names):
                if n and _is_valid_java_ident(n):
                    param_names[i] = n
        else:
            is_static = bool(access_flags & 0x0008)
            slots = _method_param_slot_indices(desc, is_static)
            if len(slots) == arg_count:
                for i, slot in enumerate(slots):
                    n = lvt.get(slot, "")
                    if n and _is_valid_java_ident(n) and n != "this":
                        param_names[i] = n
        methods.append(MethodSig(name=name, desc=desc, access=access_flags, param_names=tuple(param_names)))

    return fields, methods


def _parse_one_type(desc: str, i: int) -> tuple[str | None, int]:
    arr = 0
    while i < len(desc) and desc[i] == "[":
        arr += 1
        i += 1
    if i >= len(desc):
        return None, i
    ch = desc[i]
    i += 1
    prim = {
        "B": "byte",
        "C": "char",
        "D": "double",
        "F": "float",
        "I": "int",
        "J": "long",
        "S": "short",
        "Z": "boolean",
        "V": "void",
    }
    if ch in prim:
        t = prim[ch]
    elif ch == "L":
        semi = desc.find(";", i)
        if semi == -1:
            return None, i
        internal = desc[i:semi]
        i = semi + 1
        fq = internal.replace("/", ".").replace("$", ".")
        t = fq.split(".")[-1]
    else:
        return None, i
    if arr:
        t = t + "[]" * arr
    return t, i


def _method_desc_to_sig(desc: str) -> tuple[list[str], str]:
    if not desc.startswith("("):
        return [], "void"
    end = desc.find(")")
    if end == -1:
        return [], "void"
    args_desc = desc[1:end]
    ret_desc = desc[end + 1 :]
    args: list[str] = []
    i = 0
    while i < len(args_desc):
        t, ni = _parse_one_type(args_desc, i)
        if not t or ni <= i:
            break
        args.append(t)
        i = ni
    ret, _ = _parse_one_type(ret_desc, 0)
    return args, (ret or "void")


def _field_desc_to_type(desc: str) -> str:
    t, _ = _parse_one_type(desc, 0)
    return t or "Object"


_JAVA_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def _is_valid_java_ident(name: str) -> bool:
    return bool(_JAVA_IDENT_RE.fullmatch(name))


def _method_param_slot_indices(desc: str, is_static: bool) -> list[int]:
    if not desc.startswith("("):
        return []
    end = desc.find(")")
    if end == -1:
        return []
    args_desc = desc[1:end]
    indices: list[int] = []
    slot = 0 if is_static else 1
    i = 0
    while i < len(args_desc):
        indices.append(slot)
        ch = args_desc[i]
        if ch == "[":
            while i < len(args_desc) and args_desc[i] == "[":
                i += 1
            if i < len(args_desc) and args_desc[i] == "L":
                semi = args_desc.find(";", i)
                if semi == -1:
                    break
                i = semi + 1
            else:
                i += 1
            slot += 1
            continue
        if ch == "L":
            semi = args_desc.find(";", i)
            if semi == -1:
                break
            i = semi + 1
            slot += 1
            continue
        if ch in ("J", "D"):
            i += 1
            slot += 2
            continue
        i += 1
        slot += 1
    return indices


def _access_mods(access: int, is_method: bool) -> list[str]:
    mods: list[str] = []
    if access & 0x0001:
        mods.append("public")
    elif access & 0x0004:
        mods.append("protected")
    elif access & 0x0002:
        mods.append("private")
    if access & 0x0008:
        mods.append("static")
    if access & 0x0010:
        mods.append("final")
    if is_method:
        if access & 0x0400:
            mods.append("abstract")
        if access & 0x0020:
            mods.append("synchronized")
        if access & 0x0100:
            mods.append("native")
        if access & 0x0800:
            mods.append("strictfp")
    else:
        if access & 0x0040:
            mods.append("volatile")
        if access & 0x0080:
            mods.append("transient")
    return mods


def _parse_mod_source(source: str) -> tuple[str, str | None]:
    if not source.startswith("Mod:"):
        return source, None
    s = source[len("Mod:") :]
    if "(" in s and s.endswith(")"):
        mod_id, display = s.split("(", 1)
        return mod_id, display[:-1]
    return s, None


def _safe_md_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s or "mod"


def _java_stub(t: TypeInfo) -> str:
    class_name = t.simple_name
    super_name = t.super_fq.split(".")[-1] if t.super_fq else None
    header = f"public class {class_name}"
    if super_name:
        header += f" extends {super_name}"
    header += " {"
    lines: list[str] = []
    lines.append(f"package {t.package};")
    lines.append("")
    lines.append(header)

    for f in t.fields:
        if f.access & 0x1000:
            continue
        mods = " ".join(_access_mods(f.access, is_method=False))
        ft = _field_desc_to_type(f.desc)
        prefix = (mods + " ") if mods else ""
        lines.append(f"    {prefix}{ft} {f.name};")

    for m in t.methods:
        if m.name in ("<init>", "<clinit>"):
            continue
        if m.access & 0x1000:
            continue
        if m.access & 0x0040:
            continue
        mods = " ".join(_access_mods(m.access, is_method=True))
        arg_types, ret_type = _method_desc_to_sig(m.desc)
        names = list(m.param_names) if len(m.param_names) == len(arg_types) else [f"p{i}" for i in range(len(arg_types))]
        params = ", ".join(f"{at} {names[i]}" for i, at in enumerate(arg_types))
        prefix = (mods + " ") if mods else ""
        lines.append(f"    {prefix}{ret_type} {m.name}({params});")

    lines.append("}")
    return "\n".join(lines)


def _write_mod_docs(docs_dir: Path, grouped: dict[str, dict[str, list[TypeInfo]]]) -> list[tuple[str, str, int]]:
    mods = {src: pkgs for src, pkgs in grouped.items() if src.startswith("Mod:")}
    if not mods:
        return []
    docs_dir.mkdir(parents=True, exist_ok=True)
    index: list[tuple[str, str, int]] = []

    for src, pkgs in mods.items():
        mod_id, display = _parse_mod_source(src)
        name = display or mod_id
        fname = _safe_md_name(mod_id) + ".md"
        total = sum(len(v) for v in pkgs.values())
        index.append((name, fname, total))
        lines: list[str] = []
        lines.append(f"# {name}")
        lines.append("")
        lines.append(f"- 模组 ID：`{mod_id}`")
        lines.append(f"- 事件子类：{total}")
        lines.append("")
        lines.append("## 包目录")
        lines.append("")
        for pkg in sorted(pkgs.keys()):
            lines.append(f"- [{pkg}](#{_md_anchor(pkg)})（{len(pkgs[pkg])}）")
        lines.append("")
        for pkg in sorted(pkgs.keys()):
            lines.append(f"## {pkg}")
            lines.append("")
            for t in pkgs[pkg]:
                lines.append(f"### `{t.fq_name}`")
                lines.append("")
                lines.append("```java")
                lines.append(_java_stub(t))
                lines.append("```")
                lines.append("")
        (docs_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
    index.sort(key=lambda x: x[0].lower())
    return index


def _descriptor_to_simple_name(desc: str) -> str:
    if desc.startswith("L") and desc.endswith(";"):
        core = desc[1:-1].replace("/", ".")
        return core.split(".")[-1]
    return desc


def _internal_to_fq(internal: str) -> str:
    return internal.replace("/", ".").replace("$", ".")


def _parse_modid_from_toml(text: str) -> tuple[str | None, str | None]:
    mod_id = None
    display = None
    in_mods = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[[mods]]"):
            in_mods = True
            continue
        if in_mods and line.startswith("[["):
            break
        if not in_mods:
            continue
        m = re.match(r'modId\s*=\s*["\']([^"\']+)["\']', line)
        if m and not mod_id:
            mod_id = m.group(1)
        m = re.match(r'displayName\s*=\s*["\']([^"\']+)["\']', line)
        if m and not display:
            display = m.group(1)
        if mod_id and display:
            break
    return mod_id, display


def _jar_source_label(jar_path: Path, zf: zipfile.ZipFile) -> str:
    for name in ("META-INF/neoforge.mods.toml", "META-INF/mods.toml"):
        try:
            data = zf.read(name)
        except KeyError:
            continue
        text = data.decode("utf-8", errors="ignore")
        mod_id, display = _parse_modid_from_toml(text)
        if mod_id:
            return f"Mod:{mod_id}" if not display else f"Mod:{mod_id}({display})"
    for marker, label in (
        ("net/minecraftforge/common/MinecraftForge.class", "Forge"),
        ("net/neoforged/bus/api/Event.class", "NeoForge"),
    ):
        try:
            zf.getinfo(marker)
            return label
        except KeyError:
            pass
    return f"Jar:{jar_path.name}"


def _iter_jar_types(jar_path: Path, progress: Callable[[str, int, int], None] | None = None) -> Iterator[TypeInfo]:
    with zipfile.ZipFile(jar_path, "r") as zf:
        source = _jar_source_label(jar_path, zf)
        if source.startswith("Jar:"):
            return
        class_infos = [
            info
            for info in zf.infolist()
            if info.filename.endswith(".class") and not info.filename.startswith("META-INF/")
        ]
        total = len(class_infos)
        for idx, info in enumerate(class_infos, 1):
            data = zf.read(info)
            try:
                this_internal, super_internal, ifaces_internal, ann_desc = _parse_classfile(data)
            except Exception:
                continue
            fields: tuple[FieldSig, ...] = ()
            methods: tuple[MethodSig, ...] = ()
            try:
                f, m = _parse_classfile_members(data)
                fields = tuple(f)
                methods = tuple(m)
            except Exception:
                pass
            if progress and total and (idx == 1 or idx % 200 == 0 or idx == total):
                progress(source, idx, total)
            pkg_internal = this_internal.rsplit("/", 1)[0] if "/" in this_internal else ""
            simple_internal = this_internal.rsplit("/", 1)[-1]
            pkg = pkg_internal.replace("/", ".")
            nested = simple_internal.replace("$", ".")
            super_fq = _internal_to_fq(super_internal) if super_internal else None
            ifaces_fq = tuple(_internal_to_fq(x) for x in ifaces_internal)
            anns = tuple(sorted({_descriptor_to_simple_name(d) for d in ann_desc}))
            yield TypeInfo(
                package=pkg,
                nested_name=nested,
                kind="class",
                file=f"{jar_path}!{info.filename}",
                super_fq=super_fq,
                interfaces_fq=ifaces_fq,
                annotations=anns,
                javadoc=None,
                source=source,
                fields=fields,
                methods=methods,
            )


def generate_markdown(
    src_jars: list[tuple[str, Path]] | None,
    jar_paths: list[Path] | None,
    out_md: Path,
    docs_dir: Path | None,
) -> int:
    package_docs: dict[tuple[str, str], str] = {}
    java_types: list[tuple[str, JavaType]] = []
    type_by_name: dict[str, TypeInfo] = {}
    last_len = 0

    def progress(label: str, cur: int, total: int) -> None:
        nonlocal last_len
        pct = 0 if total <= 0 else int(cur * 100 / total)
        if not sys.stdout.isatty():
            print(f"[{label}] {cur}/{total} {pct}%")
            return
        line = f"[{label}] {cur}/{total} {pct}%"
        pad = " " * max(0, last_len - len(line))
        sys.stdout.write("\r" + line + pad)
        sys.stdout.flush()
        last_len = len(line)

    def progress_done() -> None:
        nonlocal last_len
        if sys.stdout.isatty() and last_len:
            sys.stdout.write("\n")
            sys.stdout.flush()
            last_len = 0

    def progress_finish(label: str, total: int) -> None:
        if not sys.stdout.isatty() or total <= 0:
            return
        progress(label, total, total)
        progress_done()

    if src_jars:
        for src_label, src_jar in src_jars:
            with zipfile.ZipFile(src_jar, "r") as zf:
                java_entries = [n for n in zf.namelist() if n.endswith(".java")]
                total_java = len(java_entries)
                for idx, name in enumerate(java_entries, 1):
                    try:
                        original = zf.read(name).decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if name.endswith("/package-info.java") or name == "package-info.java":
                        pkg = _parse_package(original)
                        m = re.search(r"/\*\*[\s\S]*?\*/\s*package\s+[\w.]+\s*;", original)
                        if m:
                            block = re.search(r"/\*\*[\s\S]*?\*/", m.group(0))
                            if pkg and block:
                                package_docs[(src_label, pkg)] = _javadoc_summary(block.group(0))
                        if total_java and (idx == 1 or idx % 200 == 0 or idx == total_java):
                            progress(f"{src_label} 源码Jar解析", idx, total_java)
                        continue
                    for t in _iter_java_types_from_source(original, f"{src_jar}!{name}"):
                        java_types.append((src_label, t))
                    if total_java and (idx == 1 or idx % 200 == 0 or idx == total_java):
                        progress(f"{src_label} 源码Jar解析", idx, total_java)
                progress_finish(f"{src_label} 源码Jar解析", total_java)

    if java_types:
        known_fq = {t.fq_name for _, t in java_types}
        by_pkg_simple: dict[tuple[str, str], str] = {(t.package, t.simple_name): t.fq_name for _, t in java_types}
        for src_label, t in java_types:
            super_fq = _resolve_super(t, known_fq, by_pkg_simple)
            interfaces_fq = tuple(_resolve_type_ref(x, t, known_fq, by_pkg_simple) for x in t.implements_raw)
            info = _typeinfo_from_java(t, super_fq, interfaces_fq, src_label)
            if info.fq_name in type_by_name:
                type_by_name[info.fq_name] = _merge_typeinfo(type_by_name[info.fq_name], info)
            else:
                type_by_name[info.fq_name] = info

    if jar_paths:
        for jar in sorted(set(jar_paths)):
            jar_label = None
            jar_total = 0

            def jar_progress(label: str, cur: int, total: int) -> None:
                nonlocal jar_label, jar_total
                jar_label = label
                jar_total = total
                progress(label, cur, total)

            for info in _iter_jar_types(jar, progress=jar_progress):
                if info.fq_name in type_by_name:
                    type_by_name[info.fq_name] = _merge_typeinfo(type_by_name[info.fq_name], info)
                else:
                    type_by_name[info.fq_name] = info
            if jar_label:
                progress_finish(jar_label, jar_total)

    total_types = len(type_by_name)
    super_map: dict[str, str | None] = {fq: t.super_fq for fq, t in type_by_name.items()}
    event_types = [t for t in type_by_name.values() if _is_event_type(t.fq_name, super_map)]

    def source_sort_key(s: str) -> tuple[int, str]:
        if s == "Forge":
            return (0, s)
        if s == "NeoForge":
            return (1, s)
        if s.startswith("Mod:"):
            return (2, s.lower())
        if s.startswith("Jar:"):
            return (3, s.lower())
        return (4, s.lower())

    def pkg_key(p: str) -> tuple[int, str]:
        if p.startswith("net.minecraftforge.event"):
            return (0, p)
        if p.startswith("net.minecraftforge.fml.event"):
            return (1, p)
        if p.startswith("net.neoforged.neoforge.event"):
            return (2, p)
        if p.startswith("net.neoforged.fml.event"):
            return (3, p)
        if ".event" in p:
            return (4, p)
        return (5, p)

    grouped: dict[str, dict[str, list[TypeInfo]]] = {}
    for t in event_types:
        grouped.setdefault(t.source, {}).setdefault(t.package, []).append(t)

    for src in grouped:
        for pkg in grouped[src]:
            grouped[src][pkg].sort(key=lambda x: (x.simple_name.lower(), x.nested_name.lower()))

    sources_sorted = sorted(grouped.keys(), key=source_sort_key)

    lines: list[str] = []
    lines.append("# Event 子类索引")
    lines.append("")
    has_framework_sources = bool(src_jars)
    lines.append("- 说明：框架事件可从源码提取 JavaDoc 摘要；仅从 .class/.jar 解析时摘要通常为空。")
    lines.append(f"- 框架源码：{'已提供' if has_framework_sources else '未提供'}")
    lines.append("")
    lines.append("## 使用要点")
    lines.append("")
    lines.append("- 订阅方式：在任意类中编写 `@SubscribeEvent` 方法，并注册监听器。")
    lines.append("- 可取消字段：满足 `@Cancelable` 或实现 `ICancellableEvent` 即标记为“是”。")
    lines.append("")
    lines.append(f"- 统计：事件子类共 {len(event_types)} 个。")
    if docs_dir is not None:
        mod_index = _write_mod_docs(docs_dir, grouped)
        if mod_index:
            lines.append("")
            lines.append("## 模组文档")
            lines.append("")
            for name, fname, total in mod_index:
                lines.append(f"- [{name}]({fname})（{total}）")
    lines.append("## 目录")
    lines.append("")
    for src in sources_sorted:
        anchor = _md_anchor(src)
        total = sum(len(v) for v in grouped[src].values())
        lines.append(f"- [{src}](#{anchor})（{total}）")
    lines.append("")

    for src in sources_sorted:
        lines.append(f"## {src}")
        lines.append("")
        pkgs = sorted(grouped[src].keys(), key=pkg_key)
        

        for pkg in pkgs:
            anchor = _md_anchor(pkg)
            lines.append(f"- [{pkg}](#{anchor})（{len(grouped[src][pkg])}）")
        lines.append("")

        for pkg in pkgs:
            lines.append(f"### {pkg}")
            lines.append("")
            pkg_doc = package_docs.get((src, pkg), "")
            if pkg_doc:
                lines.append(f"- 包说明：{pkg_doc}")
                lines.append("")
            lines.append("| 事件类型 | 用途/触发点（摘要） | 可取消 |")
            lines.append("|---|---|---|")
            for t in grouped[src][pkg]:
                summary = _javadoc_summary(t.javadoc)
                cancelable = "是" if _is_cancelable(t.annotations, t.interfaces_fq) else ""
                lines.append(f"| `{t.fq_name}` | {summary} | {cancelable} |")
            lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(event_types)


def _md_anchor(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^\w\u4e00-\u9fff .-]+", "", s)
    s = s.replace(" ", "-").replace(".", "")
    return s


def main() -> None:
    base_dir = Path(sys.argv[0]).resolve().parent
    

    src_jars: list[tuple[str, Path]] = []
    jar_paths: list[Path] = []
    
    candidates: list[Path] = []
    candidates.extend(base_dir.glob("*.jar"))
    for sub in ["libs", "mods"]:
        sub_dir = base_dir / sub
        if sub_dir.exists():
            candidates.extend(sub_dir.glob("*.jar"))

    seen_paths = set()
    for p in candidates:
        if p in seen_paths:
            continue
        seen_paths.add(p)
        
        if p.name.endswith("-sources.jar"):
            label = "NeoForge" if "neoforge" in p.name.lower() else ("Forge" if "forge" in p.name.lower() else "源码")
            src_jars.append((label, p))
        else:
            jar_paths.append(p)

    docs_dir = base_dir / "docs"
    out = docs_dir / "events.md"

    gradle_home = Path.home() / ".gradle"
    neoforge_cache = gradle_home / "caches" / "modules-2" / "files-2.1" / "net.neoforged" / "neoforge"
    forge_cache = gradle_home / "caches" / "forge_gradle" / "maven_downloader" / "net" / "minecraftforge" / "forge"

    tips = (
        "建议：请从以下 Gradle 缓存目录查找带有 -sources.jar ：\n"
        f"  - NeoForge: {neoforge_cache}\\<版本>\\<Hash>\\neoforge-<版本>-sources.jar\n"
        f"  - Forge:    {forge_cache}\\<版本>\\forge-<版本>-sources.jar\n"
        "找到后请复制到脚本同级目录。"
    )

    if not src_jars and not jar_paths:
        print(f"提示：在 {base_dir} 下未发现有效的源码 Jar 或模组 Jar。")
        print("请将 Jar 文件放入脚本同级目录。")
        print(tips)
        return

    if not src_jars:
        print("提示：未检测到框架源码（NeoForge/Forge）")
        print(tips)
        print("-" * 50)
    
    print(f"开始生成...")
    if src_jars:
        print(f"  源码Jar: {', '.join(f'{l}:{p.name}' for l, p in src_jars)}")
    if jar_paths:
        print(f"  扫描Jar: {len(jar_paths)} 个")

    total_events = generate_markdown(src_jars or None, jar_paths or None, out, docs_dir)
    print(f"生成完毕。总共事件：{total_events}，输出：{out}")


if __name__ == "__main__":
    main()
