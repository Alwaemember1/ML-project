"""构建Windows x64老师离线版；复制现有运行环境，不安装、不训练、不下载。"""

import argparse
import hashlib
import html
import json
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def imports(path):
    """读取PE普通及延迟导入表，用来复制本机Python扩展依赖的DLL。"""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", data, 0x3c)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"不是有效PE文件：{path}")
    sections, optional_size = struct.unpack_from("<H12xH", data, pe + 6)
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    directory = optional + (112 if magic == 0x20b else 96)
    image_base = struct.unpack_from("<Q" if magic == 0x20b else "<I", data,
                                    optional + (24 if magic == 0x20b else 28))[0]
    table = optional + optional_size

    def offset(rva):
        for index in range(sections):
            row = table + 40 * index
            virtual_size, address, size, raw = struct.unpack_from("<IIII", data, row + 8)
            if address <= rva < address + max(virtual_size, size):
                return raw + rva - address
        return rva

    names = []
    for index, width, name_index in ((1, 20, 3), (13, 32, 1)):
        rva, size = struct.unpack_from("<II", data, directory + index * 8)
        if not rva or not size:
            continue
        start = offset(rva)
        for row in range(start, start + size, width):
            values = struct.unpack_from("<" + "I" * (width // 4), data, row)
            if not any(values):
                break
            name = values[name_index]
            if index == 13 and not values[0] & 1:
                name -= image_base
            at = offset(name)
            names.append(data[at:data.index(b"\0", at)].decode("ascii").lower())
    return names


def ignored(directory, names):
    return [name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pdb"))]


def teacher_documents(output):
    """同一份老师说明同时提供Markdown和无需编辑器的离线HTML。"""
    markdown = (ROOT / "docs/TEACHER_OFFLINE_README.md").read_text(encoding="utf-8")
    (output / "README.md").write_text(markdown, encoding="utf-8")

    def inline(text):
        text = html.escape(text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        return text

    lines, blocks, index = markdown.splitlines(), [], 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            blocks.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(f"<h{level}>" + inline(line[level:].strip()) + f"</h{level}>")
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    tag = "th" if not rows else "td"
                    rows.append("<tr>" + "".join(f"<{tag}>{inline(cell)}</{tag}>" for cell in cells) + "</tr>")
                index += 1
            blocks.append('<div class="table"><table>' + "".join(rows) + "</table></div>")
            continue
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                blocks.append(f'<img alt="{html.escape(match[1])}" src="{html.escape(match[2])}">')
        elif line.startswith("- ") or re.match(r"\d+\. ", line):
            blocks.append("<p>" + inline(line) + "</p>")
        else:
            blocks.append("<p>" + inline(line) + "</p>")
        index += 1
    style = """body{font:16px/1.85 'Microsoft YaHei',sans-serif;color:#203b2c;background:#edf2eb;margin:0}
    main{max-width:1050px;margin:32px auto;background:white;padding:40px 48px;border-radius:12px}
    h1,h2,h3{line-height:1.5}h1{font-size:30px}h2{margin-top:38px;font-size:24px}h3{font-size:19px}
    .table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d5dfd6;padding:9px}
    th{background:#edf2eb;text-align:left}code{font-family:Consolas,monospace;overflow-wrap:anywhere}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8f4;padding:18px}img{max-width:100%}
    @media(max-width:700px){main{margin:0;padding:24px 18px}h1{font-size:25px}}@media print{body{background:white}main{margin:0;padding:0}h2,h3{break-after:avoid}tr{break-inside:avoid}}"""
    (output / "项目说明.html").write_text('<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"><title>植物叶片病害识别系统—老师说明</title>'
        f"<style>{style}</style><main>" + "\n".join(blocks) + "</main></html>", encoding="utf-8")


def launchers(output):
    shutil.copy2(ROOT / "scripts/teacher_portable_start.py", output / "portable_start.py")
    launcher = ('@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\ncd /d "%~dp0"\r\n'
                '"%~dp0runtime\\python.exe" -X utf8 -I "%~dp0portable_start.py" %*\r\n'
                'if errorlevel 1 (\r\n echo Start failed. See README.md and .runtime\\startup.log.\r\n pause\r\n)\r\n')
    (output / "启动网页.bat").write_bytes(launcher.encode("ascii"))
    (output / "启动前检查.bat").write_bytes(launcher.replace(' %*\r\n', ' --check\r\npause\r\n').encode("ascii"))
    (output / "tools").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "scripts/verify_teacher_offline.py", output / "tools/verify_offline.py")


def build(output):
    subprocess.run([sys.executable, "-m", "scripts.build_teacher_package", "--output-dir", str(output),
                    "--no-archive"], cwd=ROOT, check=True)
    (output / "DELIVERY_MANIFEST.json").rename(output / "docs/BASE_PACKAGE_MANIFEST.json")
    runtime = output / "runtime"
    runtime.mkdir()
    base = Path(sys.base_prefix)
    for name in ("python.exe", "pythonw.exe", "LICENSE_PYTHON.txt", "EULA.txt"):
        source = base / name
        if source.is_file():
            shutil.copy2(source, runtime / name)
    for source in base.glob("*.dll"):
        shutil.copy2(source, runtime / source.name)
    shutil.copytree(base / "Lib", runtime / "Lib", ignore=lambda directory, names:
                    ignored(directory, names) + ([name for name in names if name in
                    ("site-packages", "test", "idlelib", "tkinter", "ensurepip")] if Path(directory) == base / "Lib" else []))
    shutil.copytree(base / "DLLs", runtime / "DLLs", ignore=ignored)
    packages = ROOT / ".venv/Lib/site-packages"
    shutil.copytree(packages, runtime / "Lib/site-packages", ignore=lambda directory, names:
                    ignored(directory, names) + ([name for name in names if name.startswith(("_virtualenv", "pip"))]
                                                if Path(directory) == packages else []))
    # 不复制Conda环境；只复制Python二进制实际导入的非系统DLL及其递归依赖。
    candidates = {path.name.lower(): path for path in (base / "Library/bin").glob("*.dll")}
    candidates.update({path.name.lower(): path for path in base.glob("*.dll")})
    pending = [base / "python.exe", *base.glob("*.dll"), *(base / "DLLs").glob("*.pyd")]
    seen, copied, system = set(), [], set()
    while pending:
        path = pending.pop()
        if path.name.lower() in seen:
            continue
        seen.add(path.name.lower())
        for name in imports(path):
            source = candidates.get(name)
            if source:
                target = runtime / source.name
                if not target.exists():
                    shutil.copy2(source, target)
                    copied.append(source.name)
                pending.append(source)
            else:
                system.add(name)
    (runtime / "python314._pth").write_text(".\nLib\nDLLs\nLib/site-packages\n..\nimport site\n", encoding="ascii")
    (runtime / "Lib/sitecustomize.py").write_text(
        "# Portable runtime: keep temporary files in this package.\n"
        "import os, sys\nfrom pathlib import Path\n"
        "_root = Path(sys.executable).resolve().parent.parent\n"
        "_tmp = _root / '.runtime'\n_tmp.mkdir(exist_ok=True)\n"
        "for _name in ('TEMP', 'TMP', 'TMPDIR', 'XDG_CACHE_HOME', 'LEAF_RUNTIME_DIR'):\n"
        "    os.environ[_name] = str(_tmp)\n"
        "os.environ['PYTHONPYCACHEPREFIX'] = str(_tmp / 'pycache')\n"
        "sys.pycache_prefix = str(_tmp / 'pycache')\n", encoding="utf-8")
    teacher_documents(output)
    launchers(output)
    training = ROOT / "reports/v1_head_20261017_102902_302541"
    for name in ("completion.json", "training_config.json", "training_summary.json", "metrics.csv",
                 "training_curves_reviewed.png", "training.log"):
        target = output / "results/training" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(training / name, target)
    for name in ("original_test_per_class.csv", "original_test_confusion_pairs.csv"):
        shutil.copy2(ROOT / "docs/results" / name, output / "results" / name)
    (output / "RUNTIME_BUILD.json").write_text(json.dumps({
        "kind": "teacher_offline_windows_x64", "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version, "runtime_relocated": True, "venv_copied": False,
        "site_packages_copied": True, "extra_python_dlls": sorted(copied),
        "external_or_windows_imports": sorted(system), "default_device": "cpu",
        "original_environment_modified": False, "training_or_test_evaluation": False,
        "runtime_license": "runtime/LICENSE_PYTHON.txt; package dist-info licenses retained",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"directory_ready": str(output), "sealed": False}, ensure_ascii=False), flush=True)


def seal(output):
    if not (output / "RUNTIME_BUILD.json").is_file():
        raise ValueError("缺少离线环境构建标识")
    manifest = output / "DELIVERY_MANIFEST.json"
    archive = output.with_suffix(".zip")
    if manifest.exists() or archive.exists():
        raise FileExistsError("已封包，禁止覆盖")
    verification = json.loads((output / "results/offline_verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "passed":
        raise ValueError("必须先完成实际离线验收")
    record = json.loads((output / "FINAL_MODEL.json").read_text(encoding="utf-8"))
    from project_config import FROZEN_SHA256
    if record["checkpoint_sha256"] != FROZEN_SHA256 or sha256(output / record["checkpoint"]) != FROZEN_SHA256:
        raise ValueError("封包前模型SHA256核对失败")
    files = sorted(path for path in output.rglob("*") if path.is_file()
                   and ".runtime" not in path.relative_to(output).parts and "__pycache__" not in path.parts)
    manifest.write_text(json.dumps({"package_type": "teacher_offline_windows_x64",
        "model_count": 1, "runtime_included": True, "raw_dataset_included": False,
        "default_device": "cpu", "verification": "results/offline_verification.json",
        "records": [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                     "sha256": sha256(path)} for path in files]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sealing {len(files)} files...", flush=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as zipped:
        for path in [*files, manifest]:
            zipped.write(path, arcname=f"{output.name}/{path.relative_to(output).as_posix()}")
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"zip": str(archive), "bytes": archive.stat().st_size, "files": len(files)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seal-existing", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "deliverables"):
        raise ValueError("只能在项目deliverables内创建独立交付目录")
    if args.seal_existing:
        seal(output)
    else:
        build(output)


if __name__ == "__main__":
    main()
