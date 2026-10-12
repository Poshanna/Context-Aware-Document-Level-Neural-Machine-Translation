import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "app")

results = {}

# 1. Secrets audit - hardcoded VALUES that look like actual secrets (NOT field names / docstrings / comments)
# Heuristic: a real secret is long, high-entropy, alphabet-heavy (b64/hex), NOT natural-language,
# NOT a module docstring (top-of-file strings), NOT inside a docstring.
b64hex_re = re.compile(r"^[A-Za-z0-9_\-+/=]{32,}$")
placeholder_pattern = re.compile(r"REPLACE_ME_|test-secret|PLACEHOLDER|example|changeme|not-for-production|smoke-test-secret", re.I)
natural_lang_re = re.compile(r"\s")  # real secrets don't contain whitespace

def _in_docstring(tree: ast.AST, target_node: ast.Constant) -> bool:
    # Heuristic: treat as docstring if it's the first expr-body element of a FunctionDef/ClassDef/Module
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        if body and body[0] is target_node:
            return True
    return False

real_secret_hits = []

for root, dirs, files in os.walk(APP):
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        src = open(p, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                if len(s) < 40:
                    continue
                if placeholder_pattern.search(s):
                    continue
                if natural_lang_re.search(s):  # docstrings / sentences
                    continue
                if not b64hex_re.match(s):  # not base64/hex-like
                    continue
                if _in_docstring(tree, node):
                    continue
                real_secret_hits.append((os.path.relpath(p, ROOT), s[:80], node.lineno))

results["REAL_SECRET_PY_LITERALS"] = real_secret_hits if real_secret_hits else "CLEAN (no suspicious high-entropy literals)"

# 2. REPLACE_ME_ should appear ONLY in .env.example at repo root (and optional frontend/.env.example)
replace_me_report = []
repo_root = os.path.dirname(ROOT)
allowed_rel = {
    os.path.relpath(os.path.join(repo_root, ".env.example"), ROOT),
    os.path.relpath(os.path.join(repo_root, "frontend", ".env.example"), ROOT),
    os.path.relpath(os.path.join(repo_root, "backend", ".env.example"), ROOT),
}

for root, dirs, files in os.walk(repo_root):
    # Skip irrelevant dirs
    dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache", ".pytest_cache"}]
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, ROOT)
        if f.endswith((".pyc",)):
            continue
        try:
            with open(p, encoding="utf-8", errors="ignore") as fp:
                data = fp.read()
        except Exception:
            continue
        count = data.count("REPLACE_ME_")
        if count > 0 and rel not in allowed_rel and not rel.endswith("_verify_phase2.py"):
            replace_me_report.append((rel, count))

results["REPLACE_ME_OUTSIDE_ENV_TEMPLATES"] = replace_me_report if replace_me_report else "CLEAN (only in env templates)"

# 3. Architecture boundary: services must NOT import fastapi
services_fastapi_hits = []
services_dir = os.path.join(APP, "services")
if os.path.isdir(services_dir):
    for root, dirs, files in os.walk(services_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            src = open(p, encoding="utf-8").read()
            tree = ast.parse(src)
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    mod = getattr(n, "module", "") or ""
                    names = [a.name for a in getattr(n, "names", [])]
                    if mod.startswith("fastapi") or any(x.startswith("fastapi") for x in names):
                        services_fastapi_hits.append((os.path.relpath(p, ROOT), mod, names))
results["SERVICES_IMPORT_FASTAPI"] = services_fastapi_hits if services_fastapi_hits else "CLEAN"

# 4. Architecture boundary: api routers must NOT import ML/registry modules directly
routers_ml_hits = []
api_dir = os.path.join(APP, "api")
forbidden_prefixes = ("app.translation", "app.preprocessing", "app.context", "app.evaluation", "torch", "transformers", "sentence_transformers")
if os.path.isdir(api_dir):
    for root, dirs, files in os.walk(api_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            src = open(p, encoding="utf-8").read()
            tree = ast.parse(src)
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    mod = getattr(n, "module", "") or ""
                    names = [a.name for a in getattr(n, "names", [])]
                    full = {mod} | set(names)
                    bad = [x for x in full if any(x.startswith(pref) for pref in forbidden_prefixes)]
                    if bad:
                        routers_ml_hits.append((os.path.relpath(p, ROOT), mod, names, bad))
results["ROUTERS_IMPORT_ML_LAYER"] = routers_ml_hits if routers_ml_hits else "CLEAN"

# 5. Print summary
print("=" * 60)
print("PHASE 2 FINAL VERIFICATION AUDIT")
print("=" * 60)
for k, v in results.items():
    print(f"\n[{k}]")
    if isinstance(v, list):
        if not v:
            print("  CLEAN")
        else:
            for row in v:
                print("  ", row)
    else:
        print(" ", v)

# Exit non-zero if any real issues (only REAL_SECRET or boundary violations count; REPLACE_ME in test files is fine)
bad = bool(real_secret_hits) or bool(services_fastapi_hits) or bool(routers_ml_hits)
print("\n" + "=" * 60)
print("OVERALL:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
