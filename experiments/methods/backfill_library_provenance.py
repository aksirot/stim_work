"""Backfill model/family/code provenance onto existing failure-library entries.

Provenance is load-bearing: mechanism indices are comparable only within a code, and
a failing configuration is only meaningful for the decoder of ITS device family
(symmetric models share the full-symmetric device decoder; the x5 asym ray is a
different device with its own priors — an asym entry benched against the symmetric
decoder is a category error). Existing entries encode the source in `generator`;
this maps that to explicit fields, idempotently.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from repo_paths import REPO_ROOT

SLUG2MODEL = {"full_symmetric": "full symmetric", "CZ_only": "CZ only",
              "meas_only": "meas only", "prep_only": "prep only",
              "gate_idle": "gate idle", "meas_idle": "meas idle"}


def resolve(gen, default_model, default_family, default_code):
    for prefix in ("device_topup_", "w3_harvest_"):
        if gen.startswith(prefix):
            slug = gen[len(prefix):]
            return SLUG2MODEL.get(slug, slug), "symmetric", default_code
    return default_model, default_family, default_code


def backfill(path, default_model, default_family, default_code):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"{p}: missing, skipped")
        return
    lib = json.loads(p.read_text(encoding="utf-8"))
    counts = {}
    for e in lib["entries"]:
        m, f, c = resolve(e.get("generator", ""), default_model, default_family,
                          default_code)
        e.setdefault("model", m)
        e.setdefault("family", f)
        e.setdefault("code", c)
        counts[(e["model"], e["family"])] = counts.get((e["model"], e["family"]), 0) + 1
    p.write_text(json.dumps(lib), encoding="utf-8")
    print(f"{p.name}: {len(lib['entries'])} entries")
    for (m, f), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {n:4d}  model={m!r} family={f!r}")


if __name__ == "__main__":
    backfill(REPO_ROOT / "runs" / "decoder_loop" / "library.json",
             "full symmetric", "symmetric", "bb72")
    backfill(REPO_ROOT / "runs" / "decoder_loop144" / "library.json",
             "gross idle (bare memory)", "idle", "bb144")
