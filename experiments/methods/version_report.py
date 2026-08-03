"""Print exactly what this checkout is and whether it has the features a run needs.

Written for bring-up on clusters whose GitHub mirror lags by days: a bare SHA does not
tell you whether the code you are about to launch contains the decoder variant or the
splitting entry point you intend to use. This checks the FEATURES directly (imports and
attributes, not string matching) so a stale checkout fails loudly here instead of
silently running an older algorithm.

    python experiments/methods/version_report.py
    python experiments/methods/version_report.py --require ghw_deep,paper_splitting
"""
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import platform
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
REPO = pathlib.Path(__file__).resolve().parents[2]


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f"<git failed: {e}>"


def feature_checks():
    """name -> (ok, detail). Each probes the real object, not the source text."""
    out = {}

    try:
        import run_error_model_comparison as rmc
        variants = sorted(rmc.DECODER_VARIANTS)
        out["decoder variants"] = (True, ", ".join(variants))
        cfg = rmc.DECODER_VARIANTS.get("ghw_deep")
        out["ghw_deep"] = (cfg is not None,
                           f"num_sets={cfg.get('num_sets')} pre_iter={cfg.get('pre_iter')}"
                           if cfg else "MISSING")
        out["EMC_CALIB device mode"] = (hasattr(rmc, "CALIB_MODE"),
                                        getattr(rmc, "CALIB_MODE", "MISSING"))
        out["DECODER_P"] = (True, str(getattr(rmc, "DECODER_P", "?")))
    except Exception as e:                                   # noqa: BLE001
        out["run_error_model_comparison"] = (False, f"import failed: {e}")

    try:
        import splitting
        out["paper_splitting (Alg 2/3)"] = (
            hasattr(splitting, "multi_seeded_split_estimate"),
            "multi_seeded_split_estimate" if hasattr(splitting, "multi_seeded_split_estimate")
            else "MISSING")
        out["replica_exchange (tempering)"] = (
            hasattr(splitting, "replica_exchange_estimate"), "")
    except Exception as e:                                   # noqa: BLE001
        out["splitting"] = (False, f"import failed: {e}")

    for mod, label in (("splitting_v2", "splitting v2 (ergodicity-hardened)"),
                       ("portfolio_relay", "PortfolioRelay")):
        try:
            importlib.import_module(mod)
            out[label] = (True, "")
        except Exception as e:                               # noqa: BLE001
            out[label] = (False, str(e).split("\n")[0])

    for rel, label in (
            ("experiments/methods/splitting_paper_72.py", "paper-splitting driver"),
            ("experiments/methods/tech2_subset_seeds.py", "tech2 subset seeding"),
            ("experiments/methods/onset_topup_72.py", "onset top-up"),
            ("container/run_sys_topup.sh", "top-up launcher"),
            ("container/run_emc_device.sh", "device-campaign launcher")):
        out[label] = ((REPO / rel).exists(), rel)

    # onset top-up feature flags that only exist in recent versions
    p = REPO / "experiments/methods/onset_topup_72.py"
    if p.exists():
        src = p.read_text(encoding="utf-8", errors="replace")
        out["top-up: specimen recording"] = ("failure_configs" in src, "")
        out["top-up: ONSET_WEIGHTS env"] = ("ONSET_WEIGHTS" in src, "")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--require", default="",
                    help="comma-separated feature keywords that MUST be present; "
                         "exit 1 if any is missing")
    a = ap.parse_args(argv)

    print("=" * 68)
    print(f"repo      : {REPO}")
    print(f"branch    : {git('rev-parse', '--abbrev-ref', 'HEAD')}")
    print(f"commit    : {git('log', '-1', '--format=%h  %cd  %s', '--date=iso')}")
    print(f"describe  : {git('describe', '--always', '--dirty', '--tags')}")
    dirty = git("status", "--porcelain")
    print(f"worktree  : {'DIRTY (' + str(len(dirty.splitlines())) + ' files)' if dirty else 'clean'}")
    print(f"remotes   : {'; '.join(git('remote', '-v').splitlines()[::2]) or '<none>'}")
    print("-" * 68)
    print(f"python    : {platform.python_version()} ({sys.executable})")
    for mod in ("numpy", "scipy", "stim", "relay_bp"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:10s}: {getattr(m, '__version__', 'installed')}")
        except Exception as e:                               # noqa: BLE001
            print(f"{mod:10s}: MISSING ({str(e).split(chr(10))[0]})")
    print(f"cores     : {os.cpu_count()}   "
          f"RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', '<unset>')}")
    print("-" * 68)

    checks = feature_checks()
    missing = []
    for name, (ok, detail) in checks.items():
        print(f"[{'ok ' if ok else 'NO '}] {name:34s} {detail}")
        if not ok:
            missing.append(name)

    req = [r.strip().lower() for r in a.require.split(",") if r.strip()]
    if req:
        print("-" * 68)
        bad = []
        for r in req:
            hit = [n for n in checks if r in n.lower().replace(" ", "_")
                   or r in n.lower()]
            if not hit or not all(checks[n][0] for n in hit):
                bad.append(r)
        if bad:
            print(f"REQUIRED FEATURES MISSING: {', '.join(bad)}  -> checkout is too old")
            return 1
        print(f"all required features present: {', '.join(req)}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
