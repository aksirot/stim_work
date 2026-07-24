# Prebuilt podman image (split to fit GitHub's 100 MB/file limit)

The fail-fast QEC container image — gzipped and split into <100 MB parts because the compute
service can't reach Docker Hub to `podman build` the base layer itself. Built and
preflight-validated on 2026-07-24 (12/12 checks, including the `relay_bp` decode round-trip).

## Use it on the compute service

```bash
# from the repo root, after fetching this branch (see below)
cat qec_image_parts/qec_image.tar.gz.part* > qec_image.tar.gz   # reassemble (byte-identical)
podman load -i qec_image.tar.gz                                  # -> localhost/stim-work-qec:latest
podman run --rm stim-work-qec:latest python test.py --no-slurm   # confirm: 12/12
```

## Fetching just this branch (if your checkout is on `main`)

```bash
git fetch origin qec-image
git checkout origin/qec-image -- qec_image_parts/     # grab only the parts, stay on main
# ...then the cat + podman load above
```

This `qec-image` branch exists ONLY to ship the binary — **do not merge it into `main`** (it
would bloat history). Delete it once the image is loaded on the cluster:
`git push origin --delete qec-image`.
