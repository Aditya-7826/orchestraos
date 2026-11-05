# Image the analytics agent's code runs in. Built once: `make sandbox-image`.
# Started per run with --network none, read-only root, capped memory/CPU (see tools/sandbox.py).
FROM python:3.12-slim

RUN pip install --no-cache-dir "pandas>=2.2" "numpy>=2.0" "matplotlib>=3.9"

# Run as an unprivileged user; matplotlib needs a writable config dir, so point it at /tmp.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox
ENV MPLCONFIGDIR=/tmp HOME=/tmp
WORKDIR /output
