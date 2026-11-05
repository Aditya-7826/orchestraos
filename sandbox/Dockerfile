# Image the analytics agent's code runs in. Built once: `make sandbox-image`.
# Started per run with --network none, read-only root, capped memory/CPU (see tools/sandbox.py).
FROM python:3.12-slim

# pandas 2.x on purpose: models are trained mostly on pandas 2 code, and pandas 3 removed
# APIs they still write (e.g. the "M" month-end alias), which made generated code fail.
RUN pip install --no-cache-dir "pandas>=2.2,<3" "numpy>=2.0,<3" "matplotlib>=3.9,<4"

# Run as an unprivileged user; matplotlib needs a writable config dir, so point it at /tmp.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox
ENV MPLCONFIGDIR=/tmp HOME=/tmp
WORKDIR /output
