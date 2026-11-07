FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    R_ENVIRON_USER=/dev/null \
    R_PROFILE_USER=/dev/null

# The release executor is one immutable Python/R environment. Debian package
# resolution uses the dated snapshots recorded by the pinned base image.
RUN sed -i \
        -e 's|URIs: http://deb.debian.org/debian$|URIs: http://snapshot.debian.org/archive/debian/20260623T000000Z|' \
        -e 's|URIs: http://deb.debian.org/debian-security$|URIs: http://snapshot.debian.org/archive/debian-security/20260623T000000Z|' \
        /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99snapshot

# Direct R
# dependencies are version-pinned; the runtime probe records the complete
# transitive Debian, Python, and R package inventories. The local image digest
# is recorded separately so execution remains bound to the attested image.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        r-base-core=4.5.0-3 \
        r-recommended=4.5.0-3 \
        r-cran-jsonlite=1.9.1+dfsg-1 \
        r-cran-sandwich=3.1-1-1 \
        r-cran-survey=4.4-2-2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
COPY wheels /tmp/wheels
COPY r-package /tmp/trialagentbenchR
COPY parquet_to_csv.py /opt/trialagentbench/parquet_to_csv.py

# Participant code runs as a non-root user and must read the parquet bridge
# whatever the checkout umask. COPY --chmod would also strip search permission
# from the created /opt/trialagentbench directory, so the mode is set here.
RUN cd /tmp \
    && pip install --no-cache-dir --require-hashes -r requirements.lock \
    && R CMD INSTALL --no-byte-compile trialagentbenchR \
    && rm -rf requirements.lock wheels trialagentbenchR \
    && chmod 0644 /opt/trialagentbench/parquet_to_csv.py

WORKDIR /workspace
