# Base image con R 4.4 e Tidyverse pre-installati
FROM rocker/tidyverse:4.4

# Metadati
LABEL maintainer="Flegrei Alert Team"
LABEL description="Ambiente per analisi dati Flegrei Alert 2026 (R 4.4 + Python 3.12)"

# Disabilita prompt interattivi durante la build
ENV DEBIAN_FRONTEND=noninteractive

# Disabilita il blocco PEP 668 per permettere l'installazione di pip system-wide nel container
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Aggiorna il sistema e installa dipendenze per Python 3.12
# Aggiungiamo deadsnakes PPA per garantire la disponibilità di Python 3.12 su base Ubuntu
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    curl \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Configura Python 3.12 come default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Installa pip per Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Installa pacchetti R aggiuntivi richiesti
# tidyverse è già incluso nella base image
# survival e lmtest sono richiesti dalla checklist
RUN install2.r --error \
    survival \
    lmtest \
    changepoint \
    strucchange \
    tseries \
    vars \
    && rm -rf /tmp/downloaded_packages

# Installa pacchetti Python richiesti
# tick: libreria per processi di Hawkes
RUN python3 -m pip install --no-cache-dir \
    "numpy<2" \
    pandas \
    requests \
    "scipy<1.13" \
    scikit-learn \
    setuptools \
    wheel

# Fix per la compilazione di tick: esportiamo CFLAGS con il percorso degli header di numpy
# Eseguiamo l'installazione in un unico comando shell per mantenere la variabile d'ambiente
RUN export CFLAGS="-I$(python3 -c 'import numpy; print(numpy.get_include())')" \
    && python3 -m pip install --no-cache-dir --no-build-isolation tick

# Imposta la directory di lavoro
WORKDIR /app

# Copia il contenuto del progetto nel container
COPY . /app