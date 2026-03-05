FROM python:3.9-slim

# Install system dependencies for R
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install R packages
COPY install_r_packages.R .
RUN Rscript install_r_packages.R || echo "Some R packages may not be available"

# Copy the entire repo
COPY . .

# Unzip Transactions data if needed
RUN cd R_Py-Input_Tables && \
    if [ -f Transactions.csv.zip ] && [ ! -f Transactions.csv ]; then \
        unzip Transactions.csv.zip; \
    fi

# Set entrypoint
ENTRYPOINT ["python", "scripts/deploy.py"]
