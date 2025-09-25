# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /finance

# Copy requirements and install
COPY requirements .
RUN python -m pip install --no-cache-dir -r requirements

# Copy script into container
COPY finance.py .

CMD ["streamlit", "run", "finance.py"]