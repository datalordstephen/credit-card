# Use the exact Python version from .python-version
FROM python:3.11.9-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /code

# add venv to path so that installed packages are found globally
ENV PATH="/code/.venv/bin:$PATH"

# Copy uv project files
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
# COPY "api/" "models/" "src" ./

COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/

# expose
EXPOSE 8000

# Run your application
ENTRYPOINT ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]