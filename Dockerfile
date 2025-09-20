FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .
EXPOSE 8000
CMD ["python", "-c", "print('clinical-data-platform image OK')"]
