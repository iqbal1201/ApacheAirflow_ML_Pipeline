FROM apache/airflow:2.8.1
EXPOSE 8081
RUN pip install --upgrade pip
COPY requirements.txt .
RUN python -m pip install -r requirements.txt