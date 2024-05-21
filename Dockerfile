FROM python:3.12-bullseye

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install numpy plotly sympy dash dash-bootstrap-components dash-daq

WORKDIR /app
COPY ./dispersion_calc.py .

EXPOSE 8080
ENTRYPOINT ["python", "dispersion_calc.py"]