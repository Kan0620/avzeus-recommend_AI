FROM python:3.9.3-slim-buster

RUN apt-get update && apt-get upgrade -y

RUN pip3 install torch

RUN pip3 install numpy

RUN pip3 install flask

COPY . .

CMD ["python3","server.py"]
