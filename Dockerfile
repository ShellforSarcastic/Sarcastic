FROM python:3

ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development 

COPY ./requirements.txt /app/


RUN pip install -r requirements.txt

COPY . /app
EXPOSE 5000

# configure the container to run in an executed manner

CMD ["flask","run" ]