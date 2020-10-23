# set base image
FROM python:3.8-slim-buster

FROM osgeo/gdal:ubuntu-small-latest

# install dependencies
#COPY requirements.txt .
#RUN pip install -r requirements.txt

# set the working directory in the container
WORKDIR /code

COPY ./code /code

CMD [ "python", "setup.py build_ext --build-lib ."]
