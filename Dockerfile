# set base image
FROM python:3.8-slim-buster AS compile-image

RUN apt-get update \
  && apt-get -y install gcc

#RUN apk add --no-cache --virtual .build-deps gcc musl-dev \
# && pip install cython==29.21 \
# && apk del .build-deps

# install dependencies
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# copy the python code to the compile image
WORKDIR /code
COPY ./code .

# compile the cython files
RUN python setup.py build_ext --build-lib .

# set the second image
FROM osgeo/gdal:ubuntu-small-latest

# copy the necessary files from the compile image
COPY --from=compile-image /root/.local /root/.local
COPY --from=compile-image /code /code

# make sure the scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# set the directory for host-shared data
VOLUME /data

# set the working directory in the container
WORKDIR /code

# copy the training data to the image
COPY ./training_datasets ./training_datasets
