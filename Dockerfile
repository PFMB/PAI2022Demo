FROM jupyter/minimal-notebook:python-3.7.12 
#FROM jupyter/scipy-notebook:latestdocker pull jupyter/scipy-notebook
#WORKDIR /pai-demos

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER

ADD demos2022 work
ADD data data
USER root
RUN apt-get update && apt-get install -y libgl1 
RUN apt-get install -y build-essential libopenblas-dev python3-opengl xvfb xauth

RUN mv work demos2022
RUN chmod -R 777 demos2022
USER $NB_UID
# WORKDIR demos2022
