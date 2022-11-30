FROM pytorchlightning/pytorch_lightning

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN pip install pandas
RUN pip install pytorch-lightning
RUN pip install torchmetrics


RUN apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common -y
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
#RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt install r-base -y
#RUN apt install build-essential -y
RUN R -e "install.packages('dplyr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('neurobase',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('oro.nifti',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('plyr',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('optparse',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN pip install python-dotenv

RUN mkdir -p /app
WORKDIR /app

COPY . /app

CMD ["flask", "run"]