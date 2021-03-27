FROM continuumio/miniconda3
WORKDIR /multidex
COPY environment.yml .
RUN conda env create -f environment.yml
COPY /multidex .
EXPOSE 10001
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "multidex", "python", "multidex_dock.py"]

