FROM continuumio/miniconda3
WORKDIR /app
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY . .
RUN conda env create -f tf_env.yml
EXPOSE 5000
SHELL ["conda", "run", "--no-capture-output", "-n", "tf_env", "/bin/bash", "-c"]
EXPOSE 5003
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tf_env", "python3", "api_server_side.py"]
CMD ["sh", "-c", "python api_server_side.py]
