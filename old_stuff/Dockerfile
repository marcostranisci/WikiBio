FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
CMD nvidia-smi
#set up environment
RUN apt update
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
#RUN pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip3 install -r requirements.txt
COPY timebank_tabular/ timebank_tabular/
COPY src/ src/
COPY fine_tuning_output/ fine_tuning_output/

CMD ["python3", "src/kfold_training.py"]

