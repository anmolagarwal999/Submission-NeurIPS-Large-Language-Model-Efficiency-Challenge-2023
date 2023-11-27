FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

WORKDIR /workspace
COPY . /workspace/

# RUN wget https://gist.githubusercontent.com/mreso/ec65015cbfbd395f0c2adc17147adf1f/raw/41070f1058820b9e89bae885968cc666a7d6aa59/custom_dataset.py

RUN cd /workspace/llama_recipes_external_code && pip install -e .

ENV HUGGINGFACE_TOKEN="<Enter HF token>"

# making executable runnable
RUN chmod +x ./run.sh

# CMD [ "python", "train.py"]
# CMD ["python3", "train.py", "32", "8"]
CMD ["bash", "run.sh"]
