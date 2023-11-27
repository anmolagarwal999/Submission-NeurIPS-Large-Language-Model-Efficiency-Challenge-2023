# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /lit-gpt/ /submission/
COPY /eval_related_utils/ /submission/eval_related_utils/
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

RUN apt-get update && apt-get install -y git
# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install -r requirements.txt huggingface_hub sentencepiece tokenizers bitsandbytes scipy
# RUN pip install transformers==4.33.1
RUN pip install transformers --upgrade
RUN pip install accelerate
RUN pip install pynvml
RUN huggingface-cli login --token <insert huggingface model token>
RUN python -m pip install optimum
RUN pip install human-eval
RUN pip install colorama

# # some huggingface_hub versions require that the target dir exists
# RUN mkdir -p checkpoints/openlm-research/open_llama_3b
# # get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
# RUN python scripts/download.py --repo_id openlm-research/open_llama_3b
# RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
ENV HUGGINGFACE_TOKEN="<insert huggingface model token>"
ENV HUGGINGFACE_REPO="meta-llama/Llama-2-7b"

# Copy over single file server
# COPY ./main.py /submission/main.py
COPY ./run_inference_endpoint.py /submission/main.py
# COPY ./job_args_config.json /submission/job_args_config.json
# COPY ./generation_args_config.json /submission/generation_args_config.json
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
ENV HUGGINGFACE_HUB_CACHE="/home/anmol/huggingface_hub_cache_dir"
# Run the server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

# python3 run_inference.py --input_file json_prompts_migration.json --model meta-llama/Llama-2-13b-chat-hf --output_dir ./


# docker rm -vf $(docker ps -aq)
# docker rmi -f $(docker images -aq)
# uvicorn  main:app --host  0.0.0.0 --port 8080
