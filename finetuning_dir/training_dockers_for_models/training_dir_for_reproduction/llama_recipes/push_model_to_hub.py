from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from huggingface_hub import login, HfApi
from huggingface_hub import create_repo
import os, sys

os.environ["HUGGINGFACE_TOKEN"] = "<Enter your HF ID>"

# model_path = "/home/anmol/nips_challenge/efficiency_challenge_repo/code/00_starter_repo/neurips_llm_efficiency_challenge/sample-submissions/llama_recipes/models_saved/mistral_weights_by_aj/WHOLE_best_model_yet_epoch_1"
model_path = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)


repo_name = sys.argv[2]

print("Model path is: ", model_path)
print("Repo name is: ", repo_name)
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)