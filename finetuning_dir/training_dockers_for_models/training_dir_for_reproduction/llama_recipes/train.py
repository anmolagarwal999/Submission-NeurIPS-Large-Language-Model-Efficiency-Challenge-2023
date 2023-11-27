import uuid
from alpaca_dataset import InstructionDataset as get_anmol_dataset
from llama_recipes.finetuning import main as finetuning
from huggingface_hub import create_repo
from huggingface_hub import login, HfApi
import sys
from datetime import datetime
import pytz
import os
os.environ["HUGGINGFACE_HUB_CACHE"]="/submission/hf_cache"
os.environ["training_data_path"] = "./pegasus_combined_general_train_dataset.json"
# os.system("pip env list")
################
# os.environ['CUDA_VISIBLE_DEVICES']='2,3'
# os.environ['training_data_path']="/home/t-agarwalan/Desktop/nips_effeciency_challenge/EfficiencyChallenge/data/training_datasets/training_datasets/combined_train_dataset.json"
# os.environ['validation_data_path']="/home/t-agarwalan/Desktop/nips_effeciency_challenge/EfficiencyChallenge/data/training_datasets/training_datasets/combined_valid_dataset.json"
###############

###########
# pytorch related

#########


rand_str = uuid.uuid4()
rand_str = f"model_reproduced_github"
# rand_str = "debug_mistral"

################

# Get the timezone object for GMT+5.5
tz = pytz.timezone('Asia/Kolkata')

# Get the current time in the GMT+5.5 timezone
now = datetime.now(tz)

# Print the current time
print("Starting time is: ", now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
print("RANDOM STRING is: ", rand_str)
###############
os.environ["HUGGINGFACE_TOKEN"] = "<ENTER YOUR HF TOKEN>"
#TODO: for people trying to reproduce our submission, replace `anmolagarwal999` with your own HF username
os.environ["HUGGINGFACE_REPO"] = f"anmolagarwal999/nips_challenge_{rand_str}"
# os.environ['HUGGINGFACE_HUB_CACHE'] = "/home/anmol/huggingface_hub_cache_dir"
print("REPO DECIDED is: ", os.environ["HUGGINGFACE_REPO"])


def main():
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    # TOT_BATCH_SIZE_WANTED = 128
    # EPOCH_BATCH_SIZE = 8

    TOT_BATCH_SIZE_WANTED = int(sys.argv[1])
    EPOCH_BATCH_SIZE = int(sys.argv[2])
    # NUM_EPOCHES = 10

    assert (TOT_BATCH_SIZE_WANTED % EPOCH_BATCH_SIZE == 0)
    TOT_GRAD_ACCUMULATION_STEPS = TOT_BATCH_SIZE_WANTED // EPOCH_BATCH_SIZE

    print("Total gradient accumulation steps are: ", TOT_GRAD_ACCUMULATION_STEPS)

    OUTPUT_DIR = "./"
    # OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models_saved",
    #                           f"{TOT_BATCH_SIZE_WANTED}_{EPOCH_BATCH_SIZE}_{rand_str}")
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models_saved",
                              f"{rand_str}")
    print("OUTPUT dir is: ", OUTPUT_DIR)

    custom_dataset_file_path = os.path.join(__file__)
    print("Custom dataset path is: ", custom_dataset_file_path)
    kwargs = {
        "model_name": "mistralai/Mistral-7B-v0.1",
        # "model_name": "codellama/CodeLlama-7b-hf",
        "use_peft": True,
        "peft_method": "lora",
        "quantization": True,
        "num_epochs": 5,
        "batch_size_training": EPOCH_BATCH_SIZE,
        "gradient_accumulation_steps": TOT_GRAD_ACCUMULATION_STEPS,
        "dataset": "custom_dataset",
        # "dataset": "alpaca_dataset",
        # "custom_dataset.file": "./custom_dataset.py",
        # "./train.py:get_anmol_dataset",
        "custom_dataset.file": f"{custom_dataset_file_path}:get_anmol_dataset",
        # "output_dir": "./random_check",
        "output_dir": OUTPUT_DIR,
    }

    print("Going to begin finetuning")
    finetuning(**kwargs)

    print("Going to use the API to create HF repo")
    # api = HfApi()

    # create_repo(os.environ["HUGGINGFACE_REPO"], private=True, exist_ok=True)

    # api.upload_folder(
    #     # folder_path='./output_dir/',
    #     folder_path=OUTPUT_DIR,
    #     repo_id=os.environ["HUGGINGFACE_REPO"],
    #     repo_type='model',
    # )


if __name__ == "__main__":
    main()


# Get the current time in the GMT+5.5 timezone
now = datetime.now(tz)

# Print the current time
print("Ending time is: ", now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))


