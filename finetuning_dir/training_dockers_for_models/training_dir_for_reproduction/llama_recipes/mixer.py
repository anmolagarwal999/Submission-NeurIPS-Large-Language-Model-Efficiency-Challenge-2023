import os
import sys
import json

DIR_CONCERNED = "model_reproduced_github"
DIR_WITH_MODELS = os.path.join("models_saved", DIR_CONCERNED)
SEND_MODELS_DIR = "send_models_dir"

all_adapters = os.listdir(DIR_WITH_MODELS)

print("All adapters are: ", all_adapters)

adapters_to_mix_substring = ["_0_3", "_2_243", "_3_243"]

print("Filtering adapters to merge")
all_adapters = list(filter(lambda x: any(
    [y in x for y in adapters_to_mix_substring]), all_adapters))
print("Adapters to merge are: ", all_adapters)

BASE_MODEL = ""

for _adapter in all_adapters:
    print("######################")
    adapter_path = os.path.join(DIR_WITH_MODELS, _adapter)
    print("Adapter path to merge: ", adapter_path)

    unique_merge_substring = DIR_CONCERNED+"_WHOLE_"+_adapter
    path_after_save = os.path.join(SEND_MODELS_DIR, unique_merge_substring)
    print("Path where merged model will be saved is: ", path_after_save)

    # merge the model
    merge_command = f"""python3 merge_adapter_into_base.py --base_model_name="mistralai/Mistral-7B-v0.1" --adapter_model_name="{adapter_path}"  --output_name="{path_after_save}" """

    print("Merge command is: ", merge_command)
    os.system(merge_command)

    # now push to huggingface
    #TODO for people trying to reproduce: Replace "anmolagarwal999" with your own HF username
    push_cmd = f"python3 push_model_to_hub.py {path_after_save} anmolagarwal999/{unique_merge_substring}"
    print("Push command is: ", push_cmd)
    os.system(push_cmd)
    print("Pushed.")
