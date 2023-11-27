#### Steps for Reproduction

Submission we would like to reproduce: submission_1

###### Steps to follow:
```bash

# cmd 1
cd llama_recipes/

# cmd 2
docker build -f ./Dockerfile -t llama_recipes_train .

# cmd 3
docker run --gpus "device=0" \
     -v ./models_saved:/workspace/models_saved  \
     -v ./training_logs:/workspace/training_logs \
     --rm -ti llama_recipes_train

```

## More details:
```txt
As a result of cmd_3, the relevant adapters would be saved in `models_saved/model_to_reproduce`.

The adapters we use are: `epoch_2_243` and `epoch_3_243`.

These adapters need to be combined with the base model and pushed to huggingface so that they can be used by the eval dockerfile.

For this purpose, the files: `push_model_to_hub.py` and `merge_adapter_into_base.py` can be used.
```
EDIT: I have now automated and integrated the above process in the Dockerfile itself.
The models used can be found as `model_reproduced_github_WHOLE_2_243` and `model_reproduced_github_WHOLE_3_243`.
These models need to be substituted in `submission_1` evaluation script.


##### Other details about base models
We use Mistral-7B as our base model.

##### Details about the dataset we are using
The dataset we are using is: `pegasus_combined_general_train_dataset.json` which we had uploaded for the community here: https://huggingface.co/datasets/ajdesh2000/pegasus_combined_general_train_dataset . 

The dataset file is a mixture of the training splits (the training splits as used by HELM) for the following datasets: TruthfulQA + BBQ + BigBench + GSM8k + MMLU + CNN/DM. The individual datasets used for mixing can be found here: https://drive.google.com/drive/folders/17yX-wE8AbBZ9aGmW1xWqxcLyLifOKkYJ?usp=sharing . 
The code etc used for extracting splits from HELM and mixing them can be found here: `training_dockers_for_models/training_dir_for_reproduction/dataset_construction_code`.



