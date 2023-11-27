# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.policies import fpSixteen, bfSixteen_mixed, get_llama_wrapper
from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from transformers import LlamaTokenizer
from tqdm import tqdm
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
import torch.cuda.nccl as nccl
from datetime import datetime
import pytz
import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging

import numpy as np

import torch
# torch.manual_seed(0)


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

# Converting Bytes to Megabytes


def byte2mb(x):
    return int(x / 2**20)


######################

# Get the timezone object for GMT+5.5
tz = pytz.timezone('Asia/Kolkata')

#####################


def move_data_to_apt_device(train_config_enable_fsdp, batch, local_rank):
    for key in batch.keys():
        if train_config_enable_fsdp:
            batch[key] = batch[key].to(local_rank)
        else:
            batch[key] = batch[key].to('cuda:0')


def log_memory_footprint_details(memtrace, train_config, rank):
    if train_config.enable_fsdp:
        if rank == 0:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(
                f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(
                f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

#TODO: update best val loss properly
#TODO: Call model.train() for consistency
def evaluate_and_save_gracefully(epoch_id, step_id,
                                train_config, 
                                 fsdp_config,
                                 model, 
                                 tokenizer, 
                                 optimizer, 
                                 eval_dataloader, 
                                 local_rank, 
                                 best_val_loss_yet, 
                                 checkpoint_times, val_loss, val_prep,
                                 rank, anmol_val_loss ):
    if not train_config.run_validation:
        return None
    print("$$$$$$ EVALUATING $$$$$$")
    print(f"Evaluating on epoch_id {epoch_id}, step_id: {step_id}")
    # find the evaluation loss and perplexity
    eval_ppl, eval_epoch_loss = evaluation(
        model, train_config, eval_dataloader, local_rank, tokenizer)
    
    checkpoint_start_time = time.perf_counter()
    
    
    print("Eval epoch loss: ", eval_epoch_loss,
            "| best_val_loss: ", best_val_loss_yet)
    is_best = eval_epoch_loss < best_val_loss_yet
    # if train_config.save_model and eval_epoch_loss < best_val_loss:
    if train_config.save_model:
        if train_config.enable_fsdp:
            dist.barrier()
        if train_config.use_peft:
            ###################
            if train_config.enable_fsdp:
                if rank == 0: print(f"we are about to save the PEFT modules")
            else: print(f"we are about to save the PEFT modules")
            ###############
            if is_best: save_dir = os.path.join(train_config.output_dir, f"best_model_yet_epoch_{epoch_id}_{step_id}")
            else: save_dir = os.path.join(train_config.output_dir, f"epoch_{epoch_id}_{step_id}")
            print("Save dir intended: ", save_dir)
            print("Rewriting save dir address")
            save_dir = os.path.join(train_config.output_dir, f"epoch_{epoch_id}_{step_id}")
            print("SAVE DIR is: ", save_dir)
            _now = datetime.now(tz)
            print("Time while saving: ", _now.strftime(
                '%Y-%m-%d %H:%M:%S %Z%z'))
            model.save_pretrained(save_dir)
            ########
            if train_config.enable_fsdp:
                if rank == 0: print(f"PEFT modules are saved in {train_config.output_dir} directory")
            else:print(f"PEFT modules are saved in {train_config.output_dir} directory")
            ###############

        else:
            if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                save_model_checkpoint(
                    model, optimizer, rank, train_config, epoch=epoch_id
                )
            elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                print(
                    " Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                print("=====================================================")

                save_model_and_optimizer_sharded(
                    model, rank, train_config)
                if train_config.save_optimizer:
                    save_model_and_optimizer_sharded(
                        model, rank, train_config, optim=optimizer)
                    print(
                        " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                    print(
                        "=====================================================")

            if not train_config.use_peft and train_config.save_optimizer:
                save_optimizer_checkpoint(
                    model, optimizer, rank, train_config, epoch=epoch_id
                )
                print(
                    " Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                print("=====================================================")
        if train_config.enable_fsdp:
            dist.barrier()
            
    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
    checkpoint_times.append(checkpoint_end_time)
    # print(a)
    # updating the best val loss
    if eval_epoch_loss < best_val_loss_yet:
        best_val_loss_yet = eval_epoch_loss
        str_display = f"best eval loss on epoch {epoch_id} and {step_id} is {best_val_loss_yet}"
        if train_config.enable_fsdp:
            if rank == 0: print(str_display)
        else: print(str_display)
        
    val_loss.append(best_val_loss_yet)
    val_prep.append(eval_ppl)
    anmol_val_loss.append({"epoch_id":epoch_id, "ministep_id":step_id, "eval_epoch_loss":eval_epoch_loss, "best_val_loss_yet":best_val_loss_yet})
    print("$$$$$$ EVALUATION DONE $$$$$$")
    return best_val_loss_yet
    
def is_worthy_ministep(ministep_id, _tot_ministeps, grad_accumulation_steps):
    if (ministep_id + 1) % grad_accumulation_steps == 0 :
        return True
    if ministep_id == _tot_ministeps - 1:
        return True
    return False

def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """

    print("Training config received is: ", train_config)

    #########################
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        # Create a ShardedGradScaler object for gradient scaling with FSDP
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        # print("")
        # Create a GradScaler object for gradient scaling without FSDP
        scaler = torch.cuda.amp.GradScaler()
        print("Anmol: Creating a GradScaler object for gradient scaling without FSDP")
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    # Set the autocast context manager for mixed precision training
    print("Use fp16 has been set to: ", train_config.use_fp16)
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    ########################

    train_prep = []  # to store training perplexity
    train_loss = []  # to store training loss
    val_prep = []  # to store validation perplexity
    val_loss = []  # to store validation loss
    epoch_times = []  # store the time it took to complete each epoch
    checkpoint_times = []  # stores the time it took to take a checkpoint
    anmol_val_loss = []

    ###########################################
    # best validation loss yet
    best_val_loss = float("inf")

    for epoch in range(train_config.num_epochs):
        #############################
        epoch_start_time = time.perf_counter()
        _now = datetime.now(tz)
        print("Epoch starting time: ", _now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
        ##############
        with MemoryTrace() as memtrace:  # track the memory usage

            # TODO:
            model.train()  # tells the model that you are training the model. This helps inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and evaluation.

            total_loss = 0.0

            # Calculate the total number of steps in the epoch
            total_effective_steps_in_epoch = len(
                train_dataloader)//gradient_accumulation_steps
            _tot_ministeps = len(train_dataloader)
            ministeps_save_arr = []
            for _curr_ministep_id in range(_tot_ministeps):
                if is_worthy_ministep(_curr_ministep_id, _tot_ministeps, gradient_accumulation_steps):
                    ministeps_save_arr.append(_curr_ministep_id)
            numElems = os.environ['eval_freq'] if 'eval_freq' in os.environ else 5
            print("NumElems are: ", numElems)
            _idx = set([int(x) for x in np.round(np.linspace(0, len(ministeps_save_arr) - 1, numElems)).astype(int)])
            essential_ministeps = [ministeps_save_arr[y] for y in _idx]
            print("Ministeps save_arr: ", len(ministeps_save_arr), ministeps_save_arr)
            print("Essential ministeps: ",len(essential_ministeps), essential_ministeps)

            epoch_pbar = tqdm(
                colour="blue", desc=f"Training Epoch: {epoch}", total=total_effective_steps_in_epoch, dynamic_ncols=True)

            print("Total ministeps are: ", len(train_dataloader))
            print("grad accumulation steps: ", gradient_accumulation_steps)
            print("Total effective steps in Epoch: ",
                  total_effective_steps_in_epoch)

            # Iterate over the training data
            for ministep_id, batch in enumerate(train_dataloader):
                major_step_id = ministep_id//gradient_accumulation_steps
                # print("Anmol: Going to perform a ministep of training. MINISTEP ID: ", ministep_id, " | major_step_id: ", major_step_id)
                move_data_to_apt_device(
                    train_config.enable_fsdp, batch, local_rank)
                # print("Minibatch device is: ", batch['input_ids'].device)

                with autocast():  # Enable automatic mixed precision if desired
                    assert (torch.is_grad_enabled())  # anmol addition
                    # https://discuss.pytorch.org/t/check-if-model-is-eval-or-train/9395
                    model.train()
                    assert (model.training)
                    # Calculate the loss for the current batch
                    loss = model(**batch).loss

                # we divide the step loss by the total number of ministeps in the step
                # tensor(5.7427, device='cuda:0', grad_fn=<NllLossBackward0>)
                loss = loss / gradient_accumulation_steps
                # Add the current loss to the total loss for the epoch
                total_loss += loss.detach().float()

                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if is_worthy_ministep(ministep_id, _tot_ministeps, gradient_accumulation_steps):
                        # print(f"{ministep_id} is worthy")
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        epoch_pbar.update(1)
                        # $$$$$$$$$$$$$$$$$$$$$$$$$
                        if train_config.run_validation and ministep_id in essential_ministeps:
            
                            _updated_best_val_loss = evaluate_and_save_gracefully(epoch, ministep_id, train_config,
                                                                                fsdp_config,model, tokenizer,optimizer,eval_dataloader,local_rank,best_val_loss,checkpoint_times,val_loss, val_prep, rank,anmol_val_loss )
                            if _updated_best_val_loss is not None:
                                best_val_loss  = _updated_best_val_loss
                        # $$$$$$$$$$$$$$$$$$$$$$$
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if is_worthy_ministep(ministep_id, _tot_ministeps, gradient_accumulation_steps):
                        # print(f"{ministep_id} is worthy")
                        optimizer.step()
                        optimizer.zero_grad()
                        epoch_pbar.update(1)
                        # $$$$$$$$$$$$$$$$$$$$$$$$$
                        if train_config.run_validation and ministep_id in essential_ministeps:
            
                            _updated_best_val_loss = evaluate_and_save_gracefully(epoch, ministep_id, train_config,
                                                                                fsdp_config,model, tokenizer,optimizer,eval_dataloader,local_rank,best_val_loss,checkpoint_times,val_loss, val_prep, rank,anmol_val_loss )
                            if _updated_best_val_loss is not None:
                                best_val_loss  = _updated_best_val_loss
                        # $$$$$$$$$$$$$$$$$$$$$$$

                epoch_pbar.set_description(
                    f"Training Epoch: {epoch}/{train_config.num_epochs}, completed (loss: {loss.detach().float()})")
            epoch_pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        _now = datetime.now(tz)
        print("Epoch ending time: ", _now.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
        print("Validation losses are: ")
        print(*anmol_val_loss, sep='\n')
        print("$$$%%%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        ######################################################
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        log_memory_footprint_details(memtrace, train_config, rank)

        # Update the learning rate as needed
        lr_scheduler.step()
        #############################################################

        # if train_config.run_validation:
            
        #     _updated_best_val_loss = evaluate_and_save_gracefully(epoch, "-1", train_config,
        #                                                           fsdp_config,model, tokenizer,optimizer,eval_dataloader,local_rank,best_val_loss,checkpoint_times,val_loss, val_prep, rank)
        #     if _updated_best_val_loss is not None:
        #         best_val_loss  = _updated_best_val_loss
                

        ###########################################################################################
        ############ END OF EVAL RUN ################################################################
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(
                f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
    #####################################

    #######################################
    print("All epoches are over")
    #######################################################
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(
        checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results = {}
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    assert (not model.training)
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss

                # print("LOSS is: ", loss)
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(
                    preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank == 0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    return eval_ppl, eval_epoch_loss


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            print(
                f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        print(
            f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(
        train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(
        fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, 'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank == 0:
            print(f"training params are saved in {file_name}")
