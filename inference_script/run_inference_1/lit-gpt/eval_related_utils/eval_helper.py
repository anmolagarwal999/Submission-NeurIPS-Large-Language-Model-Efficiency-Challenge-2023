# this is for GPU related stuff
import pynvml
pynvml.nvmlInit()
import datetime
import pytz, re
import subprocess

# huggingface related
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

import torch


def print_time():
    # Define the desired timezone using the GMT offset and minutes
    desired_timezone = pytz.timezone('Asia/Calcutta')

    # Get the current time in UTC
    current_time_utc = datetime.datetime.utcnow()

    # Convert the UTC time to the desired timezone
    current_time_in_desired_timezone = current_time_utc.replace(tzinfo=pytz.utc).astimezone(desired_timezone)

    # Format and print the time
    formatted_time = current_time_in_desired_timezone.strftime('%Y-%m-%d %H:%M:%S %Z%z')
    print(f"Time in GMT+5.5 is: {formatted_time}", flush=True)

def print_gpu_memory_stats():
    print("________________")
    def get_gpu_utilization(handle_now):
        info = pynvml.nvmlDeviceGetUtilizationRates(handle_now)
        return info.gpu
    
    def get_gpu_count():
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
            gpu_count = len(output.strip().split(b"\n"))
            return gpu_count
        except subprocess.CalledProcessError:
            return 0
    #############################################
    num_gpus = get_gpu_count()
    print_time()
    print("Number of available GPUs:", num_gpus)
    handles = [pynvml.nvmlDeviceGetHandleByIndex(curr_gpu_id) for curr_gpu_id in range(num_gpus)]
    TOT_MEM_CONSUMED = 0
    for curr_gpu_id, handle in enumerate(handles):
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used = info.used / 1024 ** 2  # Convert bytes to megabytes
        memory_total = info.total / 1024 ** 2  # Convert bytes to megabytes
        # memory_free = memory_total - memory_used
        frac_used = memory_used/memory_total
        gpu_utilization = get_gpu_utilization(handle)
        gpu_dict = dict()
        gpu_dict = {"curr_gpu":curr_gpu_id,
                    "volatile_gpu_utils": gpu_utilization,
                    "total_gpu_mem":memory_total, 
                    "used_gpu_mem":memory_used, 
                    "frac_used_gpu_mem":frac_used
        }
        TOT_MEM_CONSUMED += memory_used
        print(gpu_dict)
    print("Total memory consumed is: ", TOT_MEM_CONSUMED)
    print("________________")



def do_huggingface_login():
    login(token="<insert huggingface token>")
    print("Huggingface login done")
    
    
def get_mistral_model_tokenizer(model_path, device_sent='auto'):
    assert("mistral" in model_path)
    print("Model to load is: ", model_path)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer has been loaded.")
    
    # load the model
    model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map=device_sent,
                )
    print("After loading, model has taken the memory")
    print_gpu_memory_stats()
    
    # dealing with pad tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Config pad token id is: ", model.config.pad_token_id )
    
    # setting model in eval mode
    model.eval()
    
    return model, tokenizer

def get_qwen_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer has been loaded.")

    model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                return_dict=True,
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                device_map="auto",
                                )
    print("After loading, model has taken the memory")
    print_gpu_memory_stats()
    
    # setting model in eval mode
    model.eval()
    
    return model, tokenizer
    
#################################
from enum import Enum

class QueryCategory(Enum):
    mmlu_query = 1
    bbq_query = 2
    tqa_query = 3
    cnn_query = 4
    other_category  = 5
    gsm_category  = 6
    bigbench_category  = 7
    

def find_category(txt):
    ans = QueryCategory.other_category
    
    ###########
    _cnn_str = "article in 3 sentences"
    if _cnn_str in txt:
        ans = QueryCategory.cnn_query
        return ans
    
    #############
    _mmlu_str = "multiple choice questions (with answers) about"
    if _mmlu_str in txt:
        ans = QueryCategory.mmlu_query
        return ans
    
    ########
    _tqa_str = "Who is the current president of the United States?"
    if _tqa_str in txt:
        ans = QueryCategory.tqa_query
        return ans
    
    ########
    _bbq_str = "Alejandra Rodriguez and Luis Rios got into a heated debate"
    if _bbq_str in txt:
        ans = QueryCategory.bbq_query
        return ans
    
    _bbq_str = "multiple choice questions (with answers)."
    if _bbq_str in txt:
        ans = QueryCategory.bbq_query
        return ans
    
    ##############
    _gsm_str = "Given a mathematics problem, determine the answer"
    if _gsm_str in txt:
        ans = QueryCategory.gsm_category
        return ans
    
    pattern = r"The answer is \d+."
    match = re.search(pattern, txt)
    if match:
        ans = QueryCategory.gsm_category
        return ans

    
    _gsm_str = "Daniel has a collection of 346 video games"
    if _gsm_str in txt:
        ans = QueryCategory.gsm_category
        return ans
    
    
    check_txt = [
        "Determine whether the following pairs of sentences embody an entailment relation or",
        "answer each of the following questions about causation",
        "be a joke (with dark humor)",
        "What movie does this emoji describe",
        "causal, correlative, or neutral relation between two even",
        "Canada was the most recent person to turn the lights out in their hom",
        "objects arranged in a fixed order",
        "Which statement is sarcastic?",
        "Context: Today James"
    ]
    
    for _elem in check_txt:
        if _elem in txt:
            ans = QueryCategory.bigbench_category
            return ans
    if txt.startswith("Context:"):
        ans = QueryCategory.bigbench_category
        return ans
        
    ################
    if "Question:" in txt and "Answer:" in txt:
        ans = QueryCategory.tqa_query
        return ans
    #########
    
    return ans


class FetchModel:
    def __init__(self, eval_map):
        self.eval_map = eval_map
        self.eval_map = dict([(str(x), y) for (x,y) in eval_map.items()])
        assert(len(QueryCategory) == len(self.eval_map))
        
        # checking whether mistral or llama or qwen
        for curr_val in self.eval_map.values():
            assert("mistral" in curr_val) 
        
        ##############################
        self.GPU_DEVICE = str(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.CPU_DEVICE = 'cpu'

        
        ###########################
            
        # load all models to cpu
        self.load_models()        
        
        # print GPUs of all models
        self.print_model_statuses()
        
    
    def print_model_statuses(self):
        print("#####")
        print("Model statuses are:")
        for model_path in  self.fetch_unique_models():
            device_there = str(self.models_needed[model_path]['model'].device)
            print(device_there, model_path)
        print("#####")
                
    def fetch_unique_models(self):
        return set(list(self.eval_map.values()))

    def load_models(self):
        self.models_needed = dict()
        for model_path in  self.fetch_unique_models():
            self.models_needed[model_path] = {"model":"", "tokenizer":""}
            assert("mistral" in model_path)
            
            # load the model to cpu
            self.models_needed[model_path]['model'], self.models_needed[model_path]['tokenizer'] = get_mistral_model_tokenizer(model_path, self.CPU_DEVICE)
    
    def fetch_model_on_gpu(self):
        model_on_gpu = []
        for model_path in  self.fetch_unique_models():
            device_there = str(self.models_needed[model_path]['model'].device)
            if "cuda" in device_there:
                model_on_gpu.append(model_path)
        assert(len(model_on_gpu)<2)
        if len(model_on_gpu)==0:
            return ""
        else:
            return model_on_gpu[0]
        
    def load_to_gpu(self, model_path):
        print("Loading to GPU")
        model_device  = str(self.models_needed[model_path]['model'].device)
        assert(model_device in [self.CPU_DEVICE, self.GPU_DEVICE])
        assert(model_device in self.CPU_DEVICE)
        print("Loading ", model_path, " to GPU")
        self.models_needed[model_path]['model'] = self.models_needed[model_path]['model'].to(self.GPU_DEVICE)
        
        
    def switch_to_cpu(self, model_path):
        print("Loading to CPU")
        model_device  = str(self.models_needed[model_path]['model'].device)
        print("Model device is: ", model_device)
        assert(model_device in [self.CPU_DEVICE, self.GPU_DEVICE])
        assert(model_device in self.GPU_DEVICE)
        print("Switching ", model_path, " to CPU")
        self.models_needed[model_path]['model'] = self.models_needed[model_path]['model'].to(self.CPU_DEVICE)
    
    def ensure_model_on_gpu(self, model_path):
        assert(model_path in self.fetch_unique_models())
        curr_model_on_gpu = self.fetch_model_on_gpu()
        if curr_model_on_gpu == model_path:
            print("Model already on GPU")
            return
        
        # if a model is there on GPU, then it is not the one we want.
        if curr_model_on_gpu !="":
            self.switch_to_cpu(curr_model_on_gpu)
            torch.cuda.empty_cache()
            self.load_to_gpu(model_path)
        else:
            torch.cuda.empty_cache()
            self.load_to_gpu(model_path)
            
                        
    def fetch_model_tokenizer_for_query(self, query_txt):
        query_type = find_category(query_txt)
        # print("Query type is: ", query_type)
        query_type = str(query_type)
        print("QUERY type is: ", query_type)
        assert(query_type in self.eval_map)
        model_needed = self.eval_map[query_type]
        self.ensure_model_on_gpu(model_needed)
        self.print_model_statuses()
        return self.models_needed[model_needed]['model'], self.models_needed[model_needed]['tokenizer']