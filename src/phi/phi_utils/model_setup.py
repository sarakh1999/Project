# TODO - import relevant model and tokenizer modules from transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
import icetk
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
# import flash_attn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, pipeline
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, pipeline
from trl import SFTTrainer


# helper function provided to get model info
def get_model_info(model):
    ## model parameter type
    first_param = next(model.parameters())
    print(f"Model parameter dtype: {first_param.dtype}")

    ## which device the model is on
    device_idx = next(model.parameters()).get_device()
    device = torch.cuda.get_device_name(device_idx) if device_idx != -1 else "CPU"
    print(f"Model is currently on device: {device}")

    ## what is the memory footprint 
    print(model.get_memory_footprint())


def model_and_tokenizer_setup(model_id_or_path):
    
    model, tokenizer = None, None

    ##################################################
    # TODO: Please finish the model_and_tokenizer_setup.
    # You need to load the model and tokenizer, which will
    # be later used for inference. To have an optimized
    # version of the model, load it in float16 with flash 
    # attention 2. You also need to load the tokenizer, with
    # left padding, and pad_token should be set to eos_token.
    # Please set the argument trust_remote_code set to True
    # for both model and tokenizer load operation, as 
    # transformer verison is 4.36.2 < 4.37.0
    

    # Set configuration for loading the model with flash attention and float16 precision
    config = AutoConfig.from_pretrained(model_id_or_path, torch_dtype="auto", trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_id_or_path, 
    #                                                            config=config,
    #                                                            torch_dtype=torch.float16,
    #                                                            # use_flash_attention=True,
    #                                                            trust_remote_code=True)


    model_name = "NousResearch/Llama-2-7b-hf"
    new_model = "llama-2-7b-custom"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = True
    output_dir = "results"
    num_train_epochs = 5
    fp16 = True
    bf16 = False
    max_steps = 100
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 25
    logging_steps = 1
    packing = False
    device_map={'':torch.cuda.current_device()}
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,)
    use_flash_attention = False
    
    model = AutoModelForCausalLM.from_pretrained(
                                                model_id,
                                                quantization_config=bnb_config,
                                                use_cache=False,
                                                use_flash_attention_2=use_flash_attention,
                                                device_map="auto",)

    # Load the tokenizer with left padding
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, 
                                               use_fast=True, 
                                               padding_side="left",
                                               trust_remote_code=True)
    
    # Set pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # End of TODO.
    ##################################################

    # get_model_info(model)

    return model, tokenizer
