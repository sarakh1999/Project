import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.file_utils import load_jsonl, dump_jsonl
from src.phi.phi_utils.dataset import PhiPromptDataset
from src.phi.phi_utils.model_setup import model_and_tokenizer_setup

torch.set_default_device("cuda")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id_or_path', type=str, required=True)
    parser.add_argument('--annotations_filepath', type=str, required=True)
    parser.add_argument('--output_filepath', type=str, required=True)
    parser.add_argument('--prompt_type', type=str, required=True)
    parser.add_argument('--evidence_filepath', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)


    args = parser.parse_args()
    return args 

def batch_prompt(model, tokenizer, annotations_filepath, output_filepath, prompt_type, evidence_filepath, batch_size):
    
    prompt_dataset = PhiPromptDataset(annotations_filepath, prompt_type, evidence_filepath=evidence_filepath)
    prompt_dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False)

    output_data = []
    for batch in tqdm(prompt_dataloader):
        output_texts = []
        ##################################################
        # TODO: Please complete the implementation of this 
        # for loop. You need to tokenize a batch of samples
        # generate outputs for that batch, and then decode
        # the outputs back to regular text. The output_texts
        # variable used in the for loop following this TODO
        # is what should be the output of the program snippet 
        # within TODO
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to("cuda")

        # Generate outputs
        output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150)

        # Decode the outputs
        output_texts = [tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences]
        #print(output_texts)
        # End of TODO.
        ##################################################
        if evidence_filepath == None:
            for output_text in output_texts:
                final_response = output_text.split("Evidence Output:")[-1].split("<|endoftext|>")[0]
                tmp_response = final_response.lower()
               
                output_data.append(tmp_response)

        else:
            for output_text in output_texts:
                final_response = output_text.split("Output:")[-1].split("<|endoftext|>")[0]
                tmp_response = final_response.lower()
                print(tmp_response)
                if "ref" in tmp_response or "false" in tmp_response:
                    predicted_label = "REFUTES"
                else:
                    predicted_label = "SUPPORTS"
    
                output_data.append(predicted_label)
    if evidence_filepath == None:
        dump_jsonl(output_data, output_filepath)

    else:
        # print("HIII")
        # dump_jsonl(output_data, output_filepath)
        # print(len(output_data))
    
        # Save the predictions to a text file
        output_filepath = "fof_predictions.txt"  # Set the output filepath to the required filename
        with open(output_filepath, "w") as f:
           for prediction in output_data:
               f.write(f"{prediction}\n")

def main(args):
    model, tokenizer = model_and_tokenizer_setup(args.model_id_or_path)
    batch_prompt(model=model, tokenizer=tokenizer, annotations_filepath=args.annotations_filepath, output_filepath=args.output_filepath, prompt_type=args.prompt_type, evidence_filepath=args.evidence_filepath, batch_size=args.batch_size)

if __name__ == "__main__":
    args = parse_args()
    main(args)
