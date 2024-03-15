import pandas as pd
from torch.utils.data import Dataset
from src.utils.file_utils import load_jsonl
from src.phi.phi_utils.constants import PHI_ZERO_SHOT_EVAL_PROMPT, PHI_FEW_SHOT_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_PROMPT

class PhiPromptDataset(Dataset):
    def __init__(self, annotations_filepath, prompt_type, evidence_filepath=None):
        self.data = load_jsonl(annotations_filepath)
        self.prompt_type = prompt_type

        if evidence_filepath is not None:
            self.evidence_data = load_jsonl(evidence_filepath)
        else:
            self.evidence_data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.evidence_data is not None:
            evidence =  self.evidence_data[idx]
        prompt = ""

        if self.prompt_type == "zero_eval":
            claim = sample["claim"]
            task_type = sample.get("task_type", "true/false")  # Assuming 'task_type' key exists
            prompt = PHI_ZERO_SHOT_EVAL_PROMPT.format(claim=claim, task_type=task_type).strip()

        # elif self.prompt_type == "few_shot":
        #     # Format examples for few-shot
        #     examples = sample.get("examples", [])
        #     print(sample)
        #     print()
        #     print(examples)
        #     formatted_examples = "\n".join([
        #         f"- Claim: {ex['claim']}\n  Veracity: {ex['veracity']}" for ex in examples
        #     ])
        #     claim = sample["claim"]
        #     task_type = sample.get("task_type", "true/false")
        #     prompt = PHI_FEW_SHOT_EVAL_PROMPT.format(examples=formatted_examples, claim=claim, task_type=task_type).strip()


        elif self.prompt_type == "few_shot":
            examples = sample.get("examples", "")
            claim = sample["claim"]
            task_type = sample.get("task_type", "true/false")
            # Since 'examples' is already a string in the expected format, no need for join or list comprehension
            prompt = PHI_FEW_SHOT_EVAL_PROMPT.format(examples=examples, claim=claim, task_type=task_type).strip()
            
        elif self.prompt_type == "zero_shot_evidence":
            claim = sample["claim"]
            #information = sample.get("information", "")  # Assuming 'information' key exists
            domain = sample.get("domain", "") 
            prompt = PHI_ZERO_SHOT_EVIDENCE_PROMPT.format(claim=claim, domain=domain).strip()

        elif self.prompt_type == "zero_shot_evidence_eval":
            claim = sample["claim"]
            evidence = evidence
            print(evidence) # Assuming 'evidence' key exists
            task_type = sample.get("task_type", "true/false")
            # EVIDENCE_GENERATION_PROMPT = '''
            #     Instruct:
            #     You will be given a claim about the {task_type} and you should generate an evidence about the claim or {task_type} which may support or refutes the factuality of the claim or just have relation to it. 
            #     You have to generate a detailed evidence for the claim given information about it.
                
            #     Claim: {claim}
            #     Information: {information}
            #     Evidence Output:
            #     '''
            # inputs = tokenizer(prompt2, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # inputs = inputs.to("cuda")
            # # Generate outputs
            # output_sequences = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=100)

            # # Decode the outputs
            # output_texts = [tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences]

            
            prompt = PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT.format(claim=claim, evidence=evidence, task_type=task_type).strip()

        else:
            raise ValueError("Unsupported prompt type")

        return prompt
