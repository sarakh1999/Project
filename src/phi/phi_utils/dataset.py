import pandas as pd
from torch.utils.data import Dataset
from src.utils.file_utils import load_jsonl
from src.phi.phi_utils.constants import PHI_ZERO_SHOT_EVAL_PROMPT, PHI_FEW_SHOT_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_PROMPT

class PhiPromptDataset(Dataset):
    def __init__(self, annotations_filepath, prompt_type, evidence_filepath = None):
        self.data = load_jsonl(annotations_filepath)
        self.prompt_type = prompt_type

        if evidence_filepath is not None: 
            self.evidence_data = load_jsonl(evidence_filepath)
        else:
            self.evidence_data = None

    def __len__(self):
        return len(self.data)

    ############################################################
    # TODO: Please complete the implementation for the
    # the following transform functions and __getitem__ fn, that you 
    # will use in def __getitem__ to convert a sample into prompt.
    # You can use the templates provided to in the constants.py file

    # End of TODO.
    ##################################################
    
    def __getitem__(self, idx):

        prompt = ""
        
        ##################################################
        # TODO: Please complete the implementation of __getitem__
        # You may use if-else statements to choose the prompt
        # transform as per the prompt type given to you.
        sample = self.data[idx]
        prompt = ""

        if self.prompt_type == "zero_eval":
            claim = sample["claim"]
            task_type = sample.get("task_type", "true/false")  # Assuming 'task_type' key exists
            prompt = PHI_ZERO_SHOT_EVAL_PROMPT.format(claim=claim, task_type=task_type)
        elif self.prompt_type == "few_shot":
            examples = sample.get("examples", "")  # Assuming 'examples' key exists
            claim = sample["claim"]
            task_type = sample.get("task_type", "true/false")
            prompt = PHI_FEW_SHOT_EVAL_PROMPT.format(examples=examples, claim=claim, task_type=task_type)
        elif self.prompt_type == "zero_shot_evidence":
            claim = sample["claim"]
            information = sample.get("information", "")  # Assuming 'information' key exists
            prompt = PHI_ZERO_SHOT_EVIDENCE_PROMPT.format(claim=claim, information=information)
        elif self.prompt_type == "zero_shot_evidence_eval":
            claim = sample["claim"]
            evidence = sample.get("evidence", "")  # Assuming 'evidence' key exists
            task_type = sample.get("task_type", "true/false")
            prompt = PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT.format(claim=claim, evidence=evidence, task_type=task_type)
        else:
            raise ValueError("Unsupported prompt type")

        # End of TODO.
        ##################################################
        
        return prompt
    