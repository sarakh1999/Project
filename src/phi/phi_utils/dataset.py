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
        prompt = ""

        if self.prompt_type == "zero_eval":
            claim = sample["claim"]
            task_type = sample.get("task_type", "true/false")  # Assuming 'task_type' key exists
            prompt = PHI_ZERO_SHOT_EVAL_PROMPT.format(claim=claim, task_type=task_type).strip()

        elif self.prompt_type == "few_shot":
            # Format examples for few-shot
            examples = sample.get("examples", [])
            print(sample)
            print()
            print(examples)
            formatted_examples = "\n".join([
                f"- Claim: {ex['claim']}\n  Veracity: {ex['veracity']}" for ex in examples
            ])
            claim = sample["claim"]
            task_type = sample.get("task_type", "true/false")
            prompt = PHI_FEW_SHOT_EVAL_PROMPT.format(examples=formatted_examples, claim=claim, task_type=task_type).strip()

        elif self.prompt_type == "zero_shot_evidence":
            claim = sample["claim"]
            information = sample.get("information", "")  # Assuming 'information' key exists
            prompt = PHI_ZERO_SHOT_EVIDENCE_PROMPT.format(claim=claim, information=information).strip()

        elif self.prompt_type == "zero_shot_evidence_eval":
            claim = sample["claim"]
            evidence = sample.get("evidence", "")  # Assuming 'evidence' key exists
            task_type = sample.get("task_type", "true/false")
            prompt = PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT.format(claim=claim, evidence=evidence, task_type=task_type).strip()

        else:
            raise ValueError("Unsupported prompt type")

        return prompt
