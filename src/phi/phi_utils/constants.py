PHI_ZERO_SHOT_EVAL_PROMPT = '''
Instruct:
You will be given a claim and using commonsense reasoning, you need to respond with SUPPORTS or REFUTES, depending on whether you support or refute the claim.
Claim:{claim}
Is the claim {task_type}? 
Respond with SUPPORTS or REFUTES
Output:
'''

PHI_FEW_SHOT_EVAL_PROMPT = '''
Instruct:
You will be given a claim and using commonsense reasoning, you need to respond with SUPPORTS or REFUTES, depending on whether you support or refute the claim.

Following are some examples:
{examples}

Now Your Turn
Claim:{claim}
Is the claim {task_type}? 
Respond with SUPPORTS or REFUTES
Output:
'''

PHI_ZERO_SHOT_EVIDENCE_PROMPT = '''
Instruct:
You will be given a claim about the domain and You have to generate a detailed facutal evidence about the claim in that domain which should be related to the cliam and may support or refute the claim.

Claim: {claim}
Domian: {domain}
Evidence Output:
'''

PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT = '''
Instruct:
You will be given a claim and evidence for the claim. Using commonsense reasoning, claim and evidence, you need to respond with SUPPORTS or REFUTES, depending on whether you support or refute the claim.
Claim:{claim}
Evidence: {evidence}
Is the claim {task_type}? 
Respond with SUPPORTS or REFUTES
Output:
'''
