"""
Replaces ff layers using MOE. rest all will be averaged
"""
import torch
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_llama import LlamaForCausalLM

model_id = "./moe_model/llama3_full_moe_sensible-rational"
config = {
    "model_type": "llama",
    "num_experts_per_tok": 2,
    "experts": [
        {"expert_name": "expert_sensible", "model_id": "[path to sensibility expert]"},
        {"expert_name": "expert_rational", "model_id": "[path to rationality expert]"},
    ],
    "router_layers": ["gate_proj", "up_proj", "down_proj"],
}


# create checkpoint
expertmerger = ComposeExperts(config, torch_dtype=torch.float16)
expertmerger.compose()
expertmerger.save_checkpoint(model_id)

# load the merged checkkpoint
model = LlamaForCausalLM.from_pretrained(
    model_id
)  # 'gate' / router layers are untrained hence loaded warning would appeare for them
out = model(torch.tensor([[1, 2, 3, 33, 44]]))
print("done")
