
import torch

EMBEDDING_PATH = "path to embeddings.pt"

ldm_embed = torch.load(EMBEDDING_PATH)  # no weights_only for old torch

string_to_param = ldm_embed["string_to_param"]

params = {name: param for name, param in string_to_param.named_parameters()}
token  = list(params.keys())[0]
vector = params[token]

print(f"Token  : '{token}'")
print(f"Shape  : {vector.shape}")  # should be [1, 768]
assert vector.shape[-1] == 768, f"Wrong dim {vector.shape} — wrong base model used!"

# Save raw tensor only
torch.save({"<shadowobject>": vector.detach()}, "shadow_sd15_diffusers.pt")
print("Saved → shadow_sd15_diffusers.pt")

