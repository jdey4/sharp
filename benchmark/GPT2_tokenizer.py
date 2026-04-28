#%%
from transformers import AutoTokenizer, AutoModel
import torch
# %%
tok = AutoTokenizer.from_pretrained("gpt2")

print("vocab size:", tok.vocab_size)
print("special tokens:", tok.special_tokens_map)
# %%
text = "SHARP learns long-range structure in PG-19."

ids = tok.encode(text)
tokens = tok.convert_ids_to_tokens(ids)

print(ids)
print(tokens)
print(tok.decode(ids))
# %%
for s in ["charge", " charge", "Charge", " Charge"]:
    ids = tok.encode(s)
    print(repr(s), ids, tok.convert_ids_to_tokens(ids))
# %%
text = "This is book one. This is book two."
ids = tok.encode(text)

x = torch.tensor(ids[:-1], dtype=torch.long)
y = torch.tensor(ids[1:], dtype=torch.long)

print(x.shape, y.shape)
# %%
gpt2 = AutoModel.from_pretrained("gpt2")
E = gpt2.get_input_embeddings().weight.detach()

print(E.shape)
# %%
ids_tensor = torch.tensor(ids).unsqueeze(0)  # [1, T]

with torch.no_grad():
    emb = gpt2.get_input_embeddings()(ids_tensor)

print(emb.shape)  # [1, T, 768]
# %%
