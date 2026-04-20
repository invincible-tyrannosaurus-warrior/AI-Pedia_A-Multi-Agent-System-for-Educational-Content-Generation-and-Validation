import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
print(f'Loading tokenizer: {MODEL_NAME}')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map='auto',
    attn_implementation='eager',
)

prompt = "<|system|>\nYou are taking a multiple-choice quiz. Answer briefly and end with exactly: Final Answer: <A/B/C/D>.\n<|user|>\nQuestion: What is linear regression primarily used for?\nOptions:\nA. To determine the value of independent variables\nB. To model relationships between dependent and independent variables\nC. To visualize data\nD. To classify data points\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
print('\n===== MODEL OUTPUT =====')
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
print('========================')
