import json
import os
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
PROMPT_MESSAGES = [
    {'role': 'system', 'content': 'You are taking a multiple-choice quiz. Answer briefly and end with exactly: Final Answer: <A/B/C/D>.'},
    {'role': 'user', 'content': 'Question: What is linear regression primarily used for?\nOptions:\nA. To determine the value of independent variables\nB. To model relationships between dependent and independent variables\nC. To visualize data\nD. To classify data points'}
]


def load_model(load_in_4bit: bool):
    tok = AutoTokenizer.from_pretrained(MODEL)
    kwargs = dict(device_map='auto')
    if load_in_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    return tok, model


def run_case(name, tok, model, use_chat_template=True, use_terminators=False, repetition_penalty=None, max_new_tokens=64):
    try:
        if use_chat_template:
            text = tok.apply_chat_template(PROMPT_MESSAGES, tokenize=False, add_generation_prompt=True)
        else:
            text = (
                'Question: What is linear regression primarily used for?\n'
                'Options:\n'
                'A. To determine the value of independent variables\n'
                'B. To model relationships between dependent and independent variables\n'
                'C. To visualize data\n'
                'D. To classify data points\n\n'
                'Answer with exactly: Final Answer: <A/B/C/D>.'
            )
        inputs = tok([text], return_tensors='pt', padding=True).to(model.device)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        if use_terminators:
            eos_ids = [tok.eos_token_id]
            im_end = tok.convert_tokens_to_ids('<|im_end|>')
            if im_end is not None and im_end not in eos_ids:
                eos_ids.append(im_end)
            gen_kwargs['eos_token_id'] = eos_ids
        if repetition_penalty is not None:
            gen_kwargs['repetition_penalty'] = repetition_penalty
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        decoded = tok.decode(new_tokens, skip_special_tokens=True)
        return {'name': name, 'ok': True, 'output': decoded}
    except Exception as e:
        return {'name': name, 'ok': False, 'error': repr(e), 'traceback': traceback.format_exc()}


def main():
    results = []
    for load_in_4bit in [True, False]:
        label = '4bit' if load_in_4bit else 'fp16'
        try:
            tok, model = load_model(load_in_4bit)
            results.append({'phase': label, 'loaded': True, 'device': str(model.device)})
            cases = [
                run_case(f'{label}_chat_basic', tok, model, use_chat_template=True, use_terminators=False, repetition_penalty=None),
                run_case(f'{label}_chat_terminators', tok, model, use_chat_template=True, use_terminators=True, repetition_penalty=None),
                run_case(f'{label}_chat_terminators_rep', tok, model, use_chat_template=True, use_terminators=True, repetition_penalty=1.05),
                run_case(f'{label}_plain_basic', tok, model, use_chat_template=False, use_terminators=False, repetition_penalty=None),
            ]
            results.extend(cases)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            results.append({'phase': label, 'loaded': False, 'error': repr(e), 'traceback': traceback.format_exc()})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
