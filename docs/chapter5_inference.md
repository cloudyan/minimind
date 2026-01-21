# 第五章：推理与评估

`eval_llm.py` 提供了完整的模型推理接口：

```python
def main():
    # ... 参数解析 ...
    
    conversation = []
    model, tokenizer = init_model(args)

    for prompt in prompts:
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 应用聊天模板
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # 生成回复
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
```
