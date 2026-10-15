from scripts.nla_generation_capture import SparseCapture, decoder_layer, selected_rows


def test_answer_activations_match_the_original_generated_tokens_and_layer():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(11)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=64,pad_token_id=0,eos_token_id=None)).eval()
    prompt=torch.tensor([[2,5,8,11]])
    collector=SparseCapture(4,{'keyword':{'position':2}},[],lambda tid:'word')
    layer,_=decoder_layer(model,1,2)
    first=model.register_forward_pre_hook(collector.before,with_kwargs=True)
    second=layer.register_forward_hook(collector.after)
    try:
        with torch.inference_mode():
            output=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
                do_sample=False,max_new_tokens=4,pad_token_id=0)
    finally:first.remove();second.remove()
    with torch.inference_mode():
        replay=model(output,output_hidden_states=True,use_cache=False).hidden_states[1]
    for family in ['keyword','answer_content_token_1']:
        vector,pos,tid,_=collector.vectors[family]
        assert tid==int(output[0,pos])
        torch.testing.assert_close(vector,replay[0,pos],atol=1e-5,rtol=1e-4)
    assert collector.vectors['answer_content_token_1'][1]==4
    assert 'answer_content_token_8' not in collector.vectors


def test_selected_keywords_and_prompts_are_train_only_and_without_gold():
    rows=[];splits=[]
    for dataset,name in [('cancer_myth','fold0'),('crepe','official')]:
        train=[]
        for label in (0,1):
            for idx in range(6):
                id=f'{dataset}-{label}-{idx}';train.append(id)
                rows.append(dict(id=id,dataset=dataset,group_id=id,label=label,
                    question=f'How does condition{idx} affect treatment{label}?',reference_answer='SECRET'))
        splits.append(dict(dataset=dataset,name=name,train=train,dev=[],test=[]))
    selected=selected_rows(dict(rows=rows,splits=splits),5)
    assert len(selected)==20
    assert len({r['group_id'] for r in selected})==20
    for row in selected:
        assert 'reference_answer' not in row
        for keyword in row['keywords']:
            a,b=keyword['char_span'];assert row['question'][a:b]==keyword['text']
