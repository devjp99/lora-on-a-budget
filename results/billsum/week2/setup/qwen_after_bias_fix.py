"""Qwen decoder methods, without changes to the BANKING77 implementation."""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

TARGETS = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
METHODS = ('full_ft','lora','dora','qlora')


def build_model(method, revision, seed=42, base_model='Qwen/Qwen2.5-0.5B', checkpoint=None, device='cuda'):
    if method not in METHODS:
        raise ValueError(method)
    torch.manual_seed(seed)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    kw = dict(revision=revision, torch_dtype=torch.float32 if method=='full_ft' else torch.float16)
    if method == 'qlora':
        kw.update(device_map={'':0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16))
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint if checkpoint and method=='full_ft' else base_model, **kw)
    if method == 'qlora':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        # Normalize frozen 4-bit biases before integrity fingerprints.
        from bitsandbytes.nn import Linear4bit
        for name, layer in model.named_modules():
            if isinstance(layer, Linear4bit) and layer.bias is not None:
                bias = layer.bias
                if bias.requires_grad:
                    raise ValueError(f"Expected frozen QLoRA bias: {name}")
                dtype = layer.compute_dtype
                if dtype is None:
                    raise ValueError(f"Missing QLoRA compute dtype: {name}")
                if bias.dtype != dtype:
                    converted = bias.detach().to(dtype)
                    if not torch.equal(
                        bias.detach().float(), converted.float()
                    ):
                        raise ValueError(
                            f"Bias conversion would change values: {name}"
                        )
                    with torch.no_grad():
                        bias.data = converted
    elif method != 'full_ft':
        model.requires_grad_(False)
    if method != 'full_ft':
        if checkpoint:
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=8,
                lora_alpha=16, lora_dropout=0.05, target_modules=TARGETS,
                bias='none', use_dora=method=='dora'))
        # FP32 trainable adapters allow GradScaler to unscale gradients safely.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    if method != 'qlora':
        model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model


def audit_parameters(model, method):
    trainable = {n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
    if method != 'full_ft':
        if any('lora_' not in n for n in trainable):
            raise ValueError('Unexpected trainable backbone/head tensor')
        expected = 4703232 if method=='dora' else 4399104
        if sum(trainable.values()) != expected:
            raise ValueError(f'Adapter count differs from proposal: {sum(trainable.values())} != {expected}')
    elif any(not p.requires_grad for p in model.parameters()):
        raise ValueError('Full fine-tuning contains frozen tensors')
    return dict(trainable=sum(trainable.values()), tensors=trainable,
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
