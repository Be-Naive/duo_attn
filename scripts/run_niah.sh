model="Llama-2-7B-32K-Instruct"
model_provider=LLaMA
context_lengths_min=2000
pretrained_len=32000
sparsity=0.75
attn_pattern="attn_patterns/Llama-2-7B-32K-Instruct/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"

CUDA_VISIBLE_DEVICES=0 bash scripts/niah.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider
