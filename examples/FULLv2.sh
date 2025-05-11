
set -x
export stage=2
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export steady=FULLv2
# MODEL_PATH=/data/wuxinrui/easyr1_checkpoints/trained/global_step_26/models
MODEL_PATH=/mnt/lyc/wuxinrui/LLaMA-Factory/FULL7B_SFT/outputs_full

# MODEL_PATH=/data/wenhao/long_short_lora/models
# /data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa
# /data/wenhao/long_short_lora/models
#/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247/  # replace it with your local file path

# SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
#  The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""
# SYSTEM_PROMPT="""Return your final response within \\boxed{}. """

python3 -m verl.trainer.main \
    config=examples/FULLv2.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=8