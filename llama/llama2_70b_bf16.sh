#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NVTE_TORCH_COMPILE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
#export NCCL_IB_SL=1
#export NCCL_IB_TIMEOUT=19
export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=8
export NCCL_NVLS_ENABLE=0
export NVTE_DP_AMAX_REDUCE_INTERVAL=0
#export NVTE_ASYNC_AMAX_REDUCTION=1
export NVTE_FUSED_ATTN=0

source /home/hongbinl/private.cfg # HOME_PATH and WANDB key

EXP_NAME=llama2-70b-bf16-mlm
CHECKPOINT_PATH=$HOME_PATH/fp8/checkpoints/llama2/text/70b-mcore-bf16-perf
TOKENIZER_MODEL=$HOME_PATH/fp8/checkpoints/llama2/tokenizer.model
#DATA_PATH=$HOME_PATH/datasets/pile/llama
#DATA_PATH=$HOME_PATH/datasets/RedPajamaV2
DATA_PATH=$HOME_PATH/datasets/llama/RedPajamaV2/merged
MLM_PATH=$HOME_PATH/fp8/perf/Megatron-LM
TE_PATH=$HOME_PATH/fp8/perf/TransformerEngine

source ${DATA_PATH}/llama_blend.sh

export PYTHONPATH=${MLM_PATH}:${TE_PATH}:${PYTHONPATH}

LLAMA_ARGS="
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --global-batch-size 128 \
    --lr 0.00015 \
    --train-iters 100000 \
    --lr-decay-iters 80000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --disable-bias-linear \
    --no-position-embedding \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --make-vocab-size-divisible-by 128 \
    --norm-epsilon 1e-5 \
"

DATA_ARGS="
    --data-path ${DATA_BLEND} \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 990,8,2 \
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 250 \
    --eval-interval 1000 \
    --eval-iters 20 \
    --timing-log-level 0 \
    --wandb-project mlm_llama_perf \
    --wandb-exp-name ${EXP_NAME} \
    --wandb-save-dir $HOME_PATH/fp8/results/${EXP_NAME} \
    --tensorboard-dir $HOME_PATH/fp8/results/${EXP_NAME} \
    --tensorboard-log-interval 10 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard \
"

FP8_ARGS="
    --fp8-format hybrid \
    --fp8-margin 0 \
    --fp8-interval 1 \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max \
"
OTHER_ARGS="
    --micro-batch-size 1 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --num-layers-per-virtual-pipeline-stage 1 \
    --bf16 \
    --use-flash-attn \
    --sequence-parallel \
    --use-mcore-models \
    --use-distributed-optimizer \
    --transformer-impl transformer_engine \
    --no-check-for-nan-in-loss-and-grad \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tp-comm-overlap \
"
PROFILE_ARGS="
    --profile \
    --profile-step-start 50 \
    --profile-step-end 60 \
"

wandb login ${WANDB}
nsys profile -s none -t nvtx,cuda -o $HOME_PATH/fp8/perf/${EXP_NAME} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
python ${MLM_PATH}/pretrain_gpt.py \
    ${LLAMA_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${OTHER_ARGS} \
    --save ${CHECKPOINT_PATH} \
    --distributed-backend nccl
