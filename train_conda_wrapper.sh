#!/bin/bash
# train_conda_wrapper.sh
# Conda wrapper for running FLAN-T5 style-transfer training on Orange-Grid.

source /home/svdighe/miniconda3/etc/profile.d/conda.sh
conda activate style_env

cd /home/svdighe/text_style_transfer

echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
python -c "import transformers, sys; print('transformers:', transformers.__version__, 'python:', sys.executable)"

# DEBUG run
# python scripts/train_model.py \
#   --train_file data/processed_data/train.jsonl \
#   --model_name_or_path google/flan-t5-base \
#   --output_dir models/flan_t5_base_style_transfer \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 1 \
#   --max_train_steps 200 \
#   --fp16

# FULL RUN
python scripts/train_model.py \
  --train_file data/processed_data/train.jsonl \
  --model_name_or_path google/flan-t5-base \
  --output_dir models/flan_t5_base_style_transfer \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_train_steps -1 \
  --fp16

exit_code=$?
echo "Training completed with exit code: $exit_code"
exit $exit_code
