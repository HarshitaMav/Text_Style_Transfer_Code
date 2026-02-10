#!/bin/bash
# train_conda_wrapper.sh

# Load conda
source /home/svdighe/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate style_env

cd /home/svdighe/text_style_transfer

python -c "import transformers, sys; print('transformers:', transformers.__version__, 'python:', sys.executable)"
# Run evaluation script
python scripts/evaluate_model.py \
  --model_dir models/flan_t5_base_style_transfer \
  --test_file data/processed_data/test.jsonl \
  --batch_size 8 \
  --max_source_length 128 \
  --max_target_length 128 \
  --num_beams 4 \
  --results_dir results

exit_code=$?
echo "Evaluation completed with exit code: $exit_code"
exit $exit_code