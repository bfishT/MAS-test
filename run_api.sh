python -m swebench.inference.run_api \
          --dataset_name_or_path princeton-nlp/SWE-bench_oracle \
          --model_name_or_path gpt-4-1106-preview \
          --model_args "model=glm-5.1,api_base=https://opencode.ai/zen/go/v1" \
          --output_dir ./outputs \
          --instance_ids sympy__sympy-20590
