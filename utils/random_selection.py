
import pandas as pd

MODEL_NAMES = ['accounts/fireworks/models/deepseek-r1','gpt-4o-mini-2024-07-18','accounts/fireworks/models/llama-v3p1-405b-instruct', 'accounts/fireworks/models/qwen2p5-72b-instruct', 'accounts/fireworks/models/qwen2p5-coder-32b-instruct', ]

def main():
    data = pd.read_csv("output/all_results_filtered.csv", sep=",", encoding='utf-8')
    filtered_df1 = data[data['Strategy'] != 'basic']

    sampled_dfs = []
    
    for model in MODEL_NAMES:
        filtered_df2 = filtered_df1[(filtered_df1['ModelName'] == model) & (filtered_df1['Predicted'] == 1)]
        sampled_df = filtered_df2.sample(n=50, random_state=42)  # Ensure reproducibility
        sampled_dfs.append(sampled_df)

    final_df = pd.concat(sampled_dfs, ignore_index=True)

    final_df.to_csv("output/sampled_results.csv", index=False, encoding='utf-8')


if __name__ == '__main__':
    main()