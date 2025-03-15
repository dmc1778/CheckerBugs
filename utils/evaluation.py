import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, '/home/nima/repository/TensorGuard')
from utils.utils import load_json
from sklearn.manifold import TSNE

def load_data(csv_file):
    df = pd.read_csv(csv_file, sep=',', encoding='utf-8')
    # df = df.iloc[1: , :]
    return df

def calculate_rates(df):
    performance_df = df.groupby(['Temperature', 'Strategy']).apply(lambda g: pd.Series({
        "True Positive Rate (TPR)": ((g['Actual'] == 1) & (g['Predicted'] == 1)).sum() / (g['Actual'] == 1).sum(),
        "False Negative Rate (FNR)": ((g['Actual'] == 1) & (g['Predicted'] == 0)).sum() / (g['Actual'] == 1).sum(),
        "Detection Rate": (g['Predicted'] == 1).sum() / len(g)
    })).reset_index()
    return performance_df

def plot_performance_over_temperature(performance_df):

    performance_df['Temperature'] = pd.to_numeric(performance_df['Temperature'], errors='coerce')

    performance_df = performance_df.dropna(subset=['Temperature'])

 
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))  
    

    strategies = performance_df['Strategy'].unique()

    colors = sns.color_palette("bright", len(strategies))  
    

    for i, strategy in enumerate(strategies):
        strategy_df = performance_df[performance_df['Strategy'] == strategy]
        sns.lineplot(ax=axes[0], data=strategy_df, x="Temperature", y="True Positive Rate (TPR)", 
                     marker="o", label=f"{strategy}", linewidth=2.5, markersize=10, color=colors[i])
    axes[0].set_title("True Positive Rate (TPR) vs. Temperature", fontsize=14)
    axes[0].set_ylabel("TPR", fontsize=12)
    axes[0].grid(True)
    axes[0].legend(title="Strategy", fontsize=10)
    

    axes[0].set_xticks(np.arange(performance_df['Temperature'].min(), performance_df['Temperature'].max() + 0.1, 0.1))

    for i, strategy in enumerate(strategies):
        strategy_df = performance_df[performance_df['Strategy'] == strategy]
        sns.lineplot(ax=axes[1], data=strategy_df, x="Temperature", y="False Negative Rate (FNR)", 
                     marker="s", label=f"{strategy}", linewidth=2.5, markersize=10, color=colors[i])
    axes[1].set_title("False Negative Rate (FNR) vs. Temperature", fontsize=14)
    axes[1].set_ylabel("FNR", fontsize=12)
    axes[1].grid(True)
    axes[1].legend(title="Strategy", fontsize=10)
    

    axes[1].set_xticks(np.arange(performance_df['Temperature'].min(), performance_df['Temperature'].max() + 0.1, 0.1))


    for i, strategy in enumerate(strategies):
        strategy_df = performance_df[performance_df['Strategy'] == strategy]
        sns.lineplot(ax=axes[2], data=strategy_df, x="Temperature", y="Detection Rate", 
                     marker="^", label=f"{strategy}", linewidth=2.5, markersize=10, color=colors[i])
    axes[2].set_title("Detection Rate vs. Temperature", fontsize=14)
    axes[2].set_ylabel("Detection Rate", fontsize=12)
    axes[2].set_xlabel("Temperature", fontsize=12)
    axes[2].grid(True)
    axes[2].legend(title="Strategy", fontsize=10)
    

    axes[2].set_xticks(np.arange(performance_df['Temperature'].min(), performance_df['Temperature'].max() + 0.1, 0.1))
    
    plt.subplots_adjust(hspace=0.5) 
    
    plt.tight_layout()
    plt.show()
    
def line_graphs(df):
    df_zero_shot = df[df["Strategy"] == "few"].copy()

    df_zero_shot["Correct"] = df_zero_shot["Predicted"] == df_zero_shot["Actual"]
    accuracy_df = df_zero_shot.groupby(["ModelName", "Temperature"])["Correct"].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=accuracy_df, x="Temperature", y="Correct", hue="ModelName", marker="o")

    plt.xlabel("Temperature")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Across Temperature (Zero-Shot Strategy)")
    plt.legend(title="Model")
    plt.grid(True)
    plt.show()

def bar_chart_plot(df, strategy):

    df_zero_shot = df[df["Strategy"] == strategy].copy()

    df_zero_shot["Correct"] = df_zero_shot["Predicted"] == df_zero_shot["Actual"]
    accuracy_df = df_zero_shot.groupby(["ModelName", "Temperature"])["Correct"].mean().reset_index()


    plt.figure(figsize=(12, 6))
    sns.barplot(data=accuracy_df, x="Temperature", y="Correct", hue="ModelName")


    plt.xlabel("Temperature")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Across Temperature (Zero-Shot Strategy)")
    plt.legend(title="Model")
    plt.grid(True)
    plt.show()

def bar_chart_plot_all_one(df, libname):

    df_filtered = df[df["Library"] == libname].copy()

    df_filtered["Correct"] = df_filtered["Predicted"] == df_filtered["Actual"]
    accuracy_df = df_filtered.groupby(["ModelName", "Strategy"])["Correct"].mean().reset_index()

    print(accuracy_df)

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=accuracy_df, x="Strategy", y="Correct", hue="ModelName")

    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}', 
            (p.get_x() + p.get_width() / 2, p.get_height()), 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
        )

    plt.xlabel("Strategy")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Across Prompting Strategies (Temperature = 0)")
    plt.legend(title="Model")
    plt.grid(True)


    plt.show()
    
def codeDistribution(code_snippets1, code_snippets2):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    def get_embedding(code):
        with torch.no_grad():
            inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

    embeddings1 = np.array([get_embedding(code) for code in code_snippets1])
    embeddings2 = np.array([get_embedding(code) for code in code_snippets2])


    embeddings = np.concatenate([embeddings1, embeddings2], axis=0)
    labels = [0] * len(embeddings1) + [1] * len(embeddings2)

    # pca = PCA(n_components=3)
    # embeddings_2d = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:len(embeddings1), 0],
                embeddings_2d[:len(embeddings1), 1],
                color='blue', label='PyTorch', alpha=0.5)
    plt.scatter(embeddings_2d[len(embeddings1):, 0],
                embeddings_2d[len(embeddings1):, 1],
                color='red', label='TensorFlow', alpha=0.5)
    plt.xlabel('Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('Component 2', fontsize=14, fontweight='bold')
    # plt.title('CodeBERT embeddings PCA visualization for PyTorch and TensorFlow libraries.')
    plt.legend(prop={'size': 14, 'weight': 'bold'})
    plt.show()

    # this plots histogram for token distributions
    lengths1 = [len(code.split()) for code in code_snippets1]
    lengths2 = [len(code.split()) for code in code_snippets2]

    plt.figure(figsize=(8, 6))
    plt.hist(lengths1, bins=10, alpha=0.5, label='PyTorch')
    plt.hist(lengths2, bins=10, alpha=0.5, label='TensorFlow')
    plt.xlabel('Number of Tokens', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    # plt.title('Distribution of code snippet lengths for PyTorch and TensorFlow libraries.')
    plt.legend(prop={'size': 14, 'weight': 'bold'})
    plt.show()

def plot_model_performance(libname):
    df = load_data(f"output/all_results_filtered.csv")
    bar_chart_plot_all_one(df, libname)

def plot_data_distribution():
    patch_dict = {
        'pytorch': [],
        'tensorflow': []
    }
    for libname,v in patch_dict.items():
        
        data_patch = f"data/taxonomy_data/{libname}_test_data.json"
        data = load_json(data_patch)
    
        for idx, instance in enumerate(data):
            for change in instance['changes']:
                for k, patch in enumerate(change['patches']):
                    code_snippet = patch['hunk_buggy'].replace('-', '')
                    patch_dict[libname].append(code_snippet)
    codeDistribution(patch_dict['pytorch'], patch_dict['tensorflow'])
    
    
def plot_reasoning_accuracy():
    df = load_data(f"output/rootCauseAcc.csv")

    df["RootCauseLabel"] = df["RootCauseLabel"].replace({"o": "0", "O": "0"})
    
    df["RootCauseLabel"] = df["RootCauseLabel"].astype(int)

    ratio_df = (
        df.groupby("ModelName")["RootCauseLabel"]
        .mean()
        .reset_index(name="Fraction_of_1s")
    )
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=ratio_df, x="ModelName", y="Correct", hue="ModelName")

    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}', 
            (p.get_x() + p.get_width() / 2, p.get_height()), 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
        )

    plt.xlabel("Strategy")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Across Prompting Strategies (Temperature = 0)")
    plt.legend(title="Model")
    plt.grid(True)


    plt.show()

if __name__ == '__main__':
    # plot_model_performance('pytorch')
    plot_reasoning_accuracy()

    