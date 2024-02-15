import pandas as pd
import matplotlib.pyplot as plt
# Given data
data = {
    "nodes": [64, 64, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512, 256, 256, 256],
    "n_layers": [2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3],
    "env": [
        "FishMoving-v0",
        "FishMovingCNNEncoderContinous-v0",
        "FishMovingVisualServoContinous-v0",
        "FishMoving-v0",
        "FishMovingCNNEncoderContinous-v0",
        "FishMovingVisualServoContinous-v0",
        "FishMoving-v0",
        "FishMovingCNNEncoderContinous-v0",
        "FishMovingVisualServoContinous-v0",
        "FishMoving-v0",
        "FishMovingCNNEncoderContinous-v0",
        "FishMovingVisualServoContinous-v0",
        "FishMoving-v0",
        "FishMovingCNNEncoderContinous-v0",
        "FishMovingVisualServoContinous-v0",
        "FishMoving-v0",
        "FishMovingCNNEncoderContinous-v0",
        "FishMovingVisualServoContinous-v0"
    ],
    "score": [
        {'Training Loss': 0.0008738269},
        {'Training Loss': 0.1603182},
        {'Training Loss': 0.15669496},
        {'Training Loss': 0.00038258158},
        {'Training Loss': 0.16291375},
        {'Training Loss': 0.16306838},
        {'Training Loss': 0.00027262044},
        {'Training Loss': 0.13415168},
        {'Training Loss': 0.120742686},
        {'Training Loss': 0.00020281023},
        {'Training Loss': 0.11921499},
        {'Training Loss': 0.09933434},
        {'Training Loss': 0.00025881178},
        {'Training Loss': 0.11126157},
        {'Training Loss': 0.08905774},
        {'Training Loss': 0.00028729206},
        {'Training Loss': 0.11962191},
        {'Training Loss': 0.1008898}
    ]
}

# Create DataFrame
df = pd.DataFrame(data)
print(df)
# Extract training loss values into a separate column
df['Training Loss'] = df['score'].apply(lambda x: x['Training Loss'])
# Drop the original 'score' column
df.drop('score', axis=1, inplace=True)
# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.tight_layout(pad=6.0)

# Unique environments
envs = df['env'].unique()

for i, env in enumerate(envs):
    # Filter data for the current environment
    env_data = df[df['env'] == env]

    # Plot
    axs[i].scatter(env_data[env_data['n_layers']==2]['nodes'], env_data[env_data['n_layers']==2]['Training Loss'], marker='.', label=f'{env}-2 nodes')
    axs[i].scatter(env_data[env_data['n_layers']==3]['nodes'], env_data[env_data['n_layers']==3]['Training Loss'], marker='*', label=f'{env}-3 nodes')
    axs[i].set_title(f'Training Loss vs Nodes for {env}')
    axs[i].set_xlabel('Nodes')
    axs[i].set_ylabel('Training Loss')
    axs[i].grid(True)
    axs[i].legend()

plt.savefig('training_loss_vs_nodes.pdf')
plt.show()