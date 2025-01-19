import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import the data
df = pd.read_csv('medical_examination.csv')

# 2: Add an overweight column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3: Normalize data
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4: Function to draw the Categorical Plot
def draw_cat_plot():
    # 5: Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6: Group and reformat the data to split it by cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})
    
    # 7: Create the catplot
    fig = sns.catplot(data=df_cat, 
                      x='variable', 
                      y='total', 
                      hue='value', 
                      col='cardio', 
                      kind='bar', 
                      height=5, 
                      aspect=1).fig
    
    # 8: Save the figure
    fig.savefig('catplot.png')
    return fig

# 10: Function to draw the Heat Map
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 15: Draw the heatmap
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt='.1f', 
                cmap='coolwarm', 
                vmax=0.3, 
                vmin=-0.1, 
                center=0, 
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": 0.5})
    
    # 16: Save the figure
    fig.savefig('heatmap.png')
    return fig
    