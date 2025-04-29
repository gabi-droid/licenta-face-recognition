"""
Module for visualizing emotion distributions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_fer_distribution(df_fer):
    """
    Creează graficele pentru distribuția FER2013
    Args:
        df_fer: DataFrame cu datele FER+
    """
    plt.figure(figsize=(15, 7))
    plt.suptitle('Analiza Distribuției Emoțiilor în FER2013', fontsize=16)
    
    # 1. Bar plot pentru distribuția emoțiilor
    plt.subplot(1, 2, 1)
    
    emotion_counts = df_fer['emotion_name'].value_counts()
    colors = ['#FF6B6B', '#DDA0DD', '#4169E1', '#90EE90', 
              '#87CEEB', '#FFD700', '#A0A0A0']
    
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
    plt.title('Distribuția Emoțiilor în FER2013')
    plt.xlabel('Emoții')
    plt.ylabel('Număr de Imagini')
    plt.xticks(rotation=45, ha='right')
    
    # Adăugăm valorile deasupra barelor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # 2. Pie chart pentru distribuția procentuală
    plt.subplot(1, 2, 2)
    plt.pie(emotion_counts.values, labels=emotion_counts.index,
            autopct='%1.1f%%', colors=colors)
    plt.title('Distribuția Procentuală în FER2013')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('fer_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_raf_distribution(df_raf):
    """
    Creează graficele pentru distribuția RAF-DB
    Args:
        df_raf: DataFrame cu datele RAF-DB
    """
    plt.figure(figsize=(15, 7))
    plt.suptitle('Analiza Distribuției Emoțiilor în RAF-DB', fontsize=16)
    
    # 1. Bar plot pentru distribuția emoțiilor
    plt.subplot(1, 2, 1)
    
    emotion_counts = df_raf['emotion_name'].value_counts()
    colors = ['#FFD700', '#4169E1', '#DDA0DD', '#90EE90', 
              '#87CEEB', '#FF6B6B', '#A0A0A0']
    
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
    plt.title('Distribuția Emoțiilor în RAF-DB')
    plt.xlabel('Emoții')
    plt.ylabel('Număr de Imagini')
    plt.xticks(rotation=45, ha='right')
    
    # Adăugăm valorile deasupra barelor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # 2. Pie chart pentru distribuția procentuală
    plt.subplot(1, 2, 2)
    plt.pie(emotion_counts.values, labels=emotion_counts.index,
            autopct='%1.1f%%', colors=colors)
    plt.title('Distribuția Procentuală în RAF-DB')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('raf_db_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_affectnet_distribution(df_affectnet):
    """
    Creează graficele pentru distribuția AffectNet
    Args:
        df_affectnet: DataFrame cu datele AffectNet
    """
    plt.figure(figsize=(15, 7))
    plt.suptitle('Analiza Distribuției Emoțiilor în AffectNet', fontsize=16)
    
    # 1. Bar plot pentru distribuția emoțiilor
    plt.subplot(1, 2, 1)
    
    emotion_counts = df_affectnet['emotion_name'].value_counts()
    colors = ['#A0A0A0', '#90EE90', '#87CEEB', '#FFD700', 
              '#4169E1', '#DDA0DD', '#FF6B6B', '#8B4513']
    
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
    plt.title('Distribuția Emoțiilor în AffectNet')
    plt.xlabel('Emoții')
    plt.ylabel('Număr de Imagini')
    plt.xticks(rotation=45, ha='right')
    
    # Adăugăm valorile deasupra barelor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # 2. Pie chart pentru distribuția procentuală
    plt.subplot(1, 2, 2)
    plt.pie(emotion_counts.values, labels=emotion_counts.index,
            autopct='%1.1f%%', colors=colors)
    plt.title('Distribuția Procentuală în AffectNet')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('affectnet_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_combined_distribution(df_fer, df_raf, df_affectnet=None):
    """
    Creează grafice pentru distribuția combinată a emoțiilor
    Args:
        df_fer: DataFrame cu datele FER+
        df_raf: DataFrame cu datele RAF-DB
        df_affectnet: DataFrame cu datele AffectNet (opțional)
    """
    plt.figure(figsize=(15, 7))
    plt.suptitle('Analiza Distribuției Combinate a Emoțiilor', fontsize=16)
    
    # Combinăm datele pentru analiză
    dfs = [df_fer, df_raf]
    if df_affectnet is not None:
        dfs.append(df_affectnet)
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Distribuția totală a emoțiilor
    emotion_counts = df_combined['emotion_name'].value_counts()
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', 
              '#FF5722', '#795548', '#607D8B', '#8B4513']
    
    # 1. Bar plot pentru distribuția totală
    plt.subplot(1, 2, 1)
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
    plt.title('Distribuția Totală a Emoțiilor')
    plt.xlabel('Emoții')
    plt.ylabel('Număr Total de Imagini')
    plt.xticks(rotation=45, ha='right')
    
    # Adăugăm valorile deasupra barelor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # 2. Pie chart pentru distribuția procentuală
    plt.subplot(1, 2, 2)
    plt.pie(emotion_counts.values, labels=emotion_counts.index,
            autopct='%1.1f%%', colors=colors)
    plt.title('Distribuția Procentuală a Emoțiilor')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('combined_distribution.png', dpi=300, bbox_inches='tight')
    plt.show() 