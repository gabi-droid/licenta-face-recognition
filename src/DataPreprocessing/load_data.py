import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Structura de bază
def load_data(path_to_ferplus):
    """
    Incarca datele din fer2013new.csv și face procesarea inițială
    Args:
        path_to_ferplus: calea catre fișierul fer2013new.csv
    Returns:
        DataFrame cu datele procesate
    """
    df = pd.read_csv(path_to_ferplus)
    
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    # Adăugăm coloane pentru totalul voturilor și emoția dominantă
    df['total_votes'] = df[emotion_columns + ['unknown', 'NF']].sum(axis=1)
    
    # Calculăm procentajele pentru fiecare emoție
    for col in emotion_columns + ['unknown', 'NF']:
        df[f'{col}_percent'] = df[col] / df['total_votes'] * 100
        
    return df

def get_emotion_columns():
    """
    Returnează lista de coloane pentru emoții (fără usage, NF, unknown)
    """
    return ['neutral', 'happiness', 'surprise', 'sadness', 
            'anger', 'disgust', 'fear', 'contempt']

def print_basic_stats(df):
    """
    Afișează statistici de bază despre dataset
    """
    print("=== Statistici de bază ===")
    print(f"Număr total de imagini: {len(df)}")
    print("\nDistribuția pe seturi:")
    print(df['Usage'].value_counts())
    
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    print("\nNumăr total de voturi per emoție:")
    total_votes = df[emotion_columns + ['unknown', 'NF']].sum()
    print(total_votes)
    
    print("\nProcent din totalul voturilor:")
    print((total_votes / total_votes.sum() * 100).round(2))

def analyze_pure_votes(df):
    """
    Creează și afișează toate vizualizările pentru voturile pure
    """
    # Folosim un stil standard în loc de seaborn
    plt.style.use('default')
    
    # Creăm o figură mare pentru toate subplot-urile
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Bar plot pentru toate voturile
    plt.subplot(2, 1, 1)
    plot_vote_distribution(df)
    
    # 2. Pie chart pentru distribuția procentuală
    plt.subplot(2, 1, 2)
    plot_vote_percentages(df)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Afișăm statisticile numerice
    print_vote_statistics(df)

def plot_vote_distribution(df):
    """
    Creează bar plot-ul pentru distribuția voturilor
    """
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    
    # Calculăm suma totală a voturilor pentru fiecare emoție
    total_votes = df[emotion_columns].sum()
    
    # Creăm un color map pentru emoții (culori mai vii)
    colors = ['#A0A0A0', '#90EE90', '#FFD700', '#87CEEB', 
              '#FF6B6B', '#DDA0DD', '#4169E1', '#FFA500', '#808080', '#000000']
    
    # Creăm bar plot-ul
    bars = plt.bar(emotion_columns, total_votes, color=colors)
    
    plt.title('Distribuția Totală a Voturilor pentru Fiecare Emoție', fontsize=12, pad=20)
    plt.xlabel('Emoții', fontsize=10)
    plt.ylabel('Număr Total de Voturi', fontsize=10)
    
    # Rotăm etichetele pentru lizibilitate
    plt.xticks(rotation=45, ha='right')
    
    # Adăugăm valorile deasupra barelor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')

def plot_vote_percentages(df):
    """
    Creează pie chart-ul pentru distribuția procentuală
    """
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    
    # Calculăm procentajele
    total_votes = df[emotion_columns].sum()
    percentages = (total_votes / total_votes.sum() * 100).round(2)
    
    # Definim culorile (aceleași ca în bar plot)
    colors = ['#A0A0A0', '#90EE90', '#FFD700', '#87CEEB', 
              '#FF6B6B', '#DDA0DD', '#4169E1', '#FFA500', '#808080', '#000000']
    
    # Creăm pie chart-ul
    plt.pie(percentages, labels=emotion_columns, colors=colors,
            autopct='%1.1f%%', startangle=90)
    
    plt.title('Distribuția Procentuală a Voturilor', fontsize=12, pad=20)

def print_vote_statistics(df):
    """
    Afișează statistici detaliate despre voturi
    """
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    
    # Statistici per set de date
    print("\n=== Statistici per Set de Date ===")
    for usage in df['Usage'].unique():
        subset = df[df['Usage'] == usage]
        print(f"\n{usage}:")
        total_votes_subset = subset[emotion_columns].sum()
        percentages_subset = (total_votes_subset / total_votes_subset.sum() * 100).round(2)
        
        for emotion in emotion_columns:
            print(f"{emotion:10}: {int(total_votes_subset[emotion]):,} voturi ({percentages_subset[emotion]}%)")
        print(f"Total imagini: {len(subset):,}")
        print(f"Total voturi: {int(total_votes_subset.sum()):,}")
        print(f"Medie voturi per imagine: {(total_votes_subset.sum() / len(subset)).round(2)}")

def analyze_threshold_based(df, strong_threshold=0.7):
    """
    Analizează datele bazate pe praguri și creează vizualizări relevante
    """
    # Prima figură: Distribuția și emoțiile puternice
    plt.figure(figsize=(15, 7))
    plt.suptitle('Analiza Categoriilor și Emoțiilor Puternice', fontsize=16)
    
    # 1. Distribuția categoriilor (strong/mixed/ambiguous)
    plt.subplot(1, 2, 1)
    plot_category_distribution(df, strong_threshold)
    
    # 2. Distribuția emoțiilor puternice
    plt.subplot(1, 2, 2)
    plot_strong_emotions(df, strong_threshold)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustare pentru titlul principal
    plt.savefig('fer_plus_categories_strong.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # A doua figură: Combinații și intensitate
    plt.figure(figsize=(15, 7))
    plt.suptitle('Analiza Combinațiilor Emoțiilor', fontsize=16)
    
    # 3. Heatmap pentru combinații de emoții în cazurile mixed
    plt.subplot(1, 1, 1)
    plot_emotion_combinations(df, strong_threshold)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustare pentru titlul principal
    plt.savefig('fer_plus_combinations_intensity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Afișăm statisticile detaliate (rămâne la fel)
    print_threshold_statistics(df, strong_threshold)

def categorize_images(df, threshold=0.7):
    """
    Categorizează fiecare imagine în strong/mixed/ambiguous
    """
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    # Calculăm procentajele pentru fiecare emoție
    percentages = df[emotion_columns].div(df[emotion_columns].sum(axis=1), axis=0)
    
    # Găsim emoția dominantă și al doilea cel mai puternic scor
    max_emotions = percentages.max(axis=1)
    second_emotions = percentages.apply(lambda x: sorted(x)[-2], axis=1)
    
    # Categorizăm imaginile
    categories = pd.Series(index=df.index, dtype=str)
    categories[max_emotions >= threshold] = 'strong'
    categories[(max_emotions < threshold) & 
              (max_emotions + second_emotions >= threshold)] = 'mixed'
    categories[categories.isna()] = 'ambiguous'
    
    return categories, percentages

def plot_category_distribution(df, threshold):
    """
    Creează pie chart pentru distribuția categoriilor
    """
    categories, _ = categorize_images(df, threshold)
    category_counts = categories.value_counts()
    
    colors = ['#4CAF50', '#FFC107', '#F44336']
    plt.pie(category_counts, labels=category_counts.index,
            autopct='%1.1f%%', colors=colors)
    plt.title(f'Distribuția Categoriilor (Prag: {threshold*100}%)')

def plot_strong_emotions(df, threshold):
    """
    Analizează distribuția emoțiilor în cazurile "strong emotion"
    """
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    categories, percentages = categorize_images(df, threshold)
    strong_cases = percentages[categories == 'strong']
    
    # Găsim emoția dominantă pentru fiecare caz strong
    dominant_emotions = strong_cases.idxmax(axis=1).value_counts()
    
    colors = ['#A0A0A0', '#90EE90', '#FFD700', '#87CEEB', 
              '#FF6B6B', '#DDA0DD', '#4169E1', '#FFA500']
    
    plt.bar(dominant_emotions.index, dominant_emotions, color=colors)
    plt.title('Distribuția Emoțiilor Puternice')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Număr de Imagini')

def plot_emotion_combinations(df, threshold):
    """
    Creează heatmap pentru combinațiile de emoții în cazurile mixed
    """
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    categories, percentages = categorize_images(df, threshold)
    mixed_cases = percentages[categories == 'mixed']
    
    # Creăm matricea de corelație pentru cazurile mixed
    correlation_matrix = mixed_cases.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                fmt='.2f', square=True)
    plt.title('Corelații între Emoții (Cazuri Mixed)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

def print_threshold_statistics(df, threshold):
    """
    Afișează statistici detaliate despre categorii
    """
    categories, percentages = categorize_images(df, threshold)
    
    print(f"\n=== Statistici pentru Prag {threshold*100}% ===")
    
    # Statistici generale
    print("\nDistribuția categoriilor:")
    category_counts = categories.value_counts()
    for category in category_counts.index:
        print(f"{category:10}: {category_counts[category]:,} imagini ({category_counts[category]/len(df)*100:.1f}%)")
    
    # Statistici pentru cazurile strong
    print("\nEmoții dominante în cazurile strong:")
    strong_cases = percentages[categories == 'strong']
    dominant_emotions = strong_cases.idxmax(axis=1).value_counts()
    for emotion in dominant_emotions.index:
        print(f"{emotion:10}: {dominant_emotions[emotion]:,} imagini ({dominant_emotions[emotion]/len(strong_cases)*100:.1f}%)")

def main():
    base_path = r"C:\Users\gabri\licenta-face-recognition"
    path_to_ferplus = os.path.join(base_path, "fer2013new.csv")
    
    try:
        # Încărcăm datele
        df = load_data(path_to_ferplus)
        
        # Afișăm statisticile de bază
        print_basic_stats(df)
        
        # Salvăm DataFrame-ul procesat pentru utilizare ulterioară
        df.to_csv('processed_fer_data.csv', index=False)
        
        # Analizăm datele bazate pe praguri
        analyze_threshold_based(df, strong_threshold=0.7)
        
        print("\nDatele au fost încărcate și procesate cu succes!")
        
    except FileNotFoundError:
        print(f"Eroare: Nu s-a putut găsi fișierul la calea: {path_to_ferplus}")
    except Exception as e:
        print(f"A apărut o eroare: {str(e)}")

if __name__ == "__main__":
    main()
