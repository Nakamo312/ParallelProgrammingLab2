import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_results(csv_file, output_image):
    try:
        df = pd.read_csv(csv_file)
        df['Time (ms)'] = df['Multiplication Time (s)'] * 1000
        
        dimensions = sorted(df['Dimension'].unique(), reverse=True)
        threads = sorted(df['Threads'].unique())
        
        time_pivot = df.pivot_table(
            index='Dimension',
            columns='Threads',
            values='Time (ms)',
            aggfunc='median'
        ).reindex(index=dimensions, columns=threads)
        
        validation_pivot = df.pivot_table(
            index='Dimension',
            columns='Threads',
            values='Validation Result',
            aggfunc='all'
        ).reindex_like(time_pivot).fillna(False)

        time_pivot = time_pivot.astype(int)
        validation_pivot = validation_pivot.astype(bool)
        
        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(
            time_pivot,
            annot=True,
            fmt="d",
            cmap="YlGnBu_r",
            linewidths=0.5,
            mask=~validation_pivot,
            cbar_kws={'label': 'Время (мс)'}
        )

        best_dim = time_pivot.stack().idxmin()[0]
        best_threads = time_pivot.stack().idxmin()[1]
        best_time = time_pivot.at[best_dim, best_threads]
        
        

        plt.title('Производительность умножения матриц', fontsize=16, pad=20)
        plt.xlabel('Потоки', fontsize=12)
        plt.ylabel('Размер', fontsize=12)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', default='heatmap.png')
    args = parser.parse_args()
    
    plot_results(args.csv, args.output)