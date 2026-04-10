#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentação de Clientes com Clustering
=======================================

Este módulo implementa algoritmos de clustering para segmentação de clientes
baseado em padrões de comportamento e risco de churn.

Autor: Wanderlei
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# Configurações para reprodutibilidade
np.random.seed(42)


class CustomerSegmentation:
    """Classe para segmentação de clientes usando clustering."""
    
    def __init__(self, data_path: str = '../data/raw/cancelamentos_sample.csv'):
        """Inicializa o segmentador de clientes."""
        self.data_path = data_path
        self.df = None
        self.df_scaled = None
        self.scaler = StandardScaler()
        self.clusters = None
        self.optimal_k = None
        
    def load_data(self):
        """Carrega os dados do CSV."""
        print("📊 Carregando dados...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dados carregados: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def prepare_features(self):
        """Prepara features para clustering."""
        print("\n🔧 Preparando features para clustering...")
        
        df = self.df.copy()
        
        # Selecionar features relevantes para clustering
        feature_cols = [
            'idade', 'tempo_como_cliente', 'frequencia_uso',
            'ligacoes_callcenter', 'dias_atraso', 'total_gasto'
        ]
        
        # Remover valores ausentes
        df = df.dropna(subset=feature_cols)
        
        # Normalizar features
        features = df[feature_cols].values
        self.df_scaled = self.scaler.fit_transform(features)
        
        # Armazenar dataframe limpo
        self.df = df.reset_index(drop=True)
        
        print(f"✅ Features preparadas: {len(feature_cols)} variáveis")
        print(f"✅ Dataset após limpeza: {self.df.shape[0]} amostras")
        
        return True
    
    def find_optimal_clusters(self, max_k: int = 8):
        """Encontra o número ótimo de clusters usando múltiplas métricas."""
        print(f"\n🔍 Buscando número ótimo de clusters (k=2 a {max_k})...")
        
        # Usar amostra para cálculo das métricas (dataset muito grande)
        sample_size = min(5000, len(self.df_scaled))
        sample_indices = np.random.choice(len(self.df_scaled), sample_size, replace=False)
        df_scaled_sample = self.df_scaled[sample_indices]
        
        inertia = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df_scaled_sample)
            
            inertia.append(kmeans.inertia_)
            if k <= len(np.unique(labels)):
                silhouette_scores.append(silhouette_score(df_scaled_sample, labels))
                davies_bouldin_scores.append(davies_bouldin_score(df_scaled_sample, labels))
                calinski_harabasz_scores.append(calinski_harabasz_score(df_scaled_sample, labels))
            else:
                silhouette_scores.append(0)
                davies_bouldin_scores.append(float('inf'))
                calinski_harabasz_scores.append(0)
        
        # Encontrar optimal k baseado no silhouette score
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"\n📊 Métricas de avaliação:")
        print(f"   Silhouette Score máximo: {max(silhouette_scores):.4f} (k={self.optimal_k})")
        print(f"   Davies-Bouldin mínimo: {min(davies_bouldin_scores):.4f}")
        print(f"   Calinski-Harabasz máximo: {max(calinski_harabasz_scores):.4f}")
        
        # Plotar métricas
        self.plot_elbow_and_metrics(k_range, inertia, silhouette_scores, 
                                   davies_bouldin_scores, calinski_harabasz_scores)
        
        return self.optimal_k
    
    def plot_elbow_and_metrics(self, k_range, inertia, silhouette, db_scores, ch_scores):
        """Plota gráficos para determinação do k ótimo."""
        print("\n📊 Gerando gráficos de otimização...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Elbow method
        axes[0, 0].plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].axvline(x=self.optimal_k, color='r', linestyle='--', 
                          label=f'k ótimo = {self.optimal_k}')
        axes[0, 0].set_xlabel('Número de Clusters (k)', fontsize=12)
        axes[0, 0].set_ylabel('Inércia', fontsize=12)
        axes[0, 0].set_title('Método do Cotovelo (Elbow Method)', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[0, 1].plot(k_range, silhouette, 'go-', linewidth=2, markersize=8)
        axes[0, 1].axvline(x=self.optimal_k, color='r', linestyle='--',
                          label=f'k ótimo = {self.optimal_k}')
        axes[0, 1].set_xlabel('Número de Clusters (k)', fontsize=12)
        axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
        axes[0, 1].set_title('Silhouette Score', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[1, 0].plot(k_range, db_scores, 'ro-', linewidth=2, markersize=8)
        axes[1, 0].axvline(x=self.optimal_k, color='g', linestyle='--',
                          label=f'k ótimo = {self.optimal_k}')
        axes[1, 0].set_xlabel('Número de Clusters (k)', fontsize=12)
        axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=12)
        axes[1, 0].set_title('Davies-Bouldin Index (menor é melhor)', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz
        axes[1, 1].plot(k_range, ch_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].axvline(x=self.optimal_k, color='r', linestyle='--',
                          label=f'k ótimo = {self.optimal_k}')
        axes[1, 1].set_xlabel('Número de Clusters (k)', fontsize=12)
        axes[1, 1].set_ylabel('Calinski-Harabasz Score', fontsize=12)
        axes[1, 1].set_title('Calinski-Harabasz Score (maior é melhor)', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = '../outputs/figures/clustering_optimization.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráficos de otimização salvos em {save_path}")
    
    def perform_clustering(self, n_clusters: int = None):
        """Realiza o clustering com K-Means."""
        if n_clusters is None:
            n_clusters = self.optimal_k
        
        print(f"\n🎯 Realizando clustering com k={n_clusters}...")
        
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.df_scaled)
        
        # Adicionar cluster ao dataframe
        self.df['cluster'] = self.clusters
        
        # Calcular métricas finais (usando amostra para evitar problemas de memória)
        sample_size = min(5000, len(self.df_scaled))
        sample_indices = np.random.choice(len(self.df_scaled), sample_size, replace=False)
        
        silhouette = silhouette_score(self.df_scaled[sample_indices], self.clusters[sample_indices])
        davies_bouldin = davies_bouldin_score(self.df_scaled[sample_indices], self.clusters[sample_indices])
        calinski_harabasz = calinski_harabasz_score(self.df_scaled[sample_indices], self.clusters[sample_indices])
        
        print(f"\n📊 Métricas do clustering:")
        print(f"   Silhouette Score: {silhouette:.4f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"   Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        
        # Analisar características dos clusters
        self.analyze_clusters()
        
        return self.clusters
    
    def analyze_clusters(self):
        """Analisa as características de cada cluster."""
        print("\n📊 Análise dos Clusters:")
        print("="*70)
        
        df = self.df.copy()
        
        # Estatísticas por cluster
        cluster_stats = df.groupby('cluster').agg({
            'idade': ['mean', 'std'],
            'tempo_como_cliente': ['mean', 'std'],
            'frequencia_uso': ['mean', 'std'],
            'ligacoes_callcenter': ['mean', 'std'],
            'dias_atraso': ['mean', 'std'],
            'total_gasto': ['mean', 'std'],
            'cancelou': ['mean', 'count']
        }).round(2)
        
        print("\nEstatísticas por Cluster:")
        print(cluster_stats)
        
        # Taxa de churn por cluster
        churn_by_cluster = df.groupby('cluster')['cancelou'].mean().sort_values(ascending=False)
        print("\n🚨 Taxa de Churn por Cluster:")
        for cluster, rate in churn_by_cluster.items():
            print(f"   Cluster {cluster}: {rate*100:.2f}%")
        
        # Nomear clusters baseados nas características
        self.name_clusters(churn_by_cluster)
        
        return cluster_stats
    
    def name_clusters(self, churn_rates):
        """Atribui nomes descritivos aos clusters."""
        print("\n🏷️ Nomeando clusters...")
        
        df = self.df.copy()
        cluster_names = {}
        
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            
            # Critérios para nomeação
            avg_ligacoes = cluster_data['ligacoes_callcenter'].mean()
            avg_churn = cluster_data['cancelou'].mean()
            avg_gasto = cluster_data['total_gasto'].mean()
            avg_tempo = cluster_data['tempo_como_cliente'].mean()
            
            if avg_churn > 0.7:
                name = "Alto Risco"
            elif avg_ligacoes > 5 and avg_churn > 0.5:
                name = "Insatisfeitos"
            elif avg_gasto > df['total_gasto'].median() and avg_churn < 0.3:
                name = "Premium Fidelizado"
            elif avg_tempo < 12:
                name = "Novos Clientes"
            else:
                name = "Clientes Estáveis"
            
            cluster_names[cluster] = name
            print(f"   Cluster {cluster}: {name} (Churn: {avg_churn*100:.1f}%)")
        
        self.df['cluster_nome'] = self.df['cluster'].map(cluster_names)
        self.cluster_names = cluster_names
    
    def plot_clusters_2d(self, save_path: str = '../outputs/figures/clusters_2d.png'):
        """Plota clusters em 2D usando PCA."""
        print("\n📊 Gerando visualização 2D dos clusters...")
        
        # PCA para redução de dimensionalidade
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(self.df_scaled)
        
        self.df['PC1'] = principal_components[:, 0]
        self.df['PC2'] = principal_components[:, 1]
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(self.df['PC1'], self.df['PC2'], 
                             c=self.df['cluster'], cmap='viridis',
                             alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        plt.title('Visualização dos Clusters (PCA)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualização 2D salva em {save_path}")
    
    def plot_interactive_clusters(self, save_path: str = '../outputs/figures/interactive_clusters.html'):
        """Cria visualização interativa dos clusters com Plotly."""
        print("\n📊 Gerando visualização interativa...")
        
        df = self.df.copy()
        
        fig = px.scatter_3d(
            df,
            x='PC1', y='PC2', z='total_gasto',
            color='cluster',
            color_continuous_scale='viridis',
            hover_data=['idade', 'tempo_como_cliente', 'ligacoes_callcenter', 'cancelou'],
            title='Clusters de Clientes (3D Interactive)',
            labels={'PC1': 'Componente 1', 'PC2': 'Componente 2', 'total_gasto': 'Total Gasto'}
        )
        
        fig.update_layout(height=700, scene_aspectmode='cube')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        print(f"✅ Visualização interativa salva em {save_path}")
    
    def plot_cluster_profiles(self, save_path: str = '../outputs/figures/cluster_profiles.png'):
        """Plota perfis dos clusters."""
        print("\n📊 Gerando perfis dos clusters...")
        
        df = self.df.copy()
        
        # Normalizar dados para comparação
        feature_cols = ['idade', 'tempo_como_cliente', 'frequencia_uso',
                       'ligacoes_callcenter', 'dias_atraso', 'total_gasto']
        
        cluster_means = df.groupby('cluster')[feature_cols].mean()
        
        # Normalizar entre 0 e 1 para visualização
        cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Radar chart simplificado (barras normalizadas)
        ax1 = axes[0, 0]
        cluster_means_normalized.plot(kind='bar', ax=ax1, colormap='viridis')
        ax1.set_xlabel('Cluster', fontsize=12)
        ax1.set_ylabel('Valor Normalizado', fontsize=12)
        ax1.set_title('Perfil dos Clusters (Features Normalizadas)', fontsize=12, fontweight='bold')
        ax1.legend(title='Variáveis', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(True, alpha=0.3)
        
        # Tamanho dos clusters
        ax2 = axes[0, 1]
        cluster_sizes = df['cluster'].value_counts().sort_index()
        bars = ax2.bar(cluster_sizes.index.astype(str), cluster_sizes.values, 
                      color='steelblue', edgecolor='darkblue')
        ax2.set_xlabel('Cluster', fontsize=12)
        ax2.set_ylabel('Número de Clientes', fontsize=12)
        ax2.set_title('Tamanho dos Clusters', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, cluster_sizes.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
        
        # Churn rate por cluster
        ax3 = axes[1, 0]
        churn_rates = df.groupby('cluster')['cancelou'].mean() * 100
        colors = ['red' if rate > 50 else 'orange' if rate > 30 else 'green' for rate in churn_rates.values]
        bars = ax3.bar(churn_rates.index.astype(str), churn_rates.values, color=colors, edgecolor='black')
        ax3.set_xlabel('Cluster', fontsize=12)
        ax3.set_ylabel('Taxa de Churn (%)', fontsize=12)
        ax3.set_title('Taxa de Churn por Cluster', fontsize=12, fontweight='bold')
        ax3.axhline(y=df['cancelou'].mean()*100, color='red', linestyle='--', 
                   label=f'Média Geral: {df["cancelou"].mean()*100:.1f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, churn_rates.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Distribuição de idade por cluster
        ax4 = axes[1, 1]
        df.boxplot(column='idade', by='cluster', ax=ax4)
        ax4.set_xlabel('Cluster', fontsize=12)
        ax4.set_ylabel('Idade', fontsize=12)
        ax4.set_title('Distribuição de Idade por Cluster', fontsize=12, fontweight='bold')
        plt.suptitle('')  # Remover título automático
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Perfis dos clusters salvos em {save_path}")
    
    def generate_segmentation_report(self, save_path: str = '../outputs/customer_segmentation_report.md'):
        """Gera relatório de segmentação de clientes."""
        print("\n📝 Gerando relatório de segmentação...")
        
        report = []
        report.append("# 📊 Relatório de Segmentação de Clientes\n\n")
        report.append("## Resumo Executivo\n\n")
        report.append("Este relatório apresenta a segmentação de clientes utilizando algoritmos ")
        report.append("de clustering não supervisionado para identificar grupos com comportamentos similares.\n\n")
        
        report.append(f"## Configuração do Modelo\n\n")
        report.append(f"- **Algoritmo:** K-Means\n")
        report.append(f"- **Número de Clusters:** {self.optimal_k}\n")
        report.append(f"- **Features Utilizadas:** 6 variáveis comportamentais\n\n")
        
        report.append("## Clusters Identificados\n\n")
        for cluster_id, name in self.cluster_names.items():
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            report.append(f"### Cluster {cluster_id}: {name}\n")
            report.append(f"- **Tamanho:** {len(cluster_data)} clientes ({len(cluster_data)/len(self.df)*100:.1f}%)\n")
            report.append(f"- **Taxa de Churn:** {cluster_data['cancelou'].mean()*100:.1f}%\n")
            report.append(f"- **Idade Média:** {cluster_data['idade'].mean():.1f} anos\n")
            report.append(f"- **Tempo Médio como Cliente:** {cluster_data['tempo_como_cliente'].mean():.1f} meses\n")
            report.append(f"- **Ligações Call Center (média):** {cluster_data['ligacoes_callcenter'].mean():.1f}\n")
            report.append(f"- **Gasto Médio:** R$ {cluster_data['total_gasto'].mean():.2f}\n\n")
        
        report.append("## Recomendações Estratégicas\n\n")
        report.append("### Para Cluster de Alto Risco:\n")
        report.append("1. Implementar programa de retenção proativa\n")
        report.append("2. Oferecer benefícios personalizados\n")
        report.append("3. Reduzir tempo de espera no call center\n\n")
        
        report.append("### Para Cluster Premium Fidelizado:\n")
        report.append("1. Programa de fidelidade exclusivo\n")
        report.append("2. Upsell de produtos premium\n")
        report.append("3. Reconhecimento e recompensas\n\n")
        
        report.append("### Para Novos Clientes:\n")
        report.append("1. Onboarding otimizado\n")
        report.append("2. Acompanhamento nos primeiros 90 dias\n")
        report.append("3. Educação sobre o produto/serviço\n\n")
        
        report.append("## Próximos Passos\n\n")
        report.append("1. Validar segmentos com equipe de negócio\n")
        report.append("2. Desenvolver estratégias específicas por segmento\n")
        report.append("3. Monitorar evolução dos clusters ao longo do tempo\n")
        report.append("4. Refinar modelo com novas variáveis\n")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✅ Relatório de segmentação salvo em {save_path}")
    
    def run_full_segmentation(self):
        """Executa todo o pipeline de segmentação."""
        print("="*70)
        print("🚀 INICIANDO SEGMENTAÇÃO DE CLIENTES")
        print("="*70)
        
        if not self.load_data():
            return False
        
        if not self.prepare_features():
            return False
        
        self.find_optimal_clusters()
        self.perform_clustering()
        self.plot_clusters_2d()
        self.plot_interactive_clusters()
        self.plot_cluster_profiles()
        self.generate_segmentation_report()
        
        print("\n" + "="*70)
        print("✅ SEGMENTAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*70)
        
        return True


if __name__ == "__main__":
    segmenter = CustomerSegmentation()
    segmenter.run_full_segmentation()
