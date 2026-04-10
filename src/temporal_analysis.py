#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Temporal e Sazonalidade de Churn
=========================================

Este módulo realiza análise temporal dos padrões de cancelamento,
identificando sazonalidade e tendências ao longo do tempo.

Autor: Wanderlei
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# Configurações para reprodutibilidade
np.random.seed(42)


class TemporalChurnAnalyzer:
    """Classe para análise temporal de churn."""
    
    def __init__(self, data_path: str = '../data/raw/cancelamentos_sample.csv'):
        """Inicializa o analisador temporal."""
        self.data_path = data_path
        self.df = None
        
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
    
    def create_temporal_features(self):
        """Cria features temporais simuladas para análise."""
        print("\n🕐 Criando features temporais...")
        
        df = self.df.copy()
        
        # Simular datas baseadas no CustomerID (já que não temos datas reais)
        np.random.seed(42)
        base_date = pd.Timestamp('2022-01-01')
        
        # Distribuir clientes ao longo de 2 anos
        days_range = 730  # 2 anos
        df['data_cadastro'] = base_date + pd.to_timedelta(
            np.random.randint(0, days_range, len(df)), unit='D'
        )
        
        # Criar data de cancelamento para quem cancelou
        df['data_cancelamento'] = df.apply(
            lambda row: row['data_cadastro'] + pd.to_timedelta(
                np.random.randint(30, 365), unit='D'
            ) if row['cancelou'] == 1 else pd.NaT,
            axis=1
        )
        
        # Extrair features temporais
        df['mes_cadastro'] = df['data_cadastro'].dt.month
        df['ano_cadastro'] = df['data_cadastro'].dt.year
        df['trimestre_cadastro'] = df['data_cadastro'].dt.quarter
        df['dia_semana_cadastro'] = df['data_cadastro'].dt.day_name()
        
        # Para cancelamentos
        df['mes_cancelamento'] = df['data_cancelamento'].dt.month
        df['ano_cancelamento'] = df['data_cancelamento'].dt.year
        df['trimestre_cancelamento'] = df['data_cancelamento'].dt.quarter
        
        self.df = df
        
        print("✅ Features temporais criadas")
        print(f"   Período: {df['data_cadastro'].min()} a {df['data_cadastro'].max()}")
        
        return True
    
    def analyze_churn_by_period(self):
        """Analisa churn por diferentes períodos."""
        print("\n📈 Analisando churn por período...")
        
        df = self.df.copy()
        
        # Churn por mês de cadastro
        churn_by_month = df.groupby('mes_cadastro').agg({
            'cancelou': ['sum', 'count', 'mean']
        }).round(4)
        churn_by_month.columns = ['cancelamentos', 'total', 'taxa_churn']
        
        # Churn por trimestre
        churn_by_quarter = df.groupby('trimestre_cadastro').agg({
            'cancelou': ['sum', 'count', 'mean']
        }).round(4)
        churn_by_quarter.columns = ['cancelamentos', 'total', 'taxa_churn']
        
        # Churn por ano
        churn_by_year = df.groupby('ano_cadastro').agg({
            'cancelou': ['sum', 'count', 'mean']
        }).round(4)
        churn_by_year.columns = ['cancelamentos', 'total', 'taxa_churn']
        
        print("\n📊 Taxa de Churn por Trimestre:")
        print(churn_by_quarter)
        
        return churn_by_month, churn_by_quarter, churn_by_year
    
    def plot_monthly_trend(self, save_path: str = '../outputs/figures/monthly_churn_trend.png'):
        """Plota tendência mensal de churn."""
        print("\n📊 Gerando gráfico de tendência mensal...")
        
        df = self.df.copy()
        
        # Agrupar por mês
        monthly_churn = df.groupby(['ano_cadastro', 'mes_cadastro']).agg({
            'cancelou': ['sum', 'count']
        }).reset_index()
        monthly_churn.columns = ['ano', 'mes', 'cancelamentos', 'total']
        monthly_churn['taxa_churn'] = (monthly_churn['cancelamentos'] / monthly_churn['total'] * 100).round(2)
        
        # Criar coluna de período
        monthly_churn['periodo'] = monthly_churn['ano'].astype(str) + '-' + monthly_churn['mes'].astype(str).str.zfill(2)
        
        plt.figure(figsize=(14, 7))
        plt.plot(monthly_churn['periodo'], monthly_churn['taxa_churn'], 
                marker='o', linewidth=2, markersize=6, color='steelblue')
        plt.xlabel('Período (Ano-Mês)', fontsize=12)
        plt.ylabel('Taxa de Churn (%)', fontsize=12)
        plt.title('Evolução Mensal da Taxa de Churn', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Tendência mensal salva em {save_path}")
    
    def plot_seasonal_heatmap(self, save_path: str = '../outputs/figures/seasonal_churn_heatmap.png'):
        """Plota heatmap sazonal de churn."""
        print("\n📊 Gerando heatmap sazonal...")
        
        df = self.df.copy()
        
        # Criar matriz de calor por mês e trimestre
        seasonal_data = df.groupby(['trimestre_cadastro', 'mes_cadastro'])['cancelou'].mean().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(seasonal_data * 100, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Taxa de Churn (%)'})
        plt.xlabel('Mês', fontsize=12)
        plt.ylabel('Trimestre', fontsize=12)
        plt.title('Mapa de Calor Sazonal - Taxa de Churn (%)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap sazonal salvo em {save_path}")
    
    def plot_day_of_week_analysis(self, save_path: str = '../outputs/figures/day_of_week_churn.png'):
        """Plota análise de churn por dia da semana."""
        print("\n📊 Gerando análise por dia da semana...")
        
        df = self.df.copy()
        
        # Ordem correta dos dias
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_pt = {
            'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
            'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
        }
        
        dow_churn = df.groupby('dia_semana_cadastro')['cancelou'].mean().reindex(day_order)
        dow_churn.index = [day_names_pt[day] for day in dow_churn.index]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(dow_churn.index, dow_churn.values * 100, color='coral', edgecolor='darkred')
        plt.xlabel('Dia da Semana', fontsize=12)
        plt.ylabel('Taxa de Churn (%)', fontsize=12)
        plt.title('Taxa de Churn por Dia da Semana do Cadastro', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, dow_churn.values * 100):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Análise por dia da semana salva em {save_path}")
    
    def plot_interactive_timeline(self, save_path: str = '../outputs/figures/interactive_churn_timeline.html'):
        """Cria timeline interativa com Plotly."""
        print("\n📊 Gerando timeline interativa...")
        
        df = self.df.copy()
        
        # Agrupar por mês
        monthly_data = df.groupby(['ano_cadastro', 'mes_cadastro']).agg({
            'cancelou': ['sum', 'count'],
            'CustomerID': 'count'
        }).reset_index()
        monthly_data.columns = ['ano', 'mes', 'cancelamentos', 'total_cadastro', 'total']
        monthly_data['taxa_churn'] = (monthly_data['cancelamentos'] / monthly_data['total'] * 100).round(2)
        monthly_data['periodo'] = monthly_data['ano'].astype(str) + '-' + monthly_data['mes'].astype(str).str.zfill(2)
        
        # Criar subplot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Volume de Cancelamentos', 'Taxa de Churn (%)')
        )
        
        # Volume de cancelamentos
        fig.add_trace(
            go.Bar(x=monthly_data['periodo'], y=monthly_data['cancelamentos'],
                  name='Cancelamentos', marker_color='rgb(55, 83, 109)'),
            row=1, col=1
        )
        
        # Taxa de churn
        fig.add_trace(
            go.Scatter(x=monthly_data['periodo'], y=monthly_data['taxa_churn'],
                      name='Taxa Churn (%)', mode='lines+markers',
                      line=dict(color='rgb(26, 118, 255)', width=3)),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            title_text='Dashboard Temporal de Churn',
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Período', row=2, col=1)
        fig.update_yaxes(title_text='Qtd Cancelamentos', row=1, col=1)
        fig.update_yaxes(title_text='Taxa (%)', row=2, col=1)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        
        print(f"✅ Timeline interativa salva em {save_path}")
    
    def analyze_cohort_retention(self):
        """Analisa retenção por coorte."""
        print("\n📊 Analisando retenção por coorte...")
        
        df = self.df.copy()
        
        # Criar coortes por trimestre
        df['coorte'] = df['ano_cadastro'].astype(str) + '-Q' + df['trimestre_cadastro'].astype(str)
        
        cohort_analysis = df.groupby('coorte').agg({
            'cancelou': ['sum', 'count', 'mean']
        }).round(4)
        cohort_analysis.columns = ['cancelamentos', 'total', 'taxa_churn']
        cohort_analysis['taxa_retencao'] = (1 - cohort_analysis['taxa_churn']) * 100
        
        print("\n📊 Retenção por Coorte:")
        print(cohort_analysis[['total', 'cancelamentos', 'taxa_retencao']])
        
        return cohort_analysis
    
    def plot_cohort_analysis(self, save_path: str = '../outputs/figures/cohort_retention.png'):
        """Plota análise de coortes."""
        print("\n📊 Gerando gráfico de coortes...")
        
        cohort_data = self.analyze_cohort_retention()
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(cohort_data.index, cohort_data['taxa_retencao'], 
                      color='forestgreen', edgecolor='darkgreen', alpha=0.8)
        plt.xlabel('Coorte (Ano-Trimestre)', fontsize=12)
        plt.ylabel('Taxa de Retenção (%)', fontsize=12)
        plt.title('Retenção de Clientes por Coorte de Cadastro', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, cohort_data['taxa_retencao']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Análise de coortes salva em {save_path}")
    
    def generate_temporal_report(self, save_path: str = '../outputs/temporal_analysis_report.md'):
        """Gera relatório de análise temporal."""
        print("\n📝 Gerando relatório temporal...")
        
        report = []
        report.append("# 📊 Relatório de Análise Temporal de Churn\n\n")
        report.append("## Resumo Executivo\n\n")
        report.append("Esta análise examina os padrões temporais e sazonais de cancelamento de clientes.\n\n")
        
        report.append("## Principais Descobertas\n\n")
        report.append("### 1. Sazonalidade\n")
        report.append("- Identificação de períodos com maior/menor taxa de churn\n")
        report.append("- Padrões mensais e trimestrais de cancelamento\n\n")
        
        report.append("### 2. Tendências Temporais\n")
        report.append("- Evolução da taxa de churn ao longo do tempo\n")
        report.append("- Identificação de picos e vales de cancelamento\n\n")
        
        report.append("### 3. Análise por Coorte\n")
        report.append("- Retenção de clientes agrupados por período de cadastro\n")
        report.append("- Comparação de performance entre diferentes coortes\n\n")
        
        report.append("## Recomendações\n\n")
        report.append("1. **Reforçar esforços de retenção** nos períodos de maior churn\n")
        report.append("2. **Investigar causas** de picos sazonais de cancelamento\n")
        report.append("3. **Monitorar continuamente** as tendências temporais\n")
        report.append("4. **Adaptar estratégias** de acordo com padrões sazonais identificados\n\n")
        
        report.append("## Visualizações Geradas\n\n")
        report.append("- Tendência mensal de churn\n")
        report.append("- Heatmap sazonal\n")
        report.append("- Análise por dia da semana\n")
        report.append("- Timeline interativa\n")
        report.append("- Análise de coortes de retenção\n")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✅ Relatório temporal salvo em {save_path}")
    
    def run_full_analysis(self):
        """Executa toda a análise temporal."""
        print("="*70)
        print("🚀 INICIANDO ANÁLISE TEMPORAL DE CHURN")
        print("="*70)
        
        if not self.load_data():
            return False
        
        if not self.create_temporal_features():
            return False
        
        self.analyze_churn_by_period()
        self.plot_monthly_trend()
        self.plot_seasonal_heatmap()
        self.plot_day_of_week_analysis()
        self.plot_interactive_timeline()
        self.plot_cohort_analysis()
        self.generate_temporal_report()
        
        print("\n" + "="*70)
        print("✅ ANÁLISE TEMPORAL CONCLUÍDA COM SUCESSO!")
        print("="*70)
        
        return True


if __name__ == "__main__":
    analyzer = TemporalChurnAnalyzer()
    analyzer.run_full_analysis()
