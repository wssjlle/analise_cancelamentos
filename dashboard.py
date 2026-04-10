#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Interativo de Análise de Churn
=========================================

Aplicação Streamlit para visualização interativa dos dados de cancelamento.

Autor: Wanderlei
Data: 2024

Para executar: streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Configurações da página
st.set_page_config(
    page_title="Dashboard de Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Carrega os dados do CSV."""
    data_path = 'data/raw/cancelamentos_sample.csv'
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None


@st.cache_data
def preprocess_data(df):
    """Pré-processa os dados para análise."""
    df_processed = df.copy()
    
    # Criar faixas etárias
    df_processed['faixa_etaria'] = pd.cut(
        df_processed['idade'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56+']
    )
    
    # Criar faixas de tempo como cliente
    df_processed['tempo_cliente_categoria'] = pd.cut(
        df_processed['tempo_como_cliente'],
        bins=[0, 12, 24, 48, 100],
        labels=['0-12 meses', '13-24 meses', '25-48 meses', '48+ meses']
    )
    
    # Criar categoria de gasto
    df_processed['gasto_categoria'] = pd.qcut(
        df_processed['total_gasto'],
        q=4,
        labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto']
    )
    
    return df_processed


def main():
    """Função principal do dashboard."""
    
    # Carregar dados
    df = load_data()
    if df is None:
        st.stop()
    
    df_processed = preprocess_data(df)
    
    # Header
    st.markdown('<p class="main-header">📊 Dashboard de Análise de Churn</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Filtros")
    
    # Filtros na sidebar
    filtro_sexo = st.sidebar.multiselect(
        "Sexo:",
        options=df_processed['sexo'].unique(),
        default=df_processed['sexo'].unique()
    )
    
    filtro_assinatura = st.sidebar.multiselect(
        "Tipo de Assinatura:",
        options=df_processed['assinatura'].unique(),
        default=df_processed['assinatura'].unique()
    )
    
    filtro_contrato = st.sidebar.multiselect(
        "Duração do Contrato:",
        options=df_processed['duracao_contrato'].unique(),
        default=df_processed['duracao_contrato'].unique()
    )
    
    # Aplicar filtros
    df_filtered = df_processed[
        (df_processed['sexo'].isin(filtro_sexo)) &
        (df_processed['assinatura'].isin(filtro_assinatura)) &
        (df_processed['duracao_contrato'].isin(filtro_contrato))
    ]
    
    # KPIs principais
    st.subheader("🎯 Indicadores Principais")
    col1, col2, col3, col4 = st.columns(4)
    
    total_clientes = len(df_filtered)
    total_cancelamentos = df_filtered['cancelou'].sum()
    taxa_churn = (total_cancelamentos / total_clientes * 100) if total_clientes > 0 else 0
    media_idade = df_filtered['idade'].mean()
    media_gasto = df_filtered['total_gasto'].mean()
    
    with col1:
        st.metric(
            label="Total de Clientes",
            value=f"{total_clientes:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Cancelamentos",
            value=f"{int(total_cancelamentos):,}",
            delta=f"{taxa_churn:.1f}% churn",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Idade Média",
            value=f"{media_idade:.1f} anos",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Gasto Médio",
            value=f"R$ {media_gasto:.2f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Abas de análise
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Visão Geral",
        "👥 Perfil dos Clientes",
        "💰 Análise Financeira",
        "📞 Call Center",
        "🤖 Predição ML"
    ])
    
    with tab1:
        st.header("Visão Geral do Churn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de pizza - Distribuição de Churn
            fig_pie = px.pie(
                df_filtered,
                names='cancelou',
                title='Distribuição de Cancelamentos',
                color='cancelou',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                hole=0.4
            )
            fig_pie.update_traces(
                textinfo='percent+label',
                pull=[0.05, 0.05]
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gráfico de barras - Churn por tipo de contrato
            churn_contrato = df_filtered.groupby('duracao_contrato')['cancelou'].agg(['sum', 'count']).reset_index()
            churn_contrato['taxa'] = churn_contrato['sum'] / churn_contrato['count'] * 100
            
            fig_bar = px.bar(
                churn_contrato,
                x='duracao_contrato',
                y='taxa',
                title='Taxa de Churn por Duração do Contrato',
                labels={'duracao_contrato': 'Contrato', 'taxa': 'Taxa de Churn (%)'},
                color='taxa',
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Heatmap de correlação
        st.subheader("Matriz de Correlação")
        numeric_cols = ['idade', 'tempo_como_cliente', 'frequencia_uso', 
                       'ligacoes_callcenter', 'dias_atraso', 'total_gasto', 'cancelou']
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Correlação entre Variáveis'
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab2:
        st.header("Perfil dos Clientes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn por faixa etária
            churn_idade = df_filtered.groupby('faixa_etaria')['cancelou'].agg(['sum', 'count']).reset_index()
            churn_idade['taxa'] = churn_idade['sum'] / churn_idade['count'] * 100
            
            fig_idade = px.bar(
                churn_idade,
                x='faixa_etaria',
                y='taxa',
                title='Taxa de Churn por Faixa Etária',
                labels={'faixa_etaria': 'Faixa Etária', 'taxa': 'Taxa de Churn (%)'},
                color='taxa',
                color_continuous_scale='Oranges'
            )
            fig_idade.update_layout(height=400)
            st.plotly_chart(fig_idade, use_container_width=True)
        
        with col2:
            # Churn por tempo como cliente
            churn_tempo = df_filtered.groupby('tempo_cliente_categoria')['cancelou'].agg(['sum', 'count']).reset_index()
            churn_tempo['taxa'] = churn_tempo['sum'] / churn_tempo['count'] * 100
            
            fig_tempo = px.line(
                churn_tempo,
                x='tempo_cliente_categoria',
                y='taxa',
                title='Taxa de Churn por Tempo como Cliente',
                labels={'tempo_cliente_categoria': 'Tempo como Cliente', 'taxa': 'Taxa de Churn (%)'},
                markers=True
            )
            fig_tempo.update_traces(line=dict(width=3))
            fig_tempo.update_layout(height=400)
            st.plotly_chart(fig_tempo, use_container_width=True)
        
        # Distribuição por sexo e assinatura
        col3, col4 = st.columns(2)
        
        with col3:
            fig_sexo = px.histogram(
                df_filtered,
                x='sexo',
                color='cancelou',
                barmode='group',
                title='Distribuição de Churn por Sexo',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig_sexo.update_layout(height=400)
            st.plotly_chart(fig_sexo, use_container_width=True)
        
        with col4:
            fig_assinatura = px.histogram(
                df_filtered,
                x='assinatura',
                color='cancelou',
                barmode='group',
                title='Distribuição de Churn por Tipo de Assinatura',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig_assinatura.update_layout(height=400)
            st.plotly_chart(fig_assinatura, use_container_width=True)
    
    with tab3:
        st.header("Análise Financeira")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gasto médio por categoria de churn
            fig_gasto = px.box(
                df_filtered,
                x='cancelou',
                y='total_gasto',
                title='Distribuição de Gastos por Status de Churn',
                labels={'cancelou': 'Churn', 'total_gasto': 'Total Gasto (R$)'},
                color='cancelou',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig_gasto.update_layout(height=400)
            st.plotly_chart(fig_gasto, use_container_width=True)
        
        with col2:
            # Churn por categoria de gasto
            churn_gasto_cat = df_filtered.groupby('gasto_categoria')['cancelou'].agg(['sum', 'count']).reset_index()
            churn_gasto_cat['taxa'] = churn_gasto_cat['sum'] / churn_gasto_cat['count'] * 100
            
            fig_gasto_cat = px.bar(
                churn_gasto_cat,
                x='gasto_categoria',
                y='taxa',
                title='Taxa de Churn por Categoria de Gasto',
                labels={'gasto_categoria': 'Categoria de Gasto', 'taxa': 'Taxa de Churn (%)'},
                color='taxa',
                color_continuous_scale='YlOrRd'
            )
            fig_gasto_cat.update_layout(height=400)
            st.plotly_chart(fig_gasto_cat, use_container_width=True)
        
        # Scatter plot - Gasto vs Frequência
        fig_scatter = px.scatter(
            df_filtered,
            x='frequencia_uso',
            y='total_gasto',
            color='cancelou',
            size='dias_atraso',
            hover_data=['idade', 'tempo_como_cliente'],
            title='Relação: Frequência de Uso vs Total Gasto',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            opacity=0.6
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.header("Análise de Call Center")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ligações vs Churn
            fig_ligacoes = px.box(
                df_filtered,
                x='cancelou',
                y='ligacoes_callcenter',
                title='Ligações para Call Center por Status de Churn',
                labels={'cancelou': 'Churn', 'ligacoes_callcenter': 'Nº de Ligações'},
                color='cancelou',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig_ligacoes.update_layout(height=400)
            st.plotly_chart(fig_ligacoes, use_container_width=True)
        
        with col2:
            # Dias em atraso vs Churn
            fig_atraso = px.box(
                df_filtered,
                x='cancelou',
                y='dias_atraso',
                title='Dias em Atraso por Status de Churn',
                labels={'cancelou': 'Churn', 'dias_atraso': 'Dias em Atraso'},
                color='cancelou',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            fig_atraso.update_layout(height=400)
            st.plotly_chart(fig_atraso, use_container_width=True)
        
        # Heatmap - Ligações vs Dias em Atraso
        df_filtered['ligacoes_bin'] = pd.cut(df_filtered['ligacoes_callcenter'], 
                                             bins=[0, 2, 4, 6, 100], 
                                             labels=['0-2', '3-4', '5-6', '7+'])
        df_filtered['atraso_bin'] = pd.cut(df_filtered['dias_atraso'], 
                                           bins=[0, 5, 10, 20, 100], 
                                           labels=['0-5', '6-10', '11-20', '21+'])
        
        heatmap_data = df_filtered.groupby(['ligacoes_bin', 'atraso_bin'])['cancelou'].mean().unstack(fill_value=0) * 100
        
        fig_heatmap_cc = px.imshow(
            heatmap_data,
            text_auto='.1f',
            aspect='auto',
            color_continuous_scale='Reds',
            title='Taxa de Churn: Ligações vs Dias em Atraso (%)'
        )
        fig_heatmap_cc.update_layout(height=500)
        st.plotly_chart(fig_heatmap_cc, use_container_width=True)
    
    with tab5:
        st.header("Predição de Churn com Machine Learning")
        
        st.info("""
        **Modelo Treinado:** Random Forest Classifier
        
        **Métricas de Performance:**
        - Acurácia: 99.7%
        - Precision: 100%
        - Recall: 99.5%
        - F1-Score: 99.7%
        - ROC-AUC: 100%
        
        **Top Features Mais Importantes:**
        1. Ligações para Call Center (31.1%)
        2. Total Gasto (21.6%)
        3. Idade (14.9%)
        4. Dias em Atraso (14.1%)
        5. Duração do Contrato (9.0%)
        """)
        
        # Carregar visualizações do modelo
        roc_path = 'outputs/figures/roc_curves.png'
        confusion_path = 'outputs/figures/confusion_matrices.png'
        importance_path = 'outputs/figures/feature_importance.png'
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(roc_path):
                st.image(roc_path, caption='Curvas ROC', use_container_width=True)
            else:
                st.warning("Gráfico ROC não encontrado")
        
        with col2:
            if os.path.exists(importance_path):
                st.image(importance_path, caption='Importância das Features', use_container_width=True)
            else:
                st.warning("Gráfico de importância não encontrado")
        
        if os.path.exists(confusion_path):
            st.image(confusion_path, caption='Matrizes de Confusão', use_container_width=True)
        
        # Simulador de predição
        st.subheader("🔮 Simulador de Churn")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sim_idade = st.slider("Idade:", 18, 80, 35)
            sim_tempo = st.number_input("Tempo como Cliente (meses):", 1, 120, 24)
            sim_frequencia = st.slider("Frequência de Uso:", 1, 10, 5)
        
        with col2:
            sim_ligacoes = st.number_input("Ligações Call Center:", 0, 20, 2)
            sim_atraso = st.number_input("Dias em Atraso:", 0, 60, 5)
            sim_gasto = st.number_input("Total Gasto (R$):", 0.0, 10000.0, 500.0)
        
        with col3:
            sim_sexo = st.selectbox("Sexo:", ['Masculino', 'Feminino'])
            sim_assinatura = st.selectbox("Assinatura:", ['Basica', 'Standard', 'Premium'])
            sim_contrato = st.selectbox("Contrato:", ['Mensal', 'Trimestral', 'Anual'])
        
        if st.button("🎯 Prever Churn", type="primary"):
            # Codificar variáveis categóricas
            sexo_encoded = 0 if sim_sexo == 'Masculino' else 1
            assinatura_map = {'Basica': 0, 'Standard': 1, 'Premium': 2}
            assinatura_encoded = assinatura_map.get(sim_assinatura, 0)
            contrato_map = {'Mensal': 0, 'Trimestral': 1, 'Anual': 2}
            contrato_encoded = contrato_map.get(sim_contrato, 0)
            
            # Carregar modelo
            try:
                model_path = 'models/best_churn_model.pkl'
                scaler_path = 'models/scaler.pkl'
                
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    # Preparar features
                    features = np.array([[
                        sim_idade, sim_tempo, sim_frequencia,
                        sim_ligacoes, sim_atraso, sim_gasto,
                        0,  # meses_ultima_interacao (placeholder)
                        sexo_encoded, assinatura_encoded, contrato_encoded
                    ]])
                    
                    # Escalonar
                    features_scaled = scaler.transform(features)
                    
                    # Predizer
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0][1]
                    
                    if prediction == 1:
                        st.error(f"⚠️ **Alto Risco de Churn!** ({probability*100:.1f}% de probabilidade)")
                        st.warning("""
                        **Recomendações:**
                        - Entrar em contato proativamente
                        - Oferecer desconto ou upgrade
                        - Investigar motivos das ligações ao call center
                        """)
                    else:
                        st.success(f"✅ **Baixo Risco de Churn** ({(1-probability)*100:.1f}% de probabilidade de permanecer)")
                else:
                    st.warning("Modelo não encontrado. Execute o pipeline de ML primeiro.")
            except Exception as e:
                st.error(f"Erro na predição: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Dashboard desenvolvido por Wanderlei | 2024</p>
        <p>Para mais informações, consulte a documentação do projeto</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
