#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Modelos Preditivos para Análise de Churn
=====================================================

Este módulo implementa modelos de machine learning para prever cancelamento de clientes.

Autor: Wanderlei
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Configurações para reprodutibilidade
np.random.seed(42)


class ChurnPredictor:
    """Classe para treinar e avaliar modelos preditivos de churn."""
    
    def __init__(self, data_path: str = '../data/raw/cancelamentos_sample.csv'):
        """Inicializa o preditor de churn."""
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
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
    
    def preprocess_data(self):
        """Realiza o pré-processamento dos dados."""
        print("\n🔧 Pré-processando dados...")
        
        df = self.df.copy()
        
        # Criar variável target
        df['Churn'] = df['cancelou'].astype(int)
        
        # Codificar variáveis categóricas
        categorical_cols = ['sexo', 'assinatura', 'duracao_contrato']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Selecionar features
        feature_cols = [
            'idade', 'tempo_como_cliente', 'frequencia_uso',
            'ligacoes_callcenter', 'dias_atraso', 'total_gasto',
            'meses_ultima_interacao',
            'sexo_encoded', 'assinatura_encoded', 'duracao_contrato_encoded'
        ]
        
        # Remover rows com valores ausentes nas features
        df = df.dropna(subset=feature_cols)
        
        self.X = df[feature_cols]
        self.y = df['Churn']
        
        print(f"✅ Features selecionadas: {len(feature_cols)}")
        print(f"✅ Dataset após limpeza: {self.X.shape[0]} amostras")
        
        return True
    
    def split_data(self, test_size: float = 0.2):
        """Divide os dados em treino e teste."""
        print("\n✂️ Dividindo dados em treino e teste...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        print(f"✅ Treino: {self.X_train.shape[0]} amostras")
        print(f"✅ Teste: {self.X_test.shape[0]} amostras")
        print(f"✅ Proporção de churn no treino: {self.y_train.mean():.2%}")
        print(f"✅ Proporção de churn no teste: {self.y_test.mean():.2%}")
        
    def scale_features(self):
        """Normaliza as features."""
        print("\n⚖️ Normalizando features...")
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Features normalizadas")
        
    def train_models(self):
        """Treina múltiplos modelos."""
        print("\n🤖 Treinando modelos...")
        
        # Definir modelos
        models = {
            'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            print(f"\n📈 Treinando {name}...")
            
            # Escolher dados escalonados ou não baseado no modelo
            if name == 'Regressão Logística':
                X_train, X_test = self.X_train_scaled, self.X_test_scaled
            else:
                X_train, X_test = self.X_train.values, self.X_test.values
            
            # Treinar
            model.fit(X_train, self.y_train)
            self.models[name] = model
            
            # Avaliar
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            self.results[name] = {
                'model': model,
                'metrics': metrics,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"✅ {name} treinado!")
            print(f"   Acurácia: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
    def cross_validation(self, n_splits: int = 5):
        """Realiza validação cruzada."""
        print(f"\n🔄 Realizando validação cruzada ({n_splits} folds)...")
        
        cv_results = {}
        
        for name, model in self.models.items():
            if name == 'Regressão Logística':
                X = self.X_train_scaled
            else:
                X = self.X_train.values
            
            scores = cross_val_score(model, X, self.y_train, cv=n_splits, scoring='roc_auc')
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
            
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def plot_roc_curves(self, save_path: str = '../outputs/figures/roc_curves.png'):
        """Plota curvas ROC para todos os modelos."""
        print("\n📊 Gerando curvas ROC...")
        
        plt.figure(figsize=(12, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['metrics']['roc_auc']:.4f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aleatório')
        plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
        plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
        plt.title('Curvas ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Curvas ROC salvas em {save_path}")
    
    def plot_confusion_matrices(self, save_path: str = '../outputs/figures/confusion_matrices.png'):
        """Plota matrizes de confusão para todos os modelos."""
        print("\n📊 Gerando matrizes de confusão...")
        
        fig, axes = plt.subplots(1, len(self.models), figsize=(6 * len(self.models), 5))
        
        if len(self.models) == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Não Churn', 'Churn'],
                       yticklabels=['Não Churn', 'Churn'])
            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predito', fontsize=10)
            axes[idx].set_ylabel('Real', fontsize=10)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Matrizes de confusão salvas em {save_path}")
    
    def plot_feature_importance(self, save_path: str = '../outputs/figures/feature_importance.png'):
        """Plota importância das features para Random Forest."""
        print("\n📊 Gerando importância das features...")
        
        if 'Random Forest' in self.models:
            model = self.models['Random Forest']
            feature_names = self.X.columns
            
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 8))
            plt.barh(importance['feature'], importance['importance'], color='steelblue')
            plt.xlabel('Importância', fontsize=12)
            plt.title('Importância das Features - Random Forest', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Importância das features salva em {save_path}")
            
            # Print top features
            print("\n🏆 Top 5 features mais importantes:")
            for idx, row in importance.tail(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def save_models(self, output_dir: str = '../models/'):
        """Salva os modelos treinados."""
        print("\n💾 Salvando modelos...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar melhor modelo
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['metrics']['roc_auc'])
        best_model = self.models[best_model_name]
        
        joblib.dump(best_model, os.path.join(output_dir, 'best_churn_model.pkl'))
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        # Salvar label encoders
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        
        print(f"✅ Melhor modelo ({best_model_name}) salvo em {output_dir}")
        print(f"✅ Scaler salvo em {output_dir}")
        print(f"✅ Label encoders salvos em {output_dir}")
    
    def generate_report(self, save_path: str = '../outputs/churn_model_report.md'):
        """Gera relatório em Markdown com resultados."""
        print("\n📝 Gerando relatório...")
        
        report = []
        report.append("# 📊 Relatório de Modelos Preditivos de Churn\n")
        report.append("## Resumo Executivo\n")
        report.append("Este relatório apresenta os resultados dos modelos de machine learning ")
        report.append("treinados para prever cancelamento de clientes.\n\n")
        
        report.append("## Métricas dos Modelos\n\n")
        report.append("| Modelo | Acurácia | Precision | Recall | F1-Score | ROC-AUC |\n")
        report.append("|--------|----------|-----------|--------|----------|---------|\n")
        
        for name, result in self.results.items():
            m = result['metrics']
            report.append(f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | ")
            report.append(f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |\n")
        
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['metrics']['roc_auc'])
        report.append(f"\n## 🏆 Melhor Modelo: **{best_model}**\n\n")
        report.append(f"Com ROC-AUC de **{self.results[best_model]['metrics']['roc_auc']:.4f}**\n\n")
        
        report.append("## Recomendações\n\n")
        report.append("1. Utilizar o modelo selecionado para identificar clientes com alto risco de churn\n")
        report.append("2. Implementar ações proativas de retenção para clientes identificados\n")
        report.append("3. Monitorar continuamente a performance do modelo em produção\n")
        report.append("4. Atualizar o modelo periodicamente com novos dados\n\n")
        
        report.append("## Próximos Passos\n\n")
        report.append("- Implementar pipeline de deploy do modelo\n")
        report.append("- Criar dashboard de monitoramento em tempo real\n")
        report.append("- Desenvolver sistema de alertas automáticos\n")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"✅ Relatório salvo em {save_path}")
    
    def run_full_pipeline(self):
        """Executa todo o pipeline de modelagem."""
        print("="*70)
        print("🚀 INICIANDO PIPELINE DE MODELAGEM DE CHURN")
        print("="*70)
        
        if not self.load_data():
            return False
        
        if not self.preprocess_data():
            return False
        
        self.split_data()
        self.scale_features()
        self.train_models()
        self.cross_validation()
        self.plot_roc_curves()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        self.save_models()
        self.generate_report()
        
        print("\n" + "="*70)
        print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
        return True


if __name__ == "__main__":
    predictor = ChurnPredictor()
    predictor.run_full_pipeline()
