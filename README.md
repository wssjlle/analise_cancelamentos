# 📊 Projeto: Análise de Cancelamentos (Churn Analysis)

![Badge Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-brightgreen?style=flat&logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-5.14+-purple?style=flat&logo=plotly)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Análise exploratória de dados (EDA) completa para identificar padrões e causas relacionadas ao cancelamento de serviços. Este projeto utiliza Python, Jupyter Notebook e bibliotecas modernas de visualização para gerar insights acionáveis que podem ajudar empresas a reduzir a taxa de churn de clientes.

---

## 🎯 Objetivos

- Realizar uma análise exploratória (EDA) profissional em uma base de dados de cancelamentos
- Identificar padrões de comportamento de clientes que cancelam
- Analisar o impacto de diferentes variáveis no cancelamento (idade, tempo como cliente, tipo de assinatura, etc.)
- Gerar visualizações profissionais e interativas
- Fornecer insights e recomendações estratégicas baseadas em dados
- Servir como base para desenvolvimento de modelos preditivos de churn

---

## 🔍 Principais Insights

### Descobertas da Análise

📌 **Tempo como Cliente**
- A maioria dos cancelamentos ocorre nos primeiros 12 meses de contrato
- Clientes com mais de 2 anos têm taxa de churn significativamente menor

📌 **Tipo de Contrato**
- Contratos anuais apresentam menor taxa de cancelamento
- Contratos mensais têm maior rotatividade de clientes

📌 **Atendimento**
- Clientes com mais ligações ao call center cancelam com mais frequência
- Qualidade do atendimento impacta diretamente na retenção

📌 **Pagamentos**
- Dias de atraso estão correlacionados com probabilidade de cancelamento
- Sistema de alerta preventivo pode reduzir churn

📌 **Segmentação**
- Diferentes faixas etárias apresentam comportamentos distintos
- Tipo de assinatura influencia na decisão de cancelamento

---

## 📁 Estrutura do Projeto

```
analise_cancelamentos/
├── notebooks/
│   └── analise_cancelamentos.ipynb    # Notebook principal com análise completa
├── data/
│   └── raw/
│       └── cancelamentos_sample.csv   # Base de dados original
├── outputs/
│   └── figures/                       # Gráficos e visualizações gerados
│       ├── distribuicao_categoricas.png
│       ├── visao_geral_churn.png
│       ├── churn_por_idade.png
│       ├── churn_por_tempo_cliente.png
│       ├── matriz_correlacao.png
│       └── dashboard_completo.png
├── .gitignore                         # Arquivos ignorados pelo Git
├── LICENSE                            # Licença MIT
├── README.md                          # Esta documentação
└── requirements.txt                   # Dependências do projeto
```

---

## ▶️ Como Executar o Projeto

### Pré-requisitos

- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)
- Jupyter Notebook ou JupyterLab

### Passo a Passo

1. **Clone o repositório:**
```bash
git clone https://github.com/seunome/analise_cancelamentos.git
cd analise_cancelamentos
```

2. **Crie um ambiente virtual (recomendado):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Execute o notebook:**
```bash
jupyter notebook notebooks/analise_cancelamentos.ipynb
```

5. **Ou execute via terminal:**
```bash
jupyter nbconvert --to html notebooks/analise_cancelamentos.ipynb
```

---

## 🛠️ Tecnologias Utilizadas

### Bibliotecas Principais

| Biblioteca | Versão | Finalidade |
|------------|--------|------------|
| Pandas | ≥2.0.0 | Manipulação e análise de dados |
| NumPy | ≥1.24.0 | Computação numérica |
| Matplotlib | ≥3.7.0 | Visualização de dados |
| Seaborn | ≥0.12.0 | Visualização estatística |
| Plotly | ≥5.14.0 | Gráficos interativos |
| Jupyter | ≥1.0.0 | Ambiente de desenvolvimento |

### Bibliotecas Adicionais

- **Kaleido**: Exportação de gráficos Plotly como imagem
- **SciPy**: Análises estatísticas avançadas
- **Scikit-learn**: Preparação para modelos preditivos futuros

---

## 📊 Variáveis do Dataset

| Variável | Tipo | Descrição |
|----------|------|-----------|
| CustomerID | Numérico | Identificador único do cliente |
| idade | Numérico | Idade do cliente em anos |
| sexo | Categórico | Gênero do cliente (Male/Female) |
| tempo_como_cliente | Numérico | Tempo de relacionamento em meses |
| frequencia_uso | Numérico | Frequência de uso do serviço |
| ligacoes_callcenter | Numérico | Número de ligações ao call center |
| dias_atraso | Numérico | Dias de atraso no pagamento |
| assinatura | Categórico | Tipo de assinatura (Basic/Standard/Premium) |
| duracao_contrato | Categórico | Duração do contrato (Monthly/Quarterly/Annual) |
| total_gasto | Numérico | Valor total gasto pelo cliente |
| meses_ultima_interacao | Numérico | Meses desde a última interação |
| cancelou | Binário | Indicador de cancelamento (0=Não, 1=Sim) |

---

## 📈 Métricas de Qualidade

- ✅ **Reprodutibilidade**: Seeds definidas para resultados consistentes
- ✅ **Tratamento de Dados**: Estratégia robusta para valores ausentes
- ✅ **Visualização**: Gráficos profissionais e customizados
- ✅ **Documentação**: Código comentado e estruturado
- ✅ **Insights**: Recomendações acionáveis baseadas em dados

---

## 💡 Próximos Passos Sugeridos

### Melhorias para Implementação Futura

1. **Modelo Preditivo de Churn**
   - Desenvolver modelo de machine learning (Random Forest, XGBoost)
   - Prever probabilidade de cancelamento por cliente
   - Identificar features mais importantes

2. **Análise Temporal**
   - Adicionar dimensão temporal (mês/ano de entrada)
   - Analisar sazonalidade de cancelamentos
   - Tracking de métricas ao longo do tempo

3. **Segmentação Avançada**
   - Clustering de clientes (K-Means, DBSCAN)
   - Criação de personas de clientes
   - Mapeamento de jornadas típicas de cancelamento

4. **Dashboard Interativo**
   - Implementar com Streamlit ou Dash
   - Monitoramento contínuo de métricas
   - Alertas automáticos de risco de churn

5. **Testes A/B**
   - Testar diferentes estratégias de retenção
   - Medir impacto de campanhas
   - Otimizar abordagens por segmento

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commitar suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abrir um Pull Request

---

## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE). Isso significa que você pode usar, modificar e distribuir este código livremente, desde que mantenha os créditos originais.

---

## 👨‍💻 Autor

**Wanderlei Silva dos Santos**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Perfil-blue?style=flat&logo=linkedin)](https://linkedin.com/in/wanderlei-silva-dos-santos)
[![Email](https://img.shields.io/badge/Email-wssjlle@gmail.com-red?style=flat&logo=gmail)](mailto:wssjlle@gmail.com)

---

## 📚 Referências

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Graphing Library](https://plotly.com/python/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Jupyter Best Practices](https://jupyter.org/guides)

---

> 💬 *"Sem dados, você é apenas mais uma pessoa com uma opinião."* – W. Edwards Deming

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!**
