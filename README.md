# 📊 Projeto: Análise de Cancelamentos

![Badge Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Concluído-green)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Análise de dados para identificar padrões e causas relacionadas ao cancelamento de serviços. Este projeto usa Python e Jupyter Notebook para explorar uma base de dados e gerar insights que podem ser usados por empresas para reduzir a taxa de cancelamento de clientes.

---

## 💼 Objetivos

- Realizar uma análise exploratória (EDA) em uma base de dados de cancelamentos
- Entender o comportamento dos clientes que cancelam
- Gerar insights visuais através de gráficos e comparações
- Apoiar a tomada de decisão com base em dados

---

## 🔍 Principais Insights

📌 A análise revelou os seguintes padrões:
- 📉 A maioria dos cancelamentos ocorre nos primeiros 12 meses de contrato.
- ❗ Clientes com número elevado de reclamações cancelam com mais frequência.
- 💳 Formas de pagamento influenciam diretamente na fidelização.
- 📶 Usuários com suporte técnico ativo tendem a permanecer por mais tempo.

---

## 📁 Estrutura do Projeto

```
analise_cancelamentos/
├── notebooks/
│   └── analise_cancelamentos.ipynb       # Notebook principal
├── data/
│   └── raw/
│       └── cancelamentos_sample.csv      # Base de dados
├── outputs/
│   └── figures/                          # Gráficos gerados
├── requirements.txt                     # Lista de bibliotecas
├── index.html                           # Página de apresentação (opcional para GitHub Pages)
├── .gitignore                           # Arquivos ignorados pelo Git
└── README.md                            # Documentação
```

---

## ▶️ Como Executar o Projeto

1. **Clone o repositório:**
```bash
git clone https://github.com/seunome/analise_cancelamentos.git
cd analise_cancelamentos
```

2. **Crie e ative um ambiente virtual (opcional):**
```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Execute o notebook:**
```bash
jupyter notebook notebooks/analise_cancelamentos.ipynb
```

---

## 🛠️ Tecnologias Utilizadas

- [Python](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Jupyter Notebook](https://jupyter.org/)

---

## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

---

## 👨‍💻 Autor

Desenvolvido com dedicação por **Wanderlei**.  
Caso tenha sugestões, dúvidas ou queira colaborar, fique à vontade para abrir uma issue ou pull request.

[🔗 LinkedIn (opcional)](https://linkedin.com/in/wanderlei-silva-dos-santos) • [📫 Email (opcional)](mailto:wssjlle@gmail.com)

---

> “Sem dados, você é apenas mais uma pessoa com uma opinião.” – W. Edwards Deming
