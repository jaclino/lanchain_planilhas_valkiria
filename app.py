# ====== IMPORTAÇÕES ======
import streamlit as st           # Biblioteca para criar apps web interativos de forma simples
import pandas as pd              # Biblioteca para manipulação e análise de dados
import os                        # Biblioteca para interagir com variáveis de ambiente e sistema operacional
from langchain_groq import ChatGroq           # Cliente LLM usando o provedor Groq
from langchain.prompts import PromptTemplate  # Classe para criar prompts dinâmicos para LLM
from langchain.agents import create_react_agent, AgentExecutor  # Ferramentas para criar e executar agentes ReAct
from ferramentas import criar_ferramentas     # Função customizada para criar ferramentas baseadas no DataFrame

# ====== CONFIGURAÇÃO INICIAL DO APP ======
st.set_page_config(page_title="Assistente de análise de dados com IA", layout="centered")
st.title("🦜 Assistente de análise de dados com IA")

# Descrição sobre a funcionalidade do app
st.info("""
Este assistente utiliza um agente, criado com Langchain, para te ajudar a explorar, analisar e visualizar dados.
Você pode fazer upload de um CSV e:
- Gerar relatórios automáticos de informações gerais e estatísticas descritivas
- Fazer perguntas simples sobre os dados
- Criar gráficos automaticamente
""")

# ====== UPLOAD DO CSV ======
st.markdown("### 📁 Faça upload do seu arquivo CSV")
arquivo_carregado = st.file_uploader("Selecione um arquivo CSV", type="csv", label_visibility="collapsed")

if arquivo_carregado:
    # Lê o arquivo CSV em um DataFrame (usando encoding e separador específicos)
    df = pd.read_csv(arquivo_carregado, encoding='cp1252', sep=';')
    st.success("Arquivo carregado com sucesso!")
    
    # Mostra as primeiras linhas do DataFrame para conferência
    st.markdown("### 🔍 Primeiras linhas do DataFrame")
    st.dataframe(df.head())

    # ====== CONFIGURAÇÃO DO LLM ======
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Busca a chave de API do Groq nas variáveis de ambiente
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",  # Modelo LLM utilizado
        temperature=0                  # Respostas determinísticas
    )

    # ====== CRIAÇÃO DAS FERRAMENTAS ======
    tools = criar_ferramentas(df)  # Cria as ferramentas específicas para análise do DataFrame

    # Captura prévia das primeiras linhas do DataFrame para incluir no prompt
    df_head = df.head().to_markdown()

    # ====== TEMPLATE DE PROMPT PARA O AGENTE ======
    prompt_react_pt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"df_head": df_head},
        template="""
        Você é um assistente que sempre responde em português.
        ...
        Question: {input}  
        Thought: {agent_scratchpad}
        """
    )

    # ====== CRIAÇÃO E EXECUÇÃO DO AGENTE ======
    agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt)
    orquestrador = AgentExecutor(
        agent=agente,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # ====== SEÇÃO: AÇÕES RÁPIDAS ======
    st.markdown("---")
    st.markdown("## ⚡ Ações rápidas")

    # --- Relatório de informações gerais ---
    if st.button("📄 Relatório de informações gerais", key="botao_relatorio_geral"):
        with st.spinner("Gerando relatório 🦜"):
            resposta = orquestrador.invoke({"input": "Quero um relatório com informações sobre os dados"})
            st.session_state['relatorio_geral'] = resposta["output"]

    if 'relatorio_geral' in st.session_state:
        with st.expander("Resultado: Relatório de informações gerais"):
            st.markdown(st.session_state['relatorio_geral'])
            st.download_button(
                label="📥 Baixar relatório",
                data=st.session_state['relatorio_geral'],
                file_name="relatorio_informacoes_gerais.md",
                mime="text/markdown"
            )

    # --- Relatório de estatísticas descritivas ---
    if st.button("📄 Relatório de estatísticas descritivas", key="botao_relatorio_estatisticas"):
        with st.spinner("Gerando relatório 🦜"):
            resposta = orquestrador.invoke({"input": "Quero um relatório de estatísticas descritivas"})
            st.session_state['relatorio_estatisticas'] = resposta["output"]

    if 'relatorio_estatisticas' in st.session_state:
        with st.expander("Resultado: Relatório de estatísticas descritivas"):
            st.markdown(st.session_state['relatorio_estatisticas'])
            st.download_button(
                label="📥 Baixar relatório",
                data=st.session_state['relatorio_estatisticas'],
                file_name="relatorio_estatisticas_descritivas.md",
                mime="text/markdown"
            )

    # ====== SEÇÃO: PERGUNTAS SOBRE OS DADOS ======
    st.markdown("---")
    st.markdown("## 🔎 Perguntas sobre os dados")
    pergunta_sobre_dados = st.text_input("Faça uma pergunta sobre os dados")
    if st.button("Responder pergunta", key="responder_pergunta_dados"):
        with st.spinner("Analisando os dados 🦜"):
            resposta = orquestrador.invoke({"input": pergunta_sobre_dados})
            st.markdown(resposta["output"])

    # ====== SEÇÃO: CRIAÇÃO DE GRÁFICOS ======
    st.markdown("---")
    st.markdown("## 📊 Criar gráfico com base em uma pergunta")
    pergunta_grafico = st.text_input("Digite o que deseja visualizar")
    if st.button("Gerar gráfico", key="gerar_grafico"):
        with st.spinner("Gerando o gráfico 🦜"):
            orquestrador.invoke({"input": pergunta_grafico})
