import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO





st.set_page_config("Análise de Dados de Vendas", layout="wide")


@st.cache_resource
def carregador_dados(uploaded_file):
    if uploaded_file is None:
        data = pd.read_csv("superstore_final_dataset.csv", encoding="latin-1")
        new_order = ["Order_Date", "Customer_ID", "Category", "Sales"]
        data = data[new_order]
        
        
    else:
        stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))    
        data = pd.read_csv(stringio)
    
    return data


@st.cache_resource
def gerador_calculos(data, coluna_data, coluna_id, coluna_categoria, coluna_valor):
    data = data.set_index(coluna_data)
    data.index = pd.to_datetime(data.index, format="%d/%m/%Y" if uploaded_file==None else None)
    

    total_vendas = data[coluna_valor].sum()
    vendas_por_categoria = data.groupby(coluna_categoria)[coluna_valor].sum()
    vendas_mensais = data.resample("ME")[coluna_valor].sum()    
    crescimento_perc = vendas_mensais.head(12).pct_change() *100    
    ticket_medio_categoria = data.groupby(coluna_categoria)[coluna_valor].mean()    
    ticket_medio_cliente = data.groupby(coluna_id)[coluna_valor].mean()
    melhores_clientes = data.groupby(coluna_id)[coluna_valor].sum().nlargest(10)
    melhores_meses = data.resample("ME")[coluna_valor].sum(coluna_valor).nlargest(10)
    retencao_cliente = data.groupby(coluna_id)[coluna_id].nunique()
    taxa_retencao = retencao_cliente[retencao_cliente >1].count() / retencao_cliente.count()
    vlt = data.groupby(coluna_id)[[coluna_valor]].mean()

    return total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc, ticket_medio_categoria, ticket_medio_cliente, melhores_clientes, melhores_meses, taxa_retencao, vlt

def gerador_graficos(total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc, ticket_medio_categoria, ticket_medio_cliente, melhores_clientes, melhores_meses, taxa_retencao, vlt):

    #Gráfico 1
    fig = px.bar(vendas_por_categoria, vendas_por_categoria.index, vendas_por_categoria.values, title="Vendas por Categoria", color=vendas_por_categoria.values)
    fig.update_layout(xaxis_title="Categoria", yaxis_title="Total em Vendas")
    

    fig2 = go.Figure(go.Indicator(mode="gauge+number",
                                  value=total_vendas.round(2),
                                       title={"text": f"Valor total das Vendas: {total_vendas.round(2)} "},
                                       align="center", 
                                       gauge= {"axis": {"range": [0, 1.25 * total_vendas]}}))
    
    
    
    fig3 = px.line(crescimento_perc, crescimento_perc.index, crescimento_perc.values)
    fig3.update_layout(xaxis_title="Mês", yaxis_title="Crescimento Percentual")

    
    vendas_mensais = vendas_mensais[ vendas_mensais.index.sort_values(ascending=True)].head(12)
    fig4 = px.bar(vendas_mensais, vendas_mensais.index, vendas_mensais.values, color=vendas_mensais.values)
    fig4.update_layout(xaxis_title="Mês", yaxis_title="Total Mensal em Vendas")
    

    ticket_medio_categoria = ticket_medio_categoria[ ticket_medio_categoria.index.sort_values(ascending=True)].head(12)
    fig5 = px.bar(ticket_medio_categoria, ticket_medio_categoria.index, ticket_medio_categoria.values, color=ticket_medio_categoria.values)
    fig5.update_layout(xaxis_title="Mês", yaxis_title="Ticket Médio por Categoria")
    
    ticket_medio_cliente = ticket_medio_cliente[ticket_medio_cliente.index.sort_values(ascending=True)].head(10)
    fig6 = px.bar(ticket_medio_cliente, ticket_medio_cliente.index, ticket_medio_cliente.values, color=ticket_medio_cliente.values)
    fig6.update_layout(xaxis_title="Ticket Médio Melhores Clientes", yaxis_title="Valor do Ticket")

    
    fig7 = px.bar(melhores_clientes, melhores_clientes.index, melhores_clientes.values, color=melhores_clientes.values)
    fig7.update_layout(xaxis_title="Melhores Clientes", yaxis_title="Valor Gasto")

    fig8 = px.bar(melhores_meses, melhores_meses.index, melhores_meses.values, color=melhores_meses.values)
    fig8.update_layout(xaxis_title="Melhores Meses", yaxis_title="Valor Vendido")


    fig9 = go.Figure(go.Indicator(mode="gauge+number", 
                                  value=taxa_retencao.round(2),
                                  title={"text": f"Taxa de Retenção de Clientes: {taxa_retencao.round(2)} "},
                                  gauge={"axis": {"range": [0, 100]}}))
    
    return fig, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9


with st.sidebar.expander("Configuração das Análises"):
    uploaded_file = st.file_uploader("Carrega dados apartir do computador", type="csv")
    dados = carregador_dados(uploaded_file)
    visualizacao = st.radio("Selecione o tipo de visualização", options=["Vendas por Categoria", "Vendas por Mês", "Ticket Médio", "Maiores Valores", "Taxa de Retenção"])    
    coluna_data = st.selectbox("Selecione a coluna de data", dados.columns, index=0)    
    coluna_id = st.selectbox("Selecione a coluna de identificação do cliente", dados.columns, index=1)
    coluna_categoria = st.selectbox("Selecione a coluna de categoria", dados.columns, index=2)
    coluna_valor = st.selectbox("Selecione a coluna de Valor", dados.columns, index=3)
    processar = st.button("Processar os dados")
if processar:
    total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc, ticket_medio_categoria, ticket_medio_cliente, melhores_clientes, melhores_meses, taxa_retencao, vlt= gerador_calculos(dados, coluna_data, coluna_id, coluna_categoria, coluna_valor)
    fig, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9 = gerador_graficos(total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc, ticket_medio_categoria, ticket_medio_cliente, melhores_clientes, melhores_meses, taxa_retencao, vlt)
    
    
    col1, col2 = st.columns([0.6,0.4])

    if visualizacao=="Vendas por Categoria":
        with col1:
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)


    elif visualizacao=="Vendas por Mês":
        col1, col2 = st.columns([0.4, 0.6])
        with col1:            
            st.plotly_chart(fig3, use_container_width=True)
            

        with col2:
            st.plotly_chart(fig4, use_container_width=True)

    elif visualizacao=="Ticket Médio":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            st.plotly_chart(fig6, use_container_width=True)

    elif visualizacao=="Maiores Valores":
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig7, use_container_width=True)

        with col2:
            st.plotly_chart(fig8, use_container_width=True)        
        
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig9, use_container_width=True)

        with col2:
            st.text(vlt)
    
    


















