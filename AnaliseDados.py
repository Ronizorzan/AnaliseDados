import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error



#Layout da página
st.set_page_config("Análise de Dados de Vendas", layout="wide")


#Função de Carregamento dos dados 
@st.cache_data
def carregador_dados(uploaded_file):
    if uploaded_file is None:
        data = pd.read_csv("superstore_final_dataset.csv", encoding="latin-1")
        new_order = ["Order_Date", "Customer_ID", "Category", "Sales"]
        data = data[new_order]
        
        
        
    else:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))    
            data = pd.read_csv(stringio)
        
        except Exception as e:
            st.error("Erro ao aplicar codificação UTF-8 ao conjunto de dados." \
            " Tentando abordagem alternativa... Escolha corretamente as colunas no dataframe e tente novamente.")
            stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))
            data = pd.read_csv(stringio)
        
    
    return data


#Função de Geração dos Cálculos, Gráficos e Modelo ARIMA
@st.cache_resource
def gerador_de_calculos_e_graficos(data, coluna_data, coluna_id, coluna_categoria, coluna_valor, datas, meses_prever, maiores_valores):
    #Primeira Função Aninhada(Geração dos Cálculos)
    def gerador_calculos(data):              
        data = data.set_index(coluna_data)
        data.index = pd.to_datetime(data.index, format="%d/%m/%Y" if uploaded_file==None else "mixed") 
        

        #Cálculos Principais e preparação dos dados
        vendas = data.resample("MS")[[coluna_valor]].sum().reset_index() # Os Dados para treinamento do modelo não serão filtrados
        data = data.sort_index()
        if len(datas)==2: #Filtra os meses se especificado            
            #data = data.last(str(meses) + "MS") # Filtro utilizado anteriormente
            data = data.loc[(data.index >= pd.to_datetime(datas[0], format="%Y/%m/%d")) & (data.index <= pd.to_datetime(datas[1], format="%Y/%m/%d"))]
        elif len(datas)==1:
            data = data.loc[data.index >= pd.to_datetime(datas[0], format="%Y/%m/%d")]

        data[coluna_id] = data[coluna_id].astype(str) #Transforma a coluna de identificação em string para plotagem adequada
        data_copy = data.copy()
        total_vendas = data[coluna_valor].sum()
        vendas_por_categoria = data.groupby(coluna_categoria)[[coluna_valor]].sum().nlargest(10, coluna_valor)   #Top 10 categorias mais vendidas
        vendas_mensais = data.resample("MS")[[coluna_valor]].sum()
        crescimento_perc = vendas_mensais[[coluna_valor]].pct_change() *100
        ticket_medio_categoria = data.groupby(coluna_categoria)[[coluna_valor]].mean().nlargest(10, coluna_valor) #Top 10 categorias com maior ticket médio
        ticket_medio_mes = data.resample("MS")[[coluna_valor]].mean()
        melhores_clientes = data.groupby(coluna_id)[[coluna_valor]].sum()
        clientes_frequentes = data.groupby(coluna_id)[[coluna_valor]].count()
        retencao_cliente = data.groupby(coluna_id)[coluna_valor].count() #Cálculo Retenção de Clientes Geral
        taxa_retencao = retencao_cliente[retencao_cliente >1].count() / retencao_cliente.count()        
        
        #Retenção de Clientes Mensal
        total_clientes = data.resample("MS")[coluna_id].nunique()                      
        compras_por_cliente = data.groupby([pd.Grouper(freq="MS"), coluna_id]).size() #Calcula a quantidade de compras de cada cliente em cada mês.          
        clientes_recorrentes = compras_por_cliente[compras_por_cliente > 1].groupby(level=0).count() #Identifica os clientes que fizeram mais de uma compra no mesmo mês.        
        taxa_recompra = (clientes_recorrentes / total_clientes).fillna(0) #Calcula a taxa de recompra: Dividir o número de clientes recorrentes pelo total de clientes (por mês)
        clv = data.groupby(coluna_id)[[coluna_valor]].sum().mean()


        #Divisão dos Dados entre treino e teste        
        proporcao_treino = int(0.85 * vendas.shape[0])
        vendas[coluna_valor].dropna(inplace=True)
        treino = vendas.loc[: proporcao_treino, :]
        teste = vendas.loc[proporcao_treino:, :]

        
        #Treinamento do Modelo ARIMA        
        modelo_arima = auto_arima(treino[coluna_valor], start_p=0, start_q=0, d=None, max_d=5, max_q=7, D=0,            #Hiper-parâmetros generalistas
                           seasonal=True, trace=False, stepwise=True, start_P=0, start_Q=0, max_D=7, max_Q=5, m=12)
        
        
        #Avaliação do Modelo
        previsoes_teste = modelo_arima.predict(len(teste))        
        mape = mean_absolute_percentage_error(teste[coluna_valor], previsoes_teste)
        previsoes_arima = modelo_arima.predict(n_periods=meses_prever, return_conf_int=False)

        
        #Geração do gráfico de Previsão
        data_grafico = pd.date_range(teste[coluna_data].max() + pd.DateOffset(months=1), periods=meses_prever, freq="MS")
        fig_arima = px.line(x=data_grafico, y=previsoes_arima, color_discrete_sequence=["#0550DD"], markers=True,
                             title=f"Previsões do Modelo para o período selecionado de {meses_prever} meses")

        x_arima = np.arange(len(data_grafico)) # Índices numéricos para os meses
        coef = np.polyfit(x_arima, previsoes_arima, 1) # Ajuste polinomial (grau 1 para tendência)
        poly_arima = np.poly1d(coef)
        y_fit_arima = poly_arima(x_arima)
        
        fig_arima.update_layout(xaxis_title=f"Erro percentual do Modelo: {mape*100:.2f}%", yaxis_title="Valor de Vendas Previsto")
        fig_arima.update_traces(text=previsoes_arima, hovertemplate="Mês: %{x}<br>Valor Previsto: %{y}", 
                                line=dict(color="dodgerblue"), marker=dict(color="darkorange"))
        
        fig_arima.add_scatter(x=data_grafico, y=y_fit_arima, mode='lines', name='Tendência<br>Prevista', 
                              line=dict(dash='dash', color="white")) # Adição da linha de tendência
        

        #Geração do Gráfico de comparação        
        vendas_compar = vendas.sort_values(coluna_data, ascending=False).iloc[:meses_prever,:]
        fig_compar = px.line(x=vendas_compar[coluna_data], y=vendas_compar[coluna_valor], 
                             markers=True, title="Valores Históricos do período anterior")
        
        x = np.arange(len(vendas_compar)) #Índices numéricos para os meses
        y = vendas_compar[coluna_valor].values # Valores de Vendas
        coef = np.polyfit(x, y, 1) # Ajuste polinomial (grau 1 para tendência)
        poly = np.poly1d(coef)
        y_fit = poly(x)

        
        fig_compar.update_layout(xaxis_title="Mês", yaxis_title="Valor das Vendas")
        fig_compar.update_traces(text=vendas_compar[coluna_valor], textposition="top center", hovertemplate="Mês: %{x}<br>Valor: %{y} ",
                                 line=dict(color="dodgerblue"), marker=dict(color="darkorange"))
        fig_compar.add_scatter(x=vendas_compar[coluna_data], y=y_fit, mode="lines", name="Tendência", line=dict(color="white", dash="dash")) # Adição da linha de tendência

        

        return total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc, ticket_medio_categoria, \
                            ticket_medio_mes, melhores_clientes, clientes_frequentes, taxa_retencao, taxa_recompra, clv, fig_arima, fig_compar, maiores_valores, data_copy
    
    #Segunda Função Aninhada(Geração dos Gráficos)
    def gerador_graficos(total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc,ticket_medio_categoria, ticket_medio_mes, melhores_clientes, 
                          clientes_frequentes, taxa_retencao, taxa_recompra, clv, fig_arima, fig_compar, maiores_valores, data_copy): #Cálculos necessários para o gerador de gráficos

        #Gráfico de Vendas por Categoria
        categoria_mais_vendida = vendas_por_categoria[vendas_por_categoria[coluna_valor]== vendas_por_categoria[coluna_valor].max()]
        fig = px.bar(vendas_por_categoria, vendas_por_categoria.index, vendas_por_categoria[coluna_valor],  color=vendas_por_categoria[coluna_valor], color_continuous_scale="Greens",
                     title=f"{categoria_mais_vendida.index[0]} é a categoria com maior soma de vendas com um total de {categoria_mais_vendida[coluna_valor].iloc[0]:,.2f} ")
        
        fig.update_layout(xaxis_title="Categoria", yaxis_title="Total Vendido em cada categoria", template="plotly_dark")

        fig.update_traces(text=vendas_por_categoria[coluna_valor].apply(lambda x: f"R$ {x:,.2f}"), textposition="none", hovertemplate="Vendas por Categoria: R$%{y}<br>Categoria: %{x} ")
                

        #Gráfico de Vendas Totais
        fig2 = go.Figure(go.Indicator(mode="gauge+number", 
                                      value=total_vendas,
                                      title="Vendas Totais",
                                      align="center",
                                      gauge={"axis": {"range": [0, 1.25 * total_vendas]}}))
        
        fig2.add_annotation(x=0.5, y=-0.2, text="Total Vendido:  R${:,.2f}".format(total_vendas), 
                                showarrow=False, font=dict(size=19, color="white"))
        
        
        mapeamento_meses = {1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho", 7: 
                            "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"} #Dicionário para mapeamento dos meses
        
        # Preparação dos dados
        crescimento_perc = crescimento_perc.sort_index(ascending=False).iloc[:12,:]
        x = np.arange(len(crescimento_perc))  # Criando índices numéricos para os meses
        y = crescimento_perc[coluna_valor].values  # Valores de crescimento percentual

        melhor_porc = crescimento_perc[crescimento_perc[coluna_valor]== crescimento_perc[coluna_valor].max()]
        pior_porc = crescimento_perc[crescimento_perc[coluna_valor]== crescimento_perc[coluna_valor].min()]
        
        # Criar gráfico principal
        fig3 = px.line(crescimento_perc, x=crescimento_perc.index, y=crescimento_perc[coluna_valor], markers=True, color_discrete_sequence=["#1c83e1"],
                    title=f"O mês de {mapeamento_meses[melhor_porc.index.month[0]]} de {melhor_porc.index.year[0]} apresentou um crescimento percentual nas vendas de {melhor_porc[coluna_valor].iloc[0]:,.2f}%")


        # Ajuste polinomial (grau 1 para tendência)
        coef = np.polyfit(x, y, 1)
        poly = np.poly1d(coef)

        # Geração de pontos para a linha de tendência
        x_fit = x  # Mantendo a escala original
        y_fit = poly(x_fit)  # Aplicando a função polinomial


        fig3.update_layout(yaxis_title="Variação Percentual(%)", 
                        xaxis_title=f"{mapeamento_meses[pior_porc.index.month[0]]} de {pior_porc.index.year[0]} foi o mês com pior variação percentual: {pior_porc[coluna_valor].iloc[0]:,.2f}%")

        fig3.update_traces(text=crescimento_perc[coluna_valor], hovertemplate="Variação Percentual nas Vendas: %{y}%<br>Mês: %{x}",
                           marker=dict(color="#f7fbff"))

        # Adicionar linha de tendência polinomial corrigida
        fig3.add_scatter(x=crescimento_perc.index, y=y_fit, mode='lines', name="Tendência", line=dict(dash="dash", color="#f7fbff")) # Adição da linha de tendência


        
        #Gráfico de Vendas Mensais
        vendas_mensais = vendas_mensais.sort_index(ascending=False).iloc[:12,:]
        melhor_mes = vendas_mensais[vendas_mensais[coluna_valor]== vendas_mensais[coluna_valor].max()]
        pior_mes = vendas_mensais[vendas_mensais[coluna_valor]== vendas_mensais[coluna_valor].min()]
        fig4 = px.bar(vendas_mensais, vendas_mensais.index, vendas_mensais[coluna_valor], color_continuous_scale="Blues", color=vendas_mensais[coluna_valor],
            title=f"O mês de {mapeamento_meses[melhor_mes.index.month[0]]} de {melhor_mes.index.year[0]} foi o melhor mês com um total de R${melhor_mes[coluna_valor].iloc[0]:,.2f} vendido")
                      
        
        fig4.update_layout(yaxis_title="Total Mensal em Vendas", xaxis={"tickangle": 0}, 
                    xaxis_title=f"{mapeamento_meses[pior_mes.index.month[0]]} de {pior_mes.index.year[0]} foi o mês com o menor total de vendas: R${pior_mes[coluna_valor].iloc[0]:,.2f} ") 

        fig4.update_traces(text=vendas_mensais[coluna_valor], textposition="none", hovertemplate="Vendas Totais: R$%{y}<br>Mês: %{x} ")
        

        #Gráfico de Ticket Médio por Categoria        
        maior_ticket_cat = ticket_medio_categoria[ticket_medio_categoria[coluna_valor]== ticket_medio_categoria[coluna_valor].max()]
        ticket_medio_categoria = ticket_medio_categoria.sort_values(coluna_valor, ascending=True)
        fig5 = px.bar(ticket_medio_categoria, y=ticket_medio_categoria.index, x=ticket_medio_categoria[coluna_valor], color=ticket_medio_categoria[coluna_valor],
                        title=f"O maior ticket médio foi da categoria {maior_ticket_cat.index[0]} com um valor de R${maior_ticket_cat[coluna_valor].iloc[0]:,.2f}")
        fig5.update_layout(xaxis_title="Valor do Ticket", yaxis_title="Categoria")
        fig5.update_traces(text=ticket_medio_categoria[coluna_valor].apply(lambda x: f"R$ {x:,.2f}"), textposition="inside", hovertemplate="Ticket Médio por Categoria: %{text}<br>Categoria: %{y}")
        
        #Gráfico de Ticket Médio por Mês        
        ticket_medio_mes = ticket_medio_mes.sort_index(ascending=False).iloc[:12,:]
        maior_ticket_mes = ticket_medio_mes[ticket_medio_mes[coluna_valor]== ticket_medio_mes[coluna_valor].max()]
        menor_ticket_mes = ticket_medio_mes[ticket_medio_mes[coluna_valor]== ticket_medio_mes[coluna_valor].min()]
        fig6 = px.bar(ticket_medio_mes, ticket_medio_mes.index, ticket_medio_mes[coluna_valor], color=ticket_medio_mes[coluna_valor],
            title=f"{mapeamento_meses[maior_ticket_mes.index.month[0]]} de {maior_ticket_mes.index.year[0]} apresentou o maior ticket médio com um valor de {maior_ticket_mes[coluna_valor].iloc[0]:.2f}")
        fig6.update_layout(yaxis_title="Valor do Ticket", 
                    xaxis_title=f"{mapeamento_meses[menor_ticket_mes.index.month[0]]} de {menor_ticket_mes.index.year[0]} foi o mês com o menor ticket médio: {menor_ticket_mes[coluna_valor].iloc[0]:,.2f}")
        fig6.update_traces(text=ticket_medio_mes[coluna_valor].apply(lambda x: f"R$ {x:,.2f}"), textposition="none", hovertemplate="Valor do Ticket: %{text}<br>Mês: %{x} ")

        #Gráfico de Melhores Clientes
        melhores_clientes = melhores_clientes.sort_values(coluna_valor, ascending=False).iloc[:maiores_valores,:].reset_index()
        melhor_cliente = melhores_clientes[melhores_clientes[coluna_valor]== melhores_clientes[coluna_valor].max()]
        fig7 = px.bar(melhores_clientes, melhores_clientes[coluna_id].apply(lambda x: "(" + x + ")"), melhores_clientes[coluna_valor],
                       color=melhores_clientes[coluna_valor], color_continuous_scale="Greens",
                      title=f"ID do Melhor Cliente: {melhor_cliente[coluna_id].iloc[0]}<br>Valor Total Gasto {melhor_cliente[coluna_valor].iloc[0]} ")
        fig7.update_layout(xaxis_title="ID dos Melhores Clientes", yaxis_title="Valor Gasto")
        fig7.update_traces(text=melhores_clientes[coluna_valor].apply(lambda x: f"R$ {x:,.2f}"), textposition="none", hovertemplate="Gastos do Cliente: R$%{text}<br>ID do Cliente: %{x}")

        
        #Gráfico de Clientes Mais Frequentes
        clientes_frequentes = clientes_frequentes.sort_values(coluna_valor, ascending=False).iloc[:maiores_valores,:].reset_index()
        cliente_mais_frequente = clientes_frequentes[clientes_frequentes[coluna_valor]== clientes_frequentes[coluna_valor].max()]
        fig8 = px.bar(clientes_frequentes, clientes_frequentes[coluna_id].apply(lambda x: "(" + x + ")"), 
                      clientes_frequentes[coluna_valor], color=clientes_frequentes[coluna_valor],  color_continuous_scale="Greens",
                      title=f"ID do Cliente mais frequente: {cliente_mais_frequente[coluna_id].iloc[0]}<br>Compras Realizadas: {cliente_mais_frequente[coluna_valor].iloc[0]} ")
        fig8.update_layout(xaxis_title="Clientes Mais Frequentes", yaxis_title="Quantidade de Compras realizadas")
        fig8.update_traces(text=clientes_frequentes[coluna_valor], textposition="auto", hovertemplate="Número de Compras do Cliente: %{y}<br>ID do Cliente: %{x} ")


        #Gráfico de Retenção de Clientes
        cor = "red" if taxa_retencao <0.2 else "green"
        fig9 = go.Figure(go.Indicator(mode="gauge+number", 
                                    value= taxa_retencao*100,
                                    title={"text": f"Taxa de Recorrência de Clientes: {taxa_retencao*100:.2f}% "},
                                    gauge={"axis": {"range": [0, 100]},
                                           "bar": {"color": cor}}))
        fig9.add_annotation(x=0.5, y=-0.2, 
                            text=f"Aproximadamente {round(taxa_retencao*10)} de cada 10 clientes voltaram a comprar",
                            showarrow=False, font=dict(size=25, color=cor))
        
        taxa_recompra = taxa_recompra*100
        maior_recorrencia = taxa_recompra[taxa_recompra.values== taxa_recompra.values.max()]
        menor_recorrencia = taxa_recompra[taxa_recompra.values== taxa_recompra.values.min()]
        fig10 = px.line(taxa_recompra, x=taxa_recompra.index, y=taxa_recompra.values, color_discrete_sequence=[cor])
        fig10.update_layout(xaxis_title=f"{mapeamento_meses[menor_recorrencia.index.month[0]]} de {menor_recorrencia.index.year[0]} apresentou a menor taxa de recorrência: {menor_recorrencia.values[0]:.2f}%",
            yaxis_title="Taxa de Recorrência(%)", title=f"No mês de {mapeamento_meses[maior_recorrencia.index.month[0]]} de {maior_recorrencia.index.year[0]} houve uma taxa de recompra de {maior_recorrencia.values[0]:.2f}%")
        fig10.update_traces(text=taxa_recompra.apply(lambda x: f"{x:.2f}%"), textposition="bottom center", hovertemplate="Taxa de Recompra: %{text}<br>Mês: %{x}")
        
        return fig, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, clv, fig_arima, fig_compar, data_copy
    
    
    total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc, ticket_medio_categoria, ticket_medio_mes, \
            melhores_clientes, clientes_frequentes, taxa_retencao, taxa_recompra, clv, fig_arima, fig_compar, maiores_valores, data_copy = gerador_calculos(data)
        
    return gerador_graficos(total_vendas, vendas_por_categoria, vendas_mensais, crescimento_perc,
                          ticket_medio_categoria, ticket_medio_mes, melhores_clientes, clientes_frequentes, taxa_retencao, taxa_recompra, clv, fig_arima, fig_compar, maiores_valores, data_copy)
    

#COnfiguração da barra lateral
with st.sidebar:    
    st.markdown(":blue[**Configuração das Análises**]")
    try:
        uploaded_file = st.file_uploader(":blue[Selecionar arquivo]", type="csv", help="Se nenhum arquivo for selecionado, \
                                                    \n a aplicação usará dados de exemplo")
        dados = carregador_dados(uploaded_file)  #dados brutos
    except Exception as error:
        st.error("Erro ao carregar o arquivo. Por favor selecione um conjunto de dados compatível e atualize corretamente as colunas\
                    de data, id do cliente , categoria e valor no seu arquivo CSV.")
    with st.expander("Selecionar o tipo de visualização", expanded=True):
        visualizacao = st.radio("Selecione o tipo de análise", ["Vendas por Categoria", "Vendas por Mês", "TickedMédio", "Clientes Engajados",
                                                                "Taxa de Recorrência", "Projeção"], help="Selecione o tipo de análise que gostaria de visualizar")        
    if dados.shape[1]<4: 
        st.error("Por Favor selecione um conjunto de dados válidos. O conjunto de dados deve conter "
                    "colunas de \'data, id do cliente, categoria e valor\'")      #Mensagem de erro se conjunto de dados carregado for incompatível        
    st.markdown(":blue[**Gostaria de personalizar as visualizações?**]", help="Selecione as configurações abaixo para personalizar \
                                    \n a exibição de meses, clientes e previsões", unsafe_allow_html=False)
    with st.expander("Configurações adicionais"):        
        datas = st.date_input("Insira 1 ou 2 datas para filtrar nos gráficos", [], help="Escolha uma única data se quiser inserir apenas a data inicial,\
                              \n ou duas se quiser filtrar data inicial e final, respectivamente.")
        maiores_valores = st.slider("Selecione o número de clientes que deseja exibir nos gráficos(Clientes Engajados)", 2, 20, 10, 1)                
        meses_prever = st.slider("Selecione o número de Meses que deseja prever(Projeção)", min_value=2, max_value=12, value=12, step=1)            
    st.markdown(":blue[**Fez o upload de um arquivo?**]", help="Se você fez o upload de um arquivo. Clique abaixo \
                \n para selecionar as  colunas necessárias para as análises, \
                    \n caso contrário a aplicação pode apresentar um ERRO", unsafe_allow_html=False)
    with st.expander("Clique aqui!", expanded=False):
        coluna_data = st.selectbox("Selecione a coluna de data", dados.columns, index=0)    
        coluna_id = st.selectbox("Selecione a coluna de identificação do cliente", dados.columns, index=1)
        coluna_categoria = st.selectbox("Selecione a coluna de categoria", dados.columns, index=2)
        coluna_valor = st.selectbox("Selecione a coluna de Valor", dados.columns, index=3)

    st.markdown(""" 
            <style>.footer {
            background: linear-gradient(135deg, #a0a0a0, #efefef);
            padding: 12px 17px;
            border-radius: 30px;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-top: 10px;
            margin-bottom: 15px;
            color: #262730;
            box-shadow: 0 8px 10px rgba(10,10,10,1.0);
        }
        .footer p {
            margin: 6px 0;
        }
        .footer a {
            margin: 0 10px;
            display: inline-block;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            text-decoration: none;
        }
        .footer a:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        }
        .footer img {
            height: 42px;
            width: auto;
            vertical-align: middle;
        }
        </style>
        <div class="footer">
            <p><strong>Desenvolvido por: Ronivan</strong></p>
            <a href="https://github.com/Ronizorzan" target="_blank" title="GitHub">
                <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub">
            </a>
            <a href="https://www.linkedin.com/in/ronivan-zorzan-barbosa" target="_blank" title="LinkedIn">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn">
            </a>
            <a href="https://share.streamlit.io/user/ronizorzan" target="_blank" title="Projetos Streamlit">
                <img src="https://images.seeklogo.com/logo-png/44/1/streamlit-logo-png_seeklogo-441815.png" alt="Streamlit Community">
            </a>
        </div>
        """, unsafe_allow_html=True) # Informações do desenvolvedor
    
    processar = st.button(":blue[Processar os dados]" )

if processar:
    try:
        if len(datas)==2 and datas[1] <= datas[0]: # Erro se a data inicial for maior que a data final
                raise ValueError("A data inicial deve ser menor que a data final")
        progress = st.progress(50, "Aguarde... Gerando Cálculos e gráficos")    
        fig, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, clv, grafico_arima, fig_compar, data = gerador_de_calculos_e_graficos(dados, coluna_data, coluna_id, 
                                                                                                                                    coluna_categoria, coluna_valor, datas, meses_prever, maiores_valores)
            
        progress.progress(100, "Cálculos e Gráficos Gerados...")
        
        
        col1, col2 = st.columns([0.6,0.4], gap="medium")

        #Visualização da primeira Página
        if visualizacao=="Vendas por Categoria":
            with col1:
                st.subheader("Vendas totais por categoria")
                st.plotly_chart(fig, use_container_width=True, theme="streamlit" )

                df_nulos = dados.loc[:, [coluna_data, coluna_id, coluna_categoria, coluna_valor]] # Verificação e remoção de valores nulos 
                if df_nulos.isnull().values.any(): # Verifica se existem valores nulos no dataframe
                    nulos = df_nulos.isnull().sum().sum() # Soma todos os valores nulos para exibição na aplicação
                    data = data.reset_index().dropna() # Remove valores nulos do dataframe para evitar erros nos gráficos
                    data = data.set_index(coluna_data) # Reconfiguração do índice do dataframe com a coluna de data
                    

                st.markdown(f""":green[**Descrição:**] *O gráfico acima mostra o total vendido em cada categoria no período selecionado.\
                            \nExibindo resultados nos gráficos desde a data inicial de ***{(data.index.date.min()).strftime("%d-%m-%Y")}***
                            até a data final de* ***{(data.index.date.max()).strftime("%d-%m-%Y")}.***""")
                st.markdown("<hr style='border:1px solid green'>", unsafe_allow_html=True)                
                st.markdown(""":green[**Informação:**] *Para garantir uma visualização mais robusta dos dados filtre as datas com um período
                        de pelo menos 4 meses para obter uma visualização personalizada e dinâmica dos dados. Caso contrário
                        alguns gráficos podem apresentar valores nulos ou vazios, o que pode dificultar a análise.
                        Utilizando essa abordagem você garante a minimização de gráficos incompletos e maximização de resultados relevantes nas análises.*""")
                st.markdown(":green[**Dica:**]  *Altere as datas na barra lateral ao lado e veja como os valores nos gráficos se atualizam... \
                            Alternativamente você pode mantê-lo em branco para obter uma visão menos focada utilizando todos os dados disponíveis*")            
                

            with col2:
                st.subheader("Vendas Totais")
                st.plotly_chart(fig2, use_container_width=True)            
                st.markdown(f":green[***Informação:***] *O gráfico acima mostra o total em vendas no período selecionado.\
                            \nTotal de vendas concretizadas no período: <span style='font-size: 20px'>{len(data)}</span> vendas.*", unsafe_allow_html=True)                
                
                if df_nulos.isnull().values.any(): # Exibição de valores nulos na aplicação para maior transparência
                    st.warning(f"Foram detectados {nulos} valores nulos no conjunto de dados. Esses valores serão desconsiderados nas análises. ")

                st.markdown("<hr style='border:1px solid green'>", unsafe_allow_html=True)            
                if len(datas) >0: # Se filtros de datas forem aplicados
                    dados_agrupados = data.copy()  # Trabalhar com uma cópia  
                    dados_agrupados = dados_agrupados.resample("MS")[coluna_valor].sum()  # Totais somados mensalmente
                    soma_periodo_atual = dados_agrupados.sum()  # Soma total do período atual
                    
                    # Ajuste da data inicial dos meses anteriores
                    inicio_periodo_anterior = dados_agrupados.index.min() - pd.DateOffset(months=len(dados_agrupados))

                    # Filtragem do período anterior
                    dados_completos = dados.dropna().copy()
                    dados_completos[coluna_data] = pd.to_datetime(dados_completos[coluna_data], format='mixed')
                    dados_completos = dados_completos.set_index(coluna_data)[coluna_valor].resample("MS").sum()                
                    periodo_anterior = dados_completos.loc[
                        (dados_completos.index >= inicio_periodo_anterior) & (dados_completos.index < dados_agrupados.index.min())
                    ]
                    soma_periodo_anterior = periodo_anterior.sum()  # Soma total do período anterior
                                    
                    st.metric(label=f"Total vendido no período de comparação disponível.\
                            De {(inicio_periodo_anterior).strftime('%d-%m-%Y')} até {(dados_agrupados.index.min() - pd.DateOffset(days=1)).strftime('%d-%m-%Y')}.\
                            \nTotal de {len(periodo_anterior)} meses", value=f"R$ {soma_periodo_anterior:,.2f}", delta=f"{(soma_periodo_anterior - soma_periodo_atual):,.2f}", delta_color="normal" )
                    
                                        
                    variacao_absoluta = soma_periodo_atual - soma_periodo_anterior
                    sinal = "+" if variacao_absoluta >=0 else ""
                    st.metric(label="Variação do período atual em relação ao período de comparação", value=f"R$ {sinal} {(variacao_absoluta):,.2f}",
                            delta=f"{((soma_periodo_atual / soma_periodo_anterior) - 1)* 100:,.2f}%", delta_color="normal")
                    
                    if len(periodo_anterior)< len(dados_agrupados):
                        st.warning(f"""*O conjunto de dados não é grande o suficiente para gerar um período de comparação justo.
                                   Há apenas {len(periodo_anterior)} meses restantes para o período de comparação,
                                   mas foram utilizados {len(dados_agrupados)} meses na análise principal.
                                   Para uma análise mais precisa considere utilizar um conjunto de dados maior 
                                   ou filtre um período de tempo menor.*""")
                    
                else:
                    st.markdown(""":green[**Informação importante:**] *Utilize as configurações adicionais na barra lateral para filtrar datas específicas.
                                Ao filtrar as datas uma comparação entre períodos será exibida, mas note que essa comparação é limitada aos dados disponíveis no conjunto de dados.
                                Portanto utilize a descrição no gráfico ao lado para identificar o período disponível no conjunto de dados (caso ainda não saiba)
                                e gerar as análises de sua preferência.*""")
                    
                

        #Visualização da segunda Página
        elif visualizacao=="Vendas por Mês":
            col1, col2 = st.columns(2, gap="large")
            with col1:            
                st.subheader("Variação Percentual das Vendas")          
                st.plotly_chart(fig3, use_container_width=True)            
                st.markdown("*O gráfico acima exibe o crescimento percentual das Vendas*")
                st.markdown("<hr style='border:1px solid #1c83e1'>", unsafe_allow_html=True)
                st.markdown(":blue[**Dica:**]  *Analise a variação percentual nas vendas de acordo com o mês anterior \
                e veja como elas mudam ao longo do tempo comparando com o gráfico de vendas ao lado \
                Insira o número de meses ao lado para um visão mais detalhada se necessário*")

            with col2:
                st.subheader("Vendas Mensais")  
                st.plotly_chart(fig4, use_container_width=True)            
                st.markdown("*O gráfico acima exibe o total de Vendas Mensais*")
                st.markdown("<hr style='border:1px solid #1c83e1'>", unsafe_allow_html=True)
                st.markdown(":blue[**Informação:**] *Analise o gráfico acima para obter uma visão totalizada das vendas de cada mês \
                            e veja a relação com a variação percentual das vendas ao lado \
                            Todos os gráficos possuem ícones acima que permitem interação com os gráficos."
                            "Arrastar, aplicar zoom, exibir em tela cheia, fazer o download da imagem \
                                são apenas alguns dos recursos disponíveis nos gráficos*")

        #Visualização da terceira Página
        elif visualizacao=="TickedMédio":
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.subheader("Ticket Médio por Categoria")
                st.plotly_chart(fig5, use_container_width=True)
                st.markdown("*O gráfico acima mostra o ticket médio por categoria*")
                st.markdown("<hr style='border:1px solid #1c83e1'>", unsafe_allow_html=True)            
                st.markdown(":blue[**Descrição:**] *Veja quanto cada venda em determinada categoria \
                            representa em média.*")

            with col2:
                st.subheader("Ticket Médio por Mês")
                st.plotly_chart(fig6, use_container_width=True)
                st.markdown("*O gráfico acima mostra o ticket Médio por Mês*")
                st.markdown("<hr style='border:1px solid #1c83e1'>", unsafe_allow_html=True)            
                st.markdown(":blue[**Descrição:**] *Veja o quanto o valor médio de cada venda \
                            pode variar de um mês para o outro*")

        #Visualização da quarta Página
        elif visualizacao=="Clientes Engajados":
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.subheader("*Clientes com Maiores Gastos*")
                st.plotly_chart(fig7, use_container_width=True)
                st.markdown("*O gráfico acima exibe os clientes com os maiores gastos*")
                st.markdown("<hr style='border:1px solid green'>", unsafe_allow_html=True)            
                st.markdown(":green[**Descrição:**] *Veja os clientes com maiores gastos \
                            no período selecionado. Opcionalmente, use o slider \
                            ao lado para filtrar o número de clientes desejado*")

            with col2:
                st.subheader("**Clientes Mais Frequentes**")
                st.plotly_chart(fig8, use_container_width=True)        
                st.markdown("*O gráfico acima exibe os clientes com maior número de compras*")
                st.markdown("<hr style='border:1px solid green'>", unsafe_allow_html=True)            
                st.markdown(":green[**Descrição:**] *Veja com que frequência os melhores clientes compraram na empresa \
                            no período selecionado.*")
            
        #Visualização da quinta Página
        elif visualizacao== "Taxa de Recorrência":        
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("*Clientes Recorrentes*")
                st.plotly_chart(fig9, use_container_width=True)
                st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
                st.markdown(":green[**Descrição:**] *O gráfico acima mostra a proporção de clientes que voltaram a comprar durante o período selecionado*")            
                st.markdown(f"*Valor Médio Vitalício do Cliente(CLV):* <span style='font-size: 25px; font-weight:bold; color: green'>R${clv.iloc[0]:,.2f}</span>", unsafe_allow_html=True)
                st.markdown(":green[**Informação:**] *CLV é o valor médio que um cliente gasta durante todo o seu tempo \
                            como cliente da empresa. Observe que esse valor é calculado considerando todas as compras dos clientes \
                            dentro do período selecionado e pode apresentar uma variação significativa \
                            de acordo com o período selecionado*")

            with col2:
                st.subheader("*Taxa de Recorrência Mensal*")
                st.plotly_chart(fig10, use_container_width=True)
                st.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
                st.markdown(":green[**Descrição:**] *O gráfico acima mostra a proporção de clientes com compras recorrentes em um mesmo mês \
                            Altas taxas de recompra indicam que os clientes tendem a ser mais fiéis à empresa realizando mais de uma compra \
                            no mesmo mês enquanto taxas menores indicam baixo engajamento mensal dos clientes. \
                            Observe também que o engajamento tende a diminuir de forma exponencial à medida que \
                            o número de clientes da empresa aumenta*")
                

        #Visualização da sexta Página
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("*Vendas do período anterior*")
                st.plotly_chart(fig_compar, use_container_width=True)
                st.markdown(f"*O gráfico acima mostra as vendas do período anterior ao período previsto de **{meses_prever}** meses*")
                st.markdown("<hr style='border:1px solid #1E60FF'>", unsafe_allow_html=True)                
                st.markdown(":blue[**Informação:**] *É importante que você analize as previsões do modelo e as compare com os dados disponíveis, \
                    pois é possível que o modelo capte padrões nas vendas como altas, quedas, variações, sazonalidades etc. \
                    Porém dependendo do caso pode ser necessário utilizar outras abordagens como machine learning ou redes neurais, por exemplo. \
                    Também é interessante utilizar conjuntos de dados de tamanhos ao menos razoável para que esses padrões possam ser captados* ")
            
            with col2:
                st.subheader("*Previsões para o período selecionado*")
                st.plotly_chart(grafico_arima, use_container_width=True)                
                st.markdown(f"*O gráfico acima mostra as previsões do modelo para o período selecionado de **{meses_prever}** meses*")
                st.markdown("<hr style='border:1px solid #1E60FF'>", unsafe_allow_html=True)
                st.markdown(":blue[**Informação:**] *Note que este modelo foi construído para ser um modelo generalista que se adapta aos dados \
                            da melhor maneira possível, porém à depender da qualidade e quantidade dos dados disponíveis as previsões \
                            podem ter uma certa variação. Por isso é importante se atentar para o erro percentual do modelo \
                            que está disponibilizado abaixo do gráfico com as previsões*")
    except Exception as error:
        st.warning("Atenção. É extremamente importante que você selecione corretamente as colunas do dataframe na barra lateral caso você\
                   tenha feito o upload de um arquivo. Também é importante limpar os filtros de data pois a inserção de datas inexistentes\
                   no conjunto de dados pode gerar dataframes vazios e, consequentemente, erros na aplicação .")
        st.error(f"Erro ao processar os dados inseridos. Por favor, verifique a compatibilidade do dataframe\
                 com a aplicação e tente novamente. Erro técnico encontrado: {error}")
                



    
    


















