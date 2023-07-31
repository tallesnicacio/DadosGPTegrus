import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from io import BytesIO
import subprocess

def count_tokens(text):
    try:
        result = subprocess.run(['tiktoken', text], capture_output=True, text=True)
        tokens = int(result.stdout.strip().split()[0])
        return tokens
    except Exception as e:
        print(f"Erro ao contar os tokens: {e}")
        return None

def dataframe_para_string(df):
    df_string = df.to_string(index=False)
    return df_string


st.set_page_config(page_title='📊 DadosGPTegrus', layout="wide")
st.title('📊 DadosGPTegrus')

def load_xlsx(input_xlsx):
    # Read the file and shows at page
    df = pd.read_excel(input_xlsx)
    with st.expander('See DataFrame'):
        st.write(df)
    return df

def generate_response(xlsx_file, input_query, openai_api_key):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
    df = load_xlsx(xlsx_file)
    
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)

    max_tokens_per_query = 2000
    query_tokens = count_tokens(input_query)
    if query_tokens is not None and query_tokens > max_tokens_per_query:
        st.error(f"A pergunta excede o limite de {max_tokens_per_query} tokens. Por favor, reduza a pergunta.")
        return

    df_string = dataframe_para_string(df)

    max_tokens_total = 2000
    total_tokens = count_tokens(df_string)
    if total_tokens is not None and total_tokens > max_tokens_total:
        df = df.head(max_tokens_total // 10)
        df_string = dataframe_para_string(df)

    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)

    response = agent.run(input_query + " " + df_string)
    return st.success(response)

uploaded_file = st.file_uploader('Importar arquivo XLSX', type=['xlsx'], label_visibility="hidden")
query_text = st.text_input('Escreva a sua pergunta:', placeholder='Escreva a pergunta aqui ...', disabled=not uploaded_file)

openai_api_key = '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'

if openai_api_key.startswith('sk-') and uploaded_file is not None:
    st.header('Respostas')
    generate_response(uploaded_file, query_text, openai_api_key)
