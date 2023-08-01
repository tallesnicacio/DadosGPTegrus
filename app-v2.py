import streamlit as st
import pandas as pd
import json
import openai
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from io import BytesIO
import time

# Page title
st.set_page_config(page_title='📊 DadosGPTegrus', layout="wide")
st.title('📊 DadosGPTegrus')

# Load XLSX file
def load_xlsx(input_xlsx):
    # Read the file and shows at page
    df = pd.read_excel(input_xlsx)
    with st.expander('See DataFrame'):
        st.write(df)
    return df

# Generate GPT response
def generate_response(xlsx_file, input_query, openai_api_key):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
    df = load_xlsx(xlsx_file)
    # Create Pandas DataFrame
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    
    # Keep track of attempts
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = agent.run(input_query)
            return response
        except openai.error.RateLimitError:
            st.warning(f"Erro de taxa de limitação. Tentativa {attempt+1} de {max_attempts}. Aguarde alguns segundos antes de tentar novamente.")
            time.sleep(4)  # Wait 4 seconds before retrying
        except Exception as e:
            st.error(f"Erro ao gerar a resposta: {e}")
            break
    
    st.error(f"Não foi possível gerar a resposta após {max_attempts} tentativas.")
    return None

# Inputs 
uploaded_file = st.file_uploader('Importar arquivo XLSX', type=['xlsx'], label_visibility="hidden")
query_text = st.text_input('Escreva a sua pergunta:', placeholder='Escreva a pergunta aqui ...', disabled=not uploaded_file)

# OpenAI API key
openai_api_key = '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'

# App logic
if uploaded_file is not None and openai_api_key.startswith('sk-') and query_text:
    st.header('Respostas')
    response = generate_response(uploaded_file, query_text, openai_api_key)
    if response is not None:
        st.success(response)
