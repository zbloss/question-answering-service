import streamlit as st
import requests
import os


def query_model(question: str) -> str:
    """
    Given a question this function
    will format and fire an HTTP POST
    to the Question Answering Model services,
    Returning the correct answer as a string.

    Arguments:
        question : str
            The question you want to ask the models.
    Returns:
        answer : str
            The answer to the `question`.
    """

    headers = {
        'accept': 'application/json',
        'content-type': 'application/x-www-form-urlencoded',
    }

    params = {
        'payload': question,
    }

    response = requests.post(QUERY_URL, params=params, headers=headers)

    if response.status_code == 200:
        answer = response.json()['answer']
        return answer
    else:
        return 'I am sorry but I am unable to find a relevant wine selection.'

def main():
    st.write(f'''

    # Welcome to the Question-Answering Service
            
    This is a sample project showing how to utilize:
        1. NLP Embedding Model
        2. NLP Question Answering Model
        3. Vector Database
        4. Streamlit UI
            
    to allow for a simple and repeatable Question-Answering system regardless of your domain.

    The data in this project has been predetermined to be a collection of Wine Reviews.
            
    Enter a question in the text box below to begin interacting with the technology stack!

    ''')

    with st.form('qa_form'):
        text = st.text_area('Enter question:', 'Which wine is both fruity and nutty?')
        submitted = st.form_submit_button('Submit')

        if submitted:
            answer = query_model(text)

            st.write(f'The answer to your question is: {answer}')

if __name__ == '__main__':

    QUERY_HOST = os.getenv('QUERY_HOST')
    QUERY_PORT = os.getenv('QUERY_PORT')
    QUERY_ENDPOINT = os.getenv('QUERY_ENDPOINT')

    QUERY_URL = f'{QUERY_HOST}:{QUERY_PORT}/{QUERY_ENDPOINT}'

    main()
