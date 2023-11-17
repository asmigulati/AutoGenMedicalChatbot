import streamlit as st
import openai
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import sys
import autogen
from autogen import UserProxyAgent, ConversableAgent, oai, config_list_from_json, AssistantAgent

# Load your API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit app layout
def main():
    st.title("Health Chatbot")

    # Collect user symptom input
    symptom = st.text_input("How are you feeling today?")
    
    if st.button("Submit"):
        # Process the symptom input
        ans, user_input = assess_symp(symptom)
        st.write(ans)

        if ans == "No":
            tokens = symptoms(user_input)
            remedies = give_remedy(tokens)
            st.write(remedies)
        elif ans == "Yes":
            tokens = symptoms(user_input)
            jun_doc_mode(tokens, user_input)
            st.write("Advice while waiting for the doctor:")
            remedies = give_remedy(tokens)
            st.write(remedies)
        else:
            st.write(ans)

# Define your functions here: assess_symp, symptoms, home_remedies, give_remedy, jun_doc_mode, etc.
def assess_symp():
  symptom= input("How are you feeling today?")

  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  temperature= 0.2,
  messages=[
    {"role": "system", "content": "You are a medical advisor and you have to asses wehther the user needs clinical expertise or not. If the symptoms are not severe and can be treated at home, say They don't need medical expertise. Reply in one word answer when content related to health! Yes or No if the user input says I'm bored or something unrelated to health, say something along the lines of: I'm sorry I can't help you with that, if you have any health related concerns please let me know."},
    {"role": "user", "content": symptom}
  ]
)
  return completion.choices[0].message['content'], symptom


if __name__ == "__main__":
    main()
