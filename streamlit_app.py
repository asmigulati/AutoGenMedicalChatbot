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
# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def add_to_chat(user_type, message):
    st.session_state['chat_history'].append((user_type, message))

def display_chat():
    for user_type, message in st.session_state['chat_history']:
        if user_type == "user":
            st.text_area("", value=message, key=message, height=50, disabled=True, style={"text-align": "right"})
        else:
            st.text_area("", value=message, key=message, height=50, disabled=True)


# Streamlit app layout
def main():
    st.title("Health Chatbot")
    display_chat()

    # User input
    with st.form("user_input_form", clear_on_submit=True):
        user_input = st.text_input("How are you feeling today?", key="user_input")
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        add_to_chat("user", user_input)
        
        # Process the input and get responses
        ans, user_input = assess_symp(user_input)
        add_to_chat("bot", ans)

        if ans == "No":
            tokens = symptoms(user_input)
            remedies = give_remedy(tokens)
            add_to_chat("bot", remedies)
        elif ans == "Yes":
            tokens = symptoms(user_input)
            jun_doc_mode(tokens, user_input)
            add_to_chat("bot", "Advice while waiting for the doctor:")
            remedies = give_remedy(tokens)
            add_to_chat("bot", remedies)
        else:
            add_to_chat("bot", ans)

        display_chat()

# Define your functions here: assess_symp, symptoms, home_remedies, give_remedy, jun_doc_mode, etc.
def assess_symp(symptom):
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  temperature= 0.2,
  messages=[
    {"role": "system", "content": "You are a medical advisor and you have to asses wehther the user needs clinical expertise or not. If the symptoms are not severe and can be treated at home, say They don't need medical expertise. Reply in one word answer when content related to health! Yes or No if the user input says I'm bored or something unrelated to health, say something along the lines of: I'm sorry I can't help you with that, if you have any health related concerns please let me know."},
    {"role": "user", "content": symptom}
  ]
)
  return completion.choices[0].message['content'], symptom
def symptoms(symp):
  completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  temperature= 0.2,
  messages=[
  {"role": "system", "content": "From the given input extract only the relevant medical symptoms and list them out separated with commas with no additional words "},
  {"role": "user", "content": symp}
  ]
  )
  return completion.choices[0].message['content']
def home_remedies(tokens):
  api_key = "AIzaSyDMcPs5y2Q58i8vp4SpjWmHxp35WvRrJfw"
  cse_id = "c1b34026eca3d42d6"
  texts={}

  def google_search(search_term, api_key, cse_id, **kwargs):
      service = build("customsearch", "v1", developerKey=api_key)
      res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
      return res['items']

  def extract_text_from_url(url):
      try:
          page = requests.get(url)
          soup = BeautifulSoup(page.content, "html.parser")
          text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
          return text
      except requests.exceptions.RequestException as e:
          print(f"Error fetching {url}: {e}")
          return None
  results = google_search(f"{tokens} relief", api_key, cse_id, num=5)

  for result in results:
      url = result['link']
      text = extract_text_from_url(url)
      if text:
          texts[url]=text
  return texts

def give_remedy(tokens):
  texts= str(home_remedies(tokens))
  for response in openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    temperature= 0.2,
    messages=[
    {"role": "system", "content": f"Act like a medical advisor and based on the given symptoms: {tokens} suggest what the patient can do to get better at home and how they can monitor their symptoms, do not state what might be the cause of it. provide the source from which you are extracting the remedy."},
    {"role": "user", "content": texts}
    ],
    stream=True
  ):
    try:
      print(response.choices[0].delta.content, end="")
    except:
      print("")
def jun_doc_mode(tokens, user_input):
  hist_dict={}
  autogen.ChatCompletion.start_logging(history_dict=hist_dict)
  junior_doc = autogen.AssistantAgent(
      name="junior_doc",
      llm_config = llm_config,
      is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE") or x.get("content", "").strip() == "",
      system_message =f" act like a medical assitant and ask appropriate, relevant follow up questions ONE AT A TIME to the human_user based on the symptoms {tokens} they mentioned, for example how long they have had it for, and other symptom they noticed, how severe it is and any other relevant question. you should employ a structured approach to gather the patient's clinical history, which might involve asking questions about symptoms, medical history, medications, allergies, and recent changes in health. take into consideration what has already been asked in the context that is provided to you and what info you've already gathered and then tread accordingly. Ask questions one by one, you will be given all the previous question you asked: {str(hist_dict)} once you are done asking questions, and have gathered enough information say THANK YOU and end the entire message with a TERMINATE",
  )

  human_user=autogen.UserProxyAgent(
      name = "human_user",
      human_input_mode = "ALWAYS",
      max_consecutive_auto_reply = 1,
      is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
      system_message = """
      Reply TERMINATE once the junior_doc says THANK YOU"""
  )
  terminator=autogen.UserProxyAgent("terminator",
                                    system_message="Your job is to terminate the chat if the doctor says THANK YOU or TERMINATE ",
                                    human_input_mode="NEVER",
                                    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE") or x.get("content", "").strip() == "",
                                    )

  grp_chat=autogen.GroupChat(agents=[junior_doc, human_user, terminator], messages=[], max_round=50)
  manager = autogen.GroupChatManager(groupchat=grp_chat, llm_config=llm_config, is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
      system_message = """
      Reply TERMINATE once the junior_doc says THANK YOU""")
  human_user.initiate_chat(
      manager,message=user_input,
  )



if __name__ == "__main__":
    main()
