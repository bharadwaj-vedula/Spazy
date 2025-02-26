import io
import os
import base64
from llama_index.core.instrumentation.event_handlers import null
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS

from faster_whisper import WhisperModel

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings,StorageContext,load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer


@st.cache_resource
def load_tts_model():
  print('Initialzing Model')
  model_size = "tiny.en"
  model = WhisperModel(model_size,compute_type="int8", num_workers=10)
  return model



def stt(audio,model):
    segments, info = model.transcribe(audio, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    print(transcription)
    return transcription

class config:
  data_path = "./FAQs"
  persist_dir = "./VectorDB"

@st.cache_resource
def setup_rag():

  # Rag Config
  Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
  llm = Ollama(model="gemma2:2b")

  #Vector Store Creation
  print('Creating Vector Store')
  if not os.path.exists(config.persist_dir):
    documents = SimpleDirectoryReader(config.data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(
        persist_dir= config.persist_dir
    )
  else:
    storage_context = StorageContext.from_defaults(persist_dir=config.persist_dir)
    index = load_index_from_storage(storage_context)

  #Query Engine
  print('Creating Chat engine')
  memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

  chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        "Your name is Spazy and you are a help desk support specialist for comany named Swiggy"
        "Answer the following question looking at FAQ listed provided"
        "Do not assume and generate your own answers. If you are unable to find something in FAQs reply consumers by saying we will get back to you shortly on this"
        "Start Your conversations by stating your name, greeting and asking how may I assist you ?"
    ),
    llm = llm
  )

  return chat_engine

if __name__ == "__main__":
  model = load_tts_model()
  chat_engine = setup_rag()

  if "messages" not in st.session_state:
      st.session_state.messages = [
              {"role": "assistant", "content": "Hi! How may I assist you today?"}
          ]

  with st.sidebar:
      audio_bytes = mic_recorder(start_prompt="Start Recording. ⏺️",
                                  stop_prompt="Stop Recording. ⏹️", key='recorder')


      if audio_bytes:
          #transcription
          encode_string = base64.b64encode(audio_bytes['bytes'])
          wav_file = open("./hello.mp3", "wb")
          decode_string = base64.b64decode(encode_string)
          wav_file.write(decode_string)

          transcribe = stt("./hello.mp3",model)
          # Add assistant response to chat history
          st.session_state.messages.append({"role": "user", "content":transcribe})
          os.remove("./hello.mp3")
          # RAG
          response = chat_engine.chat(transcribe)
          print(response)
          if response:
            with st.chat_message("assistant"):
              st.session_state.messages.append({"role": "assistant", "content": response})

          tts = gTTS(f'{response}')
          tts.save('./output.mp3')
          st.audio('./output.mp3',autoplay= True)



  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"])
