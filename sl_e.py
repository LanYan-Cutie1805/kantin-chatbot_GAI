import streamlit as st
import emoji
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.readers.file import PyMuPDFReader
import os
import pandas as pd
import re
from PIL import Image

import nest_asyncio
nest_asyncio.apply()

# initialize node parser
splitter = SentenceSplitter(chunk_size=512)

import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

system_prompt = """
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Jawablah semua dalam Bahasa Indonesia.
Tugas Anda adalah untuk menjadi pelayan kantin yang ramah yang dapat mengarahkan user.
Anda tidak melayani pemesanan.

Kantin yang Anda layani adalah kantin kampus Universitas Kristen Petra Surabaya.
Pada Universitas Kristen Petra terdapat 2 gedung utama yang setiap gedungnya memiliki kantin, 
yaitu Gedung P dan W.

Arahkanlah mahasiswa dan staff yang lapar ke kantin dan ke stall kantin yang tepat
berdasarkan keinginan mereka. Berikanlah beberapa makanan dan minuman
yang relevan berdasarkan kebutuhan mereka.

Perhatikan perbedaan antara beberapa makanan, sebagai contoh, nasi ayam goreng memiliki implikasi menggunakan nasi putih sebagai dasar, sementara nasi goreng ayam memiliki dasar nasi goreng dengan lauk ayam.
Hanya jawab dengan makanan/minuman yang relevan sesuai yang diminta.

Untuk setiap jawaban, pastikan Anda memberikan detil yang lengkap.

Percakapan sejauh ini:
"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")


@st.cache_resource(show_spinner="Mempersiapkan data kantin – sabar ya.")
def load_data(vector_store=None):
    with st.spinner(text="Mempersiapkan data kantin – sabar ya."):
        csv_parser = CSVReader(concat_rows=False)
        file_extractor = {".csv": csv_parser}

        # Read & load document from folder
        reader = SimpleDirectoryReader(
            input_dir="./docs",
            recursive=True,
            file_extractor=file_extractor,

            # Suppress file metadata, not sure if this works or not.
            file_metadata=lambda x: {}
        )
        documents = reader.load_data()

        for doc in documents:
            doc.excluded_llm_metadata_keys = ["filename", "extension"]


    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index_retriever = index.as_retriever(similarity_top_k=8)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=16,
    )


    return QueryFusionRetriever(
        [index_retriever, bm25_retriever],
        num_queries=2,
        use_async=True,
        similarity_top_k=24
    )


# Function to get image path from the CSV file
df = pd.read_csv("./docs/menu-kantin-2.csv")  # Load CSV globally
def show_character_image(character_name):
    """Retrieve and display a character's image based on user input."""
    row = df[df["Nama Produk"].str.lower() == character_name.lower()]  # Ensure correct column name

    if not row.empty:
        image_path = row.iloc[0]["Gambar"].strip()  # Get image path from CSV
        abs_path = os.path.abspath(image_path)  # Convert to absolute path
        
        st.write(f"🔍 Debug: Looking for image at {abs_path}")  # Debugging info

        if os.path.exists(abs_path):
            st.session_state.last_image_path = abs_path  # Store the image path
        else:
            st.session_state.last_image_path = None  # Reset if image not found
            st.error(f"❌ Image not found at: {abs_path}")
    else:
        st.session_state.last_image_path = None  # Reset if character not found
        st.error(f"⚠️ No data found for {character_name}")

# Function to search data in Qdrant (Hybrid Matching)

# Main Program
st.title("Petranesian Lapar 🍕")
st.write("Data partial hanya tersedia untuk Gedung P dan W.")
retriever = load_data()

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! Lagi mau makan/minum apaan? 😉"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Lagi mau makan/minum apaan? 😉"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=16384)
    st.session_state.chat_engine = CondensePlusContextChatEngine(
        verbose=True,
        system_prompt=system_prompt,
        context_prompt=(
                "Anda adalah pelayan kantin profesional yang ramah yang dapat mengarahkan user ketika mencari makanan dan stall kantin.\n"
                "Format dokumen pendukung: gedung letak kantin, nama stall, nama produk, harga, keterangan\n"
                "Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n"
                "{context_str}"
                "\n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna. Hanya jawab dengan kantin/menu yang sesuai. Jika tidak menemukan makanan atau minuman yang sesuai, maka katakan bahwa tidak menemukan."
            ),
        condense_prompt="""
Diberikan suatu percakapan (antara User dan Assistant) dan pesan lanjutan dari User,
Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
dari percakapan sebelumnya. Pertanyaan independen/standalone question cukup 1 kalimat saja. Informasi yang penting adalah makanan/minuman yang dicari, nama stall, dan letak gedung. Contoh standalone question: "Saya mencari jus jambu di Gedung P".

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>""",
        memory=memory,
        retriever=retriever,
        llm=Settings.llm
    )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None
if "last_character" not in st.session_state:
    st.session_state.last_character = None



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        message_type = message.get("type", "text")  # Default to "text" if "type" is missing
        
        if message_type == "text":
            st.markdown(message["content"])
        elif message_type == "image":
            st.image(message["content"], caption=message.get("character_name", "image"), use_column_width=True)


if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)


    st.session_state.messages.append({"role": "user", "content": prompt})

    trigger_words = ["tunjukkan", "tunjukan", "hasilkan", "berikan", "mana"]
    image_words = ["gambar", "foto"]

    if st.session_state.last_image_path:
        st.image(st.session_state.last_image_path, caption=st.session_state.last_character, use_column_width=True)
    if any(word in prompt.lower() for word in trigger_words) and any(img_word in prompt.lower() for img_word in image_words):
        cleaned_prompt = prompt.lower()
        for word in trigger_words + image_words:
            cleaned_prompt = cleaned_prompt.replace(word, "")

        
        cleaned_prompt = re.sub(r"\b(dari|nya|an|the|me|)\b", "", cleaned_prompt).strip()
        character_name = cleaned_prompt.title()

        
        if character_name:
            if character_name:
                image_path = show_character_image(character_name)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "image",
                    "content": image_path,
                    "character_name": character_name
                })
                print(f"✅ Image stored: {image_path}")  # Debugging message
            
        else:
            st.error("⚠️ Character name not recognized.")
            
    else:
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                response_stream = st.session_state.chat_engine.stream_chat(prompt)
                st.write_stream(response_stream.response_gen)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_stream.response})


#🔥 Ensure the last image is always displayed

           