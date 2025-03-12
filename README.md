# AI-RAG-CHATBOT
Document Genie – AI-Powered Document Insights Chatbot

📝 Overview

Document Genie is an AI-powered chatbot that extracts insights from uploaded documents using the Retrieval-Augmented Generation (RAG) framework. It leverages Google's Gemini-Pro model and vector databases (FAISS/Pinecone) to provide accurate and contextually relevant answers based on document content.

🚀 Features

📂 Multi-Format Support: Upload and analyze PDF, PPTX, Excel, Word, HTML, CSV files.

🔍 AI-Powered Search: Uses FAISS/Pinecone for fast and efficient document retrieval.

🤖 LLM-Based Responses: Integrates Google Gemini-Pro to generate intelligent answers.

⚡ User-Friendly Interface: Built with Streamlit for an interactive and responsive UI.




🛠 Tech Stack

Programming Language: Python

Frameworks: Streamlit

AI Models: Google Gemini-Pro, OpenAI APIs

Vector Databases: FAISS, Pinecone

Libraries: 
streamlit : For building the web interface.
google-generativeai : Official Google Generative AI library.
langchain : For text splitting, embeddings, and conversational chains.
PyPDF2 : For extracting text from PDF files.
chromadb:a vector database for storing embeddings
faiss-cpu : For vector storage and similarity search.
langchain_google_genai : For integrating Google genarative ai models (gemini)
os :for environment variables or file handling
pandas :for reading Excel and CSV files.
beautifulsoup4: For parsing HTML files.
python-docx: For extracting text from Word documents.
python-pptx: For extracting text from PowerPoint files.



🎯 How It Works

1. Upload a Document: Select a PDF, PPTX, Excel, Word, HTML, or CSV file.


2. Ask a Question: Enter any question related to the document.


3. Get Instant Answers: The chatbot retrieves and generates responses based on document content.




🏆 Future Enhancements

🔄 Add support for real-time document updates.

🌍 Expand AI model options (GPT-4, Claude, etc.).

📊 Improve visualization of extracted data.


