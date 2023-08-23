# Langchain Chatbot for Multiple PDFs

Langchain Chatbot is a conversational chatbot powered by OpenAI and Hugging Face models. It is designed to provide a seamless chat interface for querying information from multiple PDF documents. The chatbot utilizes the capabilities of language models and embeddings to perform conversational retrieval, enabling users to ask questions and receive relevant answers from the PDF content.

## Purpose

The purpose of this project is to create a chatbot that can interact with users and provide answers from a collection of PDF documents. The chatbot uses natural language processing and machine learning techniques to understand user queries and retrieve relevant information from the PDFs. By incorporating OpenAI and Hugging Face models, the chatbot leverages powerful language models and embeddings to enhance its conversational abilities and improve the accuracy of responses.

## Features

- Multiple PDF Support: The chatbot supports uploading multiple PDF documents, allowing users to query information from a diverse range of sources.
- Conversational Retrieval: The chatbot uses conversational retrieval techniques to provide relevant and context-aware responses to user queries.
- Language Models: The project incorporates OpenAI and Hugging Face models for natural language understanding and generation, enabling the chatbot to engage in meaningful conversations.
- PDF Text Extraction: The PDF documents are processed to extract the text content, which is used for indexing and retrieval.
- Text Chunking: The extracted text is split into smaller chunks to improve the efficiency of retrieval and provide more precise answers.

## Installation

To install and run the Langchain Chatbot, follow these steps:

Clone the repository 
```
git clone https://github.com/Abdullahw72/langchain-chatbot-multiple-PDF
```

Create a Virtual Environment
```bash
pip install virtualenv
python<version> -m venv <virtual-environment-name>
<virtual-environment-name>\Scripts\activate

```
Install the dependencies using requirements.txt

```bash
pip install -r requirements.txt
```

Add your OpenAI Key by creating a .env file in the folder and add the following within it:
```
OPENAI_API_KEY="<your key>"
```

For those of you who would like to use the HuggingFace Approach, be sure to add the HuggingFace API Key in your .env file:
```
HUGGINGFACEHUB_API_TOKEN="<your key">
```

Run the App
```bash
streamlit run app.py
```

Check out top embedding models: https://huggingface.co/blog/mteb

Check out top LLMs: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

**NOTE:** Please keep in mind that you need to check the hardware requirements for the model you choose based on your machine,
as the embeddings and the model will run locally on your system, and will be loaded in your RAM. Be sure to do some research before running the code with any choosen model.

If you would like an easy-to-setup, completely private, and minimum hardware-compatible chatbot, follow this repo: https://github.com/imartinez/privateGPT



## Usage

-  Upload PDF documents: Use the sidebar in the application to upload one or more PDF files.
-  Ask questions: In the main chat interface, enter your questions related to the content of the uploaded PDFs.
-  Receive answers: The chatbot will generate responses based on the information extracted from the PDFs.

## Sample Output

![Chat Screenhot 1](https://github.com/Abdullahw72/langchain-chatbot-multiple-PDF/blob/master/Chat_Result.png?raw=true)


![Chat Screenhot 1](https://github.com/Abdullahw72/langchain-chatbot-multiple-PDF/blob/master/Chat_Result-2.png?raw=true)


## Future Enhancements
- Integrate Vector Database for storing embeddings. (Reach out to me if you need to get this done).
- Support for additional document formats, such as Word documents or web pages.
- Integration of more advanced language models and embeddings.
- Improved error handling and user feedback.
- Enhanced user interface and customization options.

## Blog
Check out my detailed blog on this project here: https://medium.com/@abdullahw72/langchain-chatbot-for-multiple-pdfs-harnessing-gpt-and-free-huggingface-llm-alternatives-9a106c239975

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please submit a pull request or open an issue in the GitHub repository.

## License

This project is licensed under the MIT License.
