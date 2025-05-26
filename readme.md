# YouTube Video RAG Chatbot

This project retrieves transcripts from YouTube videos and uses Retrieval-Augmented Generation (RAG) with vector search and a generative AI model to answer questions or summarize content.

## Features

- Fetches YouTube video transcripts using `youtube-transcript-api`
- Splits transcripts into chunks for embedding
- Embeds chunks using HuggingFace models
- Stores and searches embeddings with FAISS vector store
- Uses Google Gemini (via LangChain) for generative answers
- Retrieval-augmented prompt ensures answers are grounded in the transcript

## Requirements

- Python 3.8+
- `youtube-transcript-api`
- `langchain`
- `langchain_huggingface`
- `langchain_community`
- `langchain_google_genai`
- `faiss-cpu`
- `python-dotenv`

Install dependencies:

```sh
pip install -r requirements.txt
```

## Usage

1. Set up your environment variables in a `.env` file (for API keys, etc.).
2. Edit `video_id` in [RAG.py](RAG.py) to the desired YouTube video.
3. Run the script:

```sh
python RAG.py
```

The script will print a summary or answer based on the transcript.

## Notes

- If transcripts are disabled for a video, the script will notify you.
- You can modify the prompt or question as needed.

## License

MIT