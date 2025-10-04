# RAG-based ISL Video Retrieval Demo

This repository is a proof-of-concept demonstrating a Retrieval-Augmented Generation (RAG) system for translating spoken text into relevant Indian Sign Language (ISL) videos.

This demo simulates the core logic of the Speech-to-Sign video translation pipeline for a two-phone communication system for the Deaf and Hard-of-Hearing community. It is designed to run locally on a CPU.

---

## System Architecture

The demo simulates the following architecture:

```
[Input Text] -> [Sentence-BERT Model] -> [Vector Embedding] -> [FAISS Search] -> [Video KB] -> [Output Playlist]
```

---

## How It Works

1.  **Knowledge Base**: A set of sample ISL videos are stored in `knowledge_base/videos/`. A `metadata.json` file maps each video file to the phrase it represents.
2.  **Setup (`setup_database.py`)**: This script reads the metadata, converts each phrase into a vector embedding using Sentence-BERT, and stores these embeddings in a FAISS vector database (`video_index.faiss`).
3.  **Demo (`demo_script.py`)**:
    * Takes a user's input text (e.g., "how are you").
    * Converts this input text into a vector embedding.
    * Searches the FAISS database to find the most similar video(s) from the knowledge base.
    * Prints a "playlist" of the video files that should be played for the user.

---

## Setup Instructions

**1. Clone the Repository:**
```bash
git clone <your-repo-link>
cd ISL-RAG-Translator-Demo
```

**2. Create a Python Virtual Environment (Optional but Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Add Your Video Files:**
* Place your sample `.mp4` video files inside the `knowledge_base/videos/` directory.
* Update the `knowledge_base/metadata.json` file to accurately map your video filenames to the text phrases they represent.

**5. Build the Vector Database:**
* Run the setup script once. This will create the `video_index.faiss` file.
```bash
python setup_database.py
```

---

## Running the Demo

* Open `demo_script.py` and change the `query_text` variable to whatever you want to test.
* Run the script:
```bash
python demo_script.py
```
* The script will output the most relevant video file(s) from your knowledge base.