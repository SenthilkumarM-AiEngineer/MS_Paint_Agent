# 🎨 Intelligent MS Paint Agent

An **AI-powered automation assistant** that controls **Microsoft Paint** using natural language commands.  
Built with **Streamlit**, **PyAutoGUI**, and **Transformer-based NLP models**, it interprets your text — like  
> *“Draw a big house at top left”*  
> *“Delete a small sun at top right”*  
> *“Redo”*  
and automatically performs the drawing inside MS Paint.

---

## 🧠 Key Features

- 🗣️ **Natural Language Understanding**  
  Uses custom fine-tuned NLP models:
  - **Action Model** → classifies intent (`draw`, `delete`, `redo`, etc.)
  - **NER Model** → extracts entities (shape, position, size)

- 🎨 **MS Paint Automation**  
  Automates drawing shapes (`house`, `tree`, `flower`, `boat`, `sun`) using **PyAutoGUI**, with:
  - **Position (anchors):** `center`, `left`, `right`, `top`, `bottom`, `top_left`, `top_right`, `bottom_left`, `bottom_right`  
    *(Synonyms:* `up → top`, `down → bottom`, `centre/middle → center`*.)*
  - **Size:** `small`, `medium`, `big` *(also accepts `large` → treated as `big`)*
  - Smart scaling, positioning, and window management.

- 💾 **Persistent Chat Sessions**  
  Each conversation creates a unique session (e.g., `chat-1`),  
  storing both your conversation and drawing state (`chat_history.csv`, `shapes_state.csv`).

- ♻️ **Redo / Delete Support**  
  Delete shapes by position, size, or type — even after restarting Paint.  
  Redo deleted shapes if the Paint process is still active.

- 💬 **Streamlit Chat UI**  
  Interactive chat interface that logs history and supports multiple chat sessions.

---


## ⚙️ Quick Setup (Windows 11, Python 3.10.9)

> ⚠️ **Note:** This application works **properly only on Windows 11 MS Paint**.  
> Earlier Paint versions (e.g., Windows 10) may not respond reliably to automation commands.

Follow these exact steps to set up and run the app:


# 1️⃣ Create and activate virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

# 2️⃣ Install dependencies
```powershell
pip install -r requirements.txt
```
# 3️⃣ Download pre-trained models (Action + NER)
```powershell
gdown --id 1W1K18dl4vg_8SYTXY6DfQwGzXGZj26Dt -O models.zip
```


# 4️⃣ Extract models.zip into the project folder
```powershell
Expand-Archive -Path models.zip -DestinationPath . -Force
```
# 5️⃣ Delete the ZIP after extraction
```powershell
Remove-Item models.zip
```
# 6️⃣ Run the app
```powershell
streamlit run app.py
```

---


## Project Structure
```
📂 Project Structure
MS_Paint_Agent/
├── app.py                       # Streamlit main app (entry point)
├── control.py                   # MS Paint automation logic (PyAutoGUI)
├── action.py                    # Intent classification model (BERT)
├── ner.py                       # Named Entity Recognition model
│
├── data/
│   ├── history/
│   │   ├── chat_history.csv     # Automatically created for chat logs
│   │   └── shapes_state.csv     # Automatically created for shape records
│   ├── saved_drawings/          # Contains saved MS Paint drawings (.png)
│   └── training/
│       ├── action.csv           # Training data for Action model
│       └── ner.xlsx             # Training data for NER model
│
├── models/
│   ├── classification/          # Pre-trained Action model (downloaded)
│   └── ner/                     # Pre-trained NER model (downloaded)
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignored files and folders
```