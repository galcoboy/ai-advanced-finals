# ai-advanced-finals
# Finals Submission Checklist — Galco

**Email to:** pythonai200425+finals@gmail.com  
**Subject:** `Finals – Galco`

---

## Assignment 1 — Theory (24 pts)

**Folder:** `assignment1_theory/`

- [ ] `THEORY_ANSWERS.md` — convert to **PDF or DOCX** 
  - Open in Word / Google Docs → Export as PDF
  - Or: paste into Word and save as `.docx`

---

## Assignment 2 — Vector Database (20 pts)

**Folder:** `assignment2_vector_db/`

- [ ] `vector_db.py`
- [ ] **Screenshot** of terminal output showing:
  - Collection count (≥ 15 documents)
  - 5 queries with **distances**
  - Analysis section at the end

**Run:**
```powershell
cd finals-submission\assignment2_vector_db
python -m pip install -r requirements.txt
python vector_db.py
```

---

## Assignment 3 — RAG Word Document (20 pts)

**Folder:** `assignment3_rag_word/`

- [ ] `rag_word.py`
- [ ] `sample_document.docx`
- [ ] **Screenshot** showing questions → answers → retrieved context chunks

**Run:**
```powershell
cd finals-submission\assignment3_rag_word
python -m pip install -r requirements.txt
python create_sample_document.py
python rag_word.py
```

---

## Assignment 4 — Restaurant Agent + n8n (40 pts)

**Folder:** `finals-submission/assignment4_restaurant/`

- [ ] `restaurant_db.py`
- [ ] `restaurant_chatbot.py`
- [ ] `restaurant_chatbot_gradio.py`
- [ ] `requirements.txt`
- [ ] **Screenshot** of n8n workflow (Webhook → IF → Gmail branches)
- [ ] **Screenshot** of Gmail notifications or n8n Executions

**Run:**
```powershell
cd finals-submission\assignment4_restaurant
python restaurant_chatbot_gradio.py
```

Ensure n8n workflow is **Active** at `http://localhost:5678/webhook/restaurant`

---

## Quick test all assignments

```powershell
cd "C:\Users\galco\cursor project\finals-submission\assignment2_vector_db"
python vector_db.py

cd ..\assignment3_rag_word
python create_sample_document.py
python rag_word.py

cd ..\..\finals-submission\assignment4_restaurant
python smoke_test_restaurant_chatbot.py
```

---

## Total: 104 points

| # | Assignment | Points | Files |
|---|------------|--------|-------|
| 1 | Theory | 24 | PDF/DOCX |
| 2 | Vector DB | 20 | .py + screenshot |
| 3 | RAG Word | 20 | .py + .docx + screenshot |
| 4 | Restaurant + n8n | 40 | .py files + 2 screenshots |
