# 🚀 **Nova NLP**  
### *Medical Conversation Summarization & Ailment Suggestion — but make it futuristic.*

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7F00FF&height=120&section=header" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=2500&pause=800&color=7F00FF&center=true&vCenter=true&width=700&lines=Turning+raw+doctor–patient+dialogues+into+clear+clinical+insight.;Abstractive+summaries+with+low+hallu%2C+high+precision.;Built+for+medical+context+—+not+generic+NLP+fluff.">

</div>

---

## 📦 **Model Download (safetensors)**

GitHub threw a tantrum about the file size, so here’s your external link:

👉 **https://mega.nz/file/b4JCxbiQ#JbJrThhDake1JP1rjwhKA6ZgNo-Yzto9Kr3CvMBsBE0**

---

## 🧠 **What Nova NLP Does**

Nova NLP turns messy, informal medical conversations into clear, structured, clinically useful summaries — **actual abstraction**, not just copy-paste paraphrasing.

It also suggests **potential ailment categories** by reading cues, symptoms, and context with domain awareness.

---

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif" width="100%" alt="Animated medical waves">

</div>

---

## 🔥 **Why This Exists**

Medical documentation is:

- slow  
- inconsistent  
- mentally draining  
- prone to missed info  

Most summarizers hallucinate like they’re speedrunning fiction-writing.  
Nova NLP keeps things **clinically grounded**, **SOAP-structured**, and **low-hallucination** on purpose.

---

## 🎯 **Problem Statement**

The system must:

- Summarize medical conversations into structured, clinician-friendly text  
- Minimize hallucinations + fabricated claims  
- Suggest plausible ailment categories (not diagnoses)  
- Stay robust on informal, fragmented dialogue  

Challenges include domain terminology, coherence, hallucination control, and evaluation beyond surface metrics.

---

## 📚 **Dataset**

**Training / Validation:** Hugging Face medical dialogue datasets  
**Extra Testing:**  
– 30 curated conversations from Kaggle  
– HF medical sets  
– Synthetic ambiguous / rare-vocab samples  

Designed to aggressively stress-test grounding and safety.

---

## 🏗️ **Full Pipeline**

### **1. Data Ingestion**
- Load JSON/CSV medical dialogues  
- Standardize speaker tags  
- Preserve order & timestamps  

### **2. Preprocessing**
- Remove formatting noise  
- Normalize casing  
- Keep clinical terms consistent  

### **3. Model Training**
- Fine-tuned encoder–decoder transformer (T5/BART family)  
- Multi-stage validation splits  
- Mixed precision + CUDA acceleration  

### **4. Inference Pipeline**
Includes:  
- Keyword extraction  
- Ailment suggestion module  
- Safety filtering  
- Structured summary assembly  

### **5. Hallucination Control**
Regex-based cleanup for:  
- Fake drug names  
- Unreal dosages  
- Invented diagnoses  

Final validation ensures summary grounding.

### **6. Evaluation**
- ROUGE → too shallow  
- **BERTScore** → semantic + factual  
- Clinician-style scoring for reliability  

---

## 🧾 **Expected Outcomes**

- Clear, concise, clinically faithful summaries  
- Lower hallucination rates  
- Helpful ailment categorization  
- Reproducible research pipeline  

---

## 🩺 **Applications**

- Telemedicine automation  
- Clinical documentation workflows  
- EMR structuring  
- Patient communication aids  
- Safety-grounded medical chatbots  

---

## 🔧 **Tech Stack**

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD43B?style=for-the-badge&logo=huggingface&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Regex](https://img.shields.io/badge/Regex-7F00FF?style=for-the-badge)

</div>

---

<div align="center">

### ✨ *Healthcare meets AI — without hallucinating into the void.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=7F00FF&height=100&section=footer" width="100%"/>

</div>
