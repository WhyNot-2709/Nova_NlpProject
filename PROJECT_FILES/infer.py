import argparse
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------- TEXT UTILITIES -------- #

def dedupe_sentences(text):
    seen = set()
    cleaned = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        ss = s.strip()
        if ss and ss.lower() not in seen:
            cleaned.append(ss)
            seen.add(ss.lower())
    return " ".join(cleaned)


def trim_subjective(section, max_sent):
    sents = re.split(r"(?<=[.!?])\s+", section.strip())
    cleaned = []

    for s in sents:
        low = s.lower()

        if any(x in low for x in [
            "denies", "no ", "not experiencing", "without", "hasn't",
            "has not", "didn't", "did not", "no history of"
        ]):
            continue

        cleaned.append(s)

    return " ".join(cleaned[:max_sent]).strip()


# -------- CLINICAL FALLBACK BUCKETS -------- #

def keyword_assessment_plan(text):
    t = text.lower()

    # Dermatologic / contact exposure
    if any(x in t for x in ["itching", "itchy", "detergent", "rash", "red spots", "new soap"]):
        return (
            "Likely contact dermatitis associated with recent irritant exposure.",
            "Discontinue the suspected irritant such as the new detergent. Recommend gentle cleansing, moisturizers, "
            "and oral antihistamines for itching. Seek urgent care if breathing difficulty, facial swelling, spreading "
            "rash, or fever develops."
        )

    # Migraine
    if "migraine" in t or ("aura" in t and "headache" in t):
        return (
            "History suggests migraine with possible medication-overuse component.",
            "Reduce OTC analgesic frequency, track triggers, hydrate, maintain regular sleep and meals, and follow up "
            "in 4–6 weeks. Seek urgent care for sudden severe headache, fever, weakness, or vision changes."
        )

    # Musculoskeletal / inflammatory
    if any(x in t for x in ["joint pain", "knees", "wrists", "stiffness", "achy", "swelling"]) and \
       any(x in t for x in ["morning stiffness", "20–30 minutes", "worse in the morning", "improves with movement"]):
        return (
            "Polyarticular joint pain with morning stiffness, possibly post-viral or early inflammatory arthritis.",
            "Recommend NSAIDs as needed, gentle mobility exercises, hydration, and activity modification. Follow up "
            "if symptoms persist beyond 1–2 weeks, worsen, or new swelling, high fever, or functional decline occurs."
        )

    # Viral URI — MUST have fever + respiratory symptoms
    resp = any(x in t for x in ["cough", "runny nose", "sore throat", "congestion"])
    if resp and "fever" in t:
        return (
            "Likely uncomplicated viral upper respiratory infection.",
            "Supportive care with hydration, rest, and antipyretics. Avoid unnecessary antibiotics. Seek medical "
            "attention for worsening breathing, dehydration, prolonged fever, or severe symptoms."
        )

    # Chest pain / ACS caution — moved LAST
    if any(x in t for x in ["chest pain", "pressure", "left arm", "jaw pain"]) or \
       ("shortness of breath" in t and "exertion" in t):
        return (
            "Chest discomfort raises concern for possible acute coronary syndrome.",
            "Recommend immediate emergency evaluation and avoidance of physical exertion."
        )

    return None, None


# -------- SOAP STRUCTURE CLEANING -------- #

def enforce_soap_structure(text):
    t_lower = text.lower()

    # hard override: ear complaints should NEVER become ACS
    if any(x in t_lower for x in [
        "ear", "ear pain", "muffled", "popping", "ringing",
        "earbuds", "cotton bud", "hearing loss", "eustachian"
    ]):
        a_fb, p_fb = (
            "Ear discomfort likely related to eustachian tube dysfunction or local irritation.",
            "Recommend avoiding cotton buds and earbuds, using warm compresses, staying hydrated, and trialing "
            "NSAIDs for discomfort. Seek follow-up if worsening pain, fever, drainage, persistent hearing loss, or "
            "severe dizziness develops."
        )
    else:
        # get standard fallback
        a_fb, p_fb = keyword_assessment_plan(text)

    # remove accidental extra labels
    cleaned = re.sub(
        r"(?m)^(R:|M:|Plan:|Summary:|Discussion:|Notes:|Prognosis:).*$",
        "",
        text
    )

    cleaned = re.sub(
        r"(?ims)(write|generate|compose|summarize|create).*?(soap|note|clinical).*?$",
        "",
        cleaned
    )

    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()

    for sec in ["S:", "O:", "A:", "P:"]:
        if sec not in cleaned:
            cleaned += f"\n{sec}\n"

    parts = re.split(r"(?m)^(S:|O:|A:|P:)", cleaned)
    soap = {"S": "", "O": "", "A": "", "P": ""}

    for i in range(1, len(parts), 2):
        soap[parts[i].replace(":", "")] = parts[i+1].strip()

    # Stop S from swallowing O
    soap["S"] = re.sub(r"(?im)\bo:\s*", "", soap["S"])
    soap["S"] = trim_subjective(soap["S"], 4)

    if not soap["O"] or len(soap["O"].split()) < 6:
        soap["O"] = (
            "No vitals, physical examination findings, laboratory data, or imaging were provided in the transcript."
        )

    # apply fallbacks if needed
    if (not soap["A"].strip() or not soap["P"].strip()) and (a_fb and p_fb):
        soap["A"], soap["P"] = a_fb, p_fb

    if not soap["A"].strip():
        soap["A"] = "Clinical assessment was not explicitly provided in the dialogue."

    if not soap["P"].strip():
        soap["P"] = (
            "Advise follow-up with a healthcare provider. Monitor symptoms and return sooner if new, worsening, "
            "or concerning features develop."
        )

    return (
        f"S: {soap['S']}\n\n"
        f"O: {soap['O']}\n\n"
        f"A: {soap['A']}\n\n"
        f"P: {soap['P']}"
    ).strip()


def clean_soap_text(text):
    if not isinstance(text, str):
        text = str(text or "")
    text = dedupe_sentences(text)
    return enforce_soap_structure(text)


def chunk_text(text, tokenizer, max_len=900):
    tokens = tokenizer.encode(text)
    return [
        tokenizer.decode(tokens[i:i+max_len])
        for i in range(0, len(tokens), max_len)
    ]


# -------- MODEL GENERATION -------- #

def generate_one(
    text,
    model,
    tokenizer,
    max_input=1024,
    max_out=350,
    device="cuda"
):
    # reject garbage input early
    if not text or len(text.strip().split()) < 4:
        return (
            "S: Insufficient dialogue provided to summarize.\n\n"
            "O: No objective data available.\n\n"
            "A: Unable to determine clinical assessment due to lack of information.\n\n"
            "P: Provide additional dialogue for SOAP note generation."
        )

    chunks = chunk_text(text, tokenizer, max_input - 100)
    stitched = []

    for c in chunks:
        inputs = tokenizer(
            c, return_tensors="pt", truncation=True,
            max_length=max_input, padding=True
        ).to(device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_out,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=False,
            length_penalty=1.0,
        )
        stitched.append(tokenizer.decode(out[0], skip_special_tokens=True))

    stitched = " ".join(stitched).strip()

    prompt = (
        f"{stitched}\n\n"
        "Write a clinically accurate SOAP note based ONLY on the dialogue. "
        "Do not reference instructions.\n"
        "S:\nO:\nA:\nP:"
    )

    final_inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_input, padding=True
    ).to(device)

    out = model.generate(
        **final_inputs,
        max_new_tokens=500,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.25,
        early_stopping=False,
        min_length=350,
        length_penalty=1.0,
    )

    return clean_soap_text(tokenizer.decode(out[0], skip_special_tokens=True))


# -------- CLI -------- #

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(
        "cuda" if args.use_cuda else "cpu"
    )

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        preds = [
            generate_one(
                l, model, tokenizer,
                max_input=args.max_input,
                max_out=args.max_out,
                device="cuda" if args.use_cuda else "cpu"
            ) for l in lines
        ]
        out_file = args.output_file or "preds.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(p + "\n")
        print("Wrote predictions to", out_file)

    else:
        while True:
            text = input("\nPaste dialogue (or 'quit'):\n").strip()
            if text.lower() in ["quit", "q", "exit"]:
                break
            if not text:
                print("\nNo dialogue provided.\n")
                continue
            print("\n=== SOAP output ===\n")
            print(
                generate_one(
                    text, model, tokenizer,
                    max_input=args.max_input,
                    max_out=args.max_out,
                    device="cuda" if args.use_cuda else "cpu"
                )
            )
            print("\n===================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./soap_model_v1")
    parser.add_argument("--input_file", default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_input", type=int, default=2048)
    parser.add_argument("--max_out", type=int, default=450)
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    main(args)
