import os
import pandas as pd
from extract_msg import Message
from transformers import pipeline

candidate_labels = [
    "Accès informatique",
    "Téléphonie",
    "Messagerie",
    "Accès dossiers patients",
    "Accès à distance",
    "Accès physiques",
    "Ingénierie et projets IT",
    "Support matériel",
    "Impression",
    "Support applications bureautique",
    "Support administratif aux applications métiers",
    "Medulla",
    "Intranet",
    "Timbrage",
    "Accès télétravail",
    "Modification jours de travail"
]

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

def classify_msg(file_path):
    msg = Message(file_path)
    text = "\n".join(filter(None, [msg.subject, msg.body]))
    out = classifier(text, candidate_labels)
    print("classified")
    return out["labels"][0], out["scores"][0]

def batch_classify(folder_path):
    records = []
    for fn in os.listdir(folder_path):
        if not fn.lower().endswith(".msg"):
            continue
        path = os.path.join(folder_path, fn)

        try:
            msg = Message(path)
            subject = msg.subject or ""

            if subject.strip().lower().startswith("re:"):
                print(f"⏭️ Skipping reply: {fn}")
                continue

            body = msg.body or ""
            full_text = subject + "\n\n" + body

            label, score = classify_msg(path)
        except Exception as e:
            print(f"❌ Failed on {fn}: {e}")
            subject = ""
            body = ""
            full_text = ""
            label = None
            score = None

        records.append({
            "FileName": fn,
            "Subject": subject,
            "EmailContent": full_text,
            "PredictedLabel": label,
            "Confidence": score
        })

    return pd.DataFrame(records)



if __name__ == "__main__":
    folder = "/home/lestoises/mail_infra"
    df = batch_classify(folder)
    df.to_csv(
        "zero_shot_predictions.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )
    print("✅ Done — see zero_shot_predictions.csv")
