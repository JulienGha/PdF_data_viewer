import os
from flask import Flask, render_template
from extract_msg import Message
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

MAIL_DIR = "/home/administrator/mail_infra"
STATIC_IMG_DIR = "static/images"
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

clf      = joblib.load("email_classifier.joblib")

app = Flask(__name__)

_cached_stats = {}

def load_and_classify(folder_path):
    records = []
    for fn in os.listdir(folder_path):
        if not fn.lower().endswith(".msg"):
            continue
        path = os.path.join(folder_path, fn)
        try:
            msg     = Message(path)
            subject = (msg.subject or "").strip()
            if subject.lower().startswith("re:"):
                continue
            body   = msg.body or ""
            text   = subject + "\n\n" + body
            emb    = embedder.encode([text])
            label  = clf.predict(emb)[0]
            author = msg.sender or "Unknown"
            records.append({"Category": label, "Author": author})
        except Exception as e:
            print(f"Error on {fn}: {e}")
    return pd.DataFrame(records)

def generate_stats():
    df = load_and_classify(MAIL_DIR)
    total = len(df)

    cat_counts = df["Category"].value_counts()
    plt.figure(figsize=(10, 6), dpi=100)
    cat_counts.plot(kind="bar")
    plt.title(f"Emails par catégorie ({total} totaux)")
    plt.xlabel("Catégorie")
    plt.ylabel("Nombre")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cat_path = os.path.join(STATIC_IMG_DIR, "emails_per_category.png")
    plt.savefig(cat_path)
    plt.close()

    auth_counts = df["Author"].value_counts().head(20)
    top_authors = auth_counts.index.tolist()

    main_cat = (
        df[df["Author"].isin(top_authors)]
        .groupby(["Author", "Category"])
        .size()
        .reset_index(name="Count")
        .sort_values(["Author", "Count"], ascending=[True, False])
        .drop_duplicates("Author")
        .set_index("Author")["Category"]
        .to_dict()
    )

    unique_cats = sorted(df["Category"].unique())
    cmap = plt.get_cmap("tab20", len(unique_cats))
    color_map = {cat: cmap(i) for i, cat in enumerate(unique_cats)}
    colors = [color_map[main_cat.get(a, unique_cats[0])] for a in top_authors]

    plt.figure(figsize=(15, 9), dpi=100)
    plt.bar(top_authors, auth_counts.values, color=colors)
    plt.title("Top 20 des auteurs (leur catégorie principale)")
    plt.xlabel("Auteur")
    plt.ylabel("Nombre")
    plt.xticks(rotation=45, ha="right")

    handles = [
        mpatches.Patch(color=color_map[cat], label=cat)
        for cat in unique_cats if cat in main_cat.values()
    ]
    plt.legend(handles=handles, title="Catégorie principale",
               bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    auth_path = os.path.join(STATIC_IMG_DIR, "emails_per_author.png")
    plt.savefig(auth_path)
    plt.close()

    pies = []
    for author in top_authors[:5]:
        sub = df[df["Author"] == author]["Category"].value_counts()
        plt.figure(figsize=(4,4), dpi=100)
        sub.plot(kind="pie", autopct="%1.1f%%", startangle=90)
        plt.title(author, fontsize=10)
        plt.ylabel("")
        plt.tight_layout()
        fname = f"pie_{author.replace(' ', '_')}.png"
        path  = os.path.join(STATIC_IMG_DIR, fname)
        plt.savefig(path)
        plt.close()
        pies.append({"author": author, "img": fname})

    return {
        "total": total,
        "cat_img":  os.path.basename(cat_path),
        "auth_img": os.path.basename(auth_path),
        "pies":     pies
    }

@app.before_first_request
def init_stats():
    global _cached_stats
    _cached_stats = generate_stats()

@app.route("/")
def index():
    return render_template("stats.html", **_cached_stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
