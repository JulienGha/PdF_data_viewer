import os

import pandas as pd
import numpy as np
from collections import defaultdict
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import re

def build_tab_view_pdf(excel_path, pdf_out="hierarchy_output.pdf"):
    """
    Reads Excel with columns:
      - Chemin, Groupe, Accès, Nom, Utilisateurs
      - NomG, UtilisateursG        (subgroup definitions)
      - Trigramme, FullName        (3-letter codes to full names)

    We do:
      1) nom_to_users[nom] = set_of_Utilisateurs
      2) nomg_to_usersg[nomg] = set_of_subgroup_members
      3) trigram_to_full[trigram] = full_name

    For each row's (Chemin, Groupe, Accès):
      - base_users = nom_to_users[groupe]
      - multi-level expand subgroups: if a user is in nomg_to_usersg, recursively expand
      - replace any 3-letter trigram with the user's full name
      - store result in data[chemin][(groupe, acces)]

    Finally, generate a tab-indented PDF with the structure:

    Chemin (bold)
        Groupe: <groupe> (Accès: <acces>) (bold)
            user_or_fullname (normal)
    ---------------------------------------
    """

    df = pd.read_excel(excel_path)

    required_cols = [
        "Chemin", "Groupe", "Accès", "Nom", "Utilisateurs",
        "NomG", "UtilisateursG",
        "Trigramme", "FullName"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in the Excel file.")

    nom_to_users = defaultdict(set)
    for _, row in df.iterrows():
        nom = str(row["Nom"]).strip()
        utilisateurs_str = str(row["Utilisateurs"]).strip()
        users = [u.strip() for u in utilisateurs_str.split(",") if u.strip()]
        nom_to_users[nom].update(users)

    nomg_to_usersg = defaultdict(set)
    for _, row in df.iterrows():
        nomg = str(row["NomG"]).strip()
        utilisateursg_str = str(row["UtilisateursG"]).strip()
        subgroup_users = [u.strip() for u in utilisateursg_str.split(",") if u.strip()]
        nomg_to_usersg[nomg].update(subgroup_users)

    trigram_to_full = {}
    for _, row in df.iterrows():
        trigram = str(row["Trigramme"]).strip()
        fullname = str(row["FullName"]).strip()
        if trigram:
            trigram_to_full[trigram] = fullname

    def fully_expand_subgroup(user, subgroup_map, visited=None):
        """
        Expands 'user' if it appears in subgroup_map (nomg_to_usersg).
        - If user is not in subgroup_map, it's a final user => return {user}.
        - If user is in subgroup_map, recursively expand each element of that subgroup.
        - 'visited' set is used to prevent infinite loops if there's a cycle.
        """
        if visited is None:
            visited = set()
        if user in visited:
            return set()
        visited.add(user)

        if user not in subgroup_map:
            visited.remove(user)
            return {user}
        else:
            final_users = set()
            for member in subgroup_map[user]:
                final_users |= fully_expand_subgroup(member, subgroup_map, visited)
            visited.remove(user)
            return final_users

    def expand_subgroups_multi_level(base_users, subgroup_map):
        expanded = set()
        for u in base_users:
            expanded |= fully_expand_subgroup(u, subgroup_map, set())
        return expanded

    def replace_trigrams(user_set, trigram_map):
        replaced = set()
        for u in user_set:
            if u in trigram_map:
                replaced.add(trigram_map[u])
            else:
                replaced.add(u)
        return replaced

    data = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        chemin = str(row["Chemin"]).strip()
        raw_grp = str(row["Groupe"]).strip()
        groupe = trigram_to_full.get(raw_grp, raw_grp)
        acces  = str(row["Accès"]).strip()

        if groupe not in nom_to_users:
            raise KeyError(f"Impossible de trouver le groupe '{groupe}' (depuis '{raw_grp}')")
        base_users = nom_to_users[groupe]

        multi_expanded = expand_subgroups_multi_level(base_users, nomg_to_usersg)

        final_users = replace_trigrams(multi_expanded, trigram_to_full)

        data[chemin][(groupe, acces)].update(final_users)

    c = canvas.Canvas(pdf_out, pagesize=A4)
    x_margin = 50
    y_position = 800
    line_height = 15

    STYLES = {
        "chemin": ("Helvetica-Bold", 10),
        "group":  ("Helvetica-Bold", 10),
        "user":   ("Helvetica",      10),
        "normal": ("Helvetica",      10),
    }

    def write_line(text, indent_level=0, style="normal"):
        nonlocal y_position
        font_name, font_size = STYLES[style]
        c.setFont(font_name, font_size)

        indent_offset = 20 * indent_level
        c.drawString(x_margin + indent_offset, y_position, text)
        y_position -= line_height

        if y_position < 50:
            c.showPage()
            y_position = 800

    for chemin in sorted(data.keys()):
        write_line(chemin, indent_level=0, style="chemin")

        for (grp, acces), users in sorted(data[chemin].items()):
            group_line = f"Groupe: {grp} (Accès: {acces})"
            write_line(group_line, indent_level=1, style="group")

            for user in sorted(users):
                write_line(user, indent_level=2, style="user")

        c.setLineWidth(0.5)
        c.setStrokeGray(0.7)
        y_line = y_position - 4
        c.line(x_margin, y_line, 550, y_line)
        y_position = y_line - line_height
        c.setStrokeGray(0)

    c.save()
    print(f"PDF with multi-level subgroup expansion & trigram replacement saved to: {pdf_out}")


def build_tab_view_pdf_without_trigram(excel_path, pdf_out="hierarchy_output.pdf"):
    # 1) Read & drop empty rows
    df = pd.read_excel(excel_path, dtype=str).dropna(how="all")

    # helper to clean & strip any column of whitespace/tabs/newlines
    def clean(col):
        return (
            df[col]
            .fillna("")
            .str.replace(r'[\t\r\n]+', ' ', regex=True)
            .str.replace(r' +', ' ', regex=True)
            .str.strip()
        )

    # clean all relevant columns
    for col in ["Chemin", "Groupe", "Accès", "GroupeG", "Utilisateur", "Trigramme", "FullName"]:
        if col in df.columns:
            df[col] = clean(col)

    # 2) Normalize Chemin → real NaN if blank or "nan"
    df.loc[df["Chemin"].str.lower() == "nan", "Chemin"] = ""
    df.loc[df["Chemin"] == "", "Chemin"] = np.nan

    # 3) Forward-fill GroupeG
    df["GroupeG"] = (
        df["GroupeG"]
        .replace(r'^\s*$', np.nan, regex=True)
        .ffill()
        .str.strip()
    )

    # 4) Build Trigram → Full Group Name map
    trigram_to_full = {}
    if {"Trigramme", "FullName"}.issubset(df.columns):
        for _, row in df[["Trigramme", "FullName"]].dropna(how="all").iterrows():
            code, full = row["Trigramme"].strip(), row["FullName"].strip()
            if code:
                trigram_to_full[code] = full

    # 5) Build GroupName → directMembers
    group_to_members = defaultdict(set)
    for _, row in df.iterrows():
        raw_grps = row["GroupeG"]
        raw_usrs = row["Utilisateur"]
        if not raw_grps or not raw_usrs:
            continue

        groups = [g.strip() for g in re.split(r"[,\t]+", raw_grps) if g.strip()]
        users = [u.strip() for u in re.split(r"[,\t]+", raw_usrs) if u.strip()]

        for grp in groups:
            # map any code in the group-list
            grp_full = trigram_to_full.get(grp, grp)
            for usr in users:
                # map any code in the user-list
                usr_full = trigram_to_full.get(usr, usr)
                group_to_members[grp_full].add(usr_full)

    # 6) Recursive expander for codes/groups → ultimate people
    def expand_deep(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited:
            return set()
        visited.add(name)

        # map code → group
        name = trigram_to_full.get(name, name)
        if name in group_to_members:
            out = set()
            for member in group_to_members[name]:
                out |= expand_deep(member, visited)
            return out
        return {name}

    data = defaultdict(lambda: defaultdict(set))
    subset = (
        df[["Chemin", "Groupe", "Accès"]]
        .dropna(subset=["Chemin", "Groupe"])
        .drop_duplicates()
    )
    for _, row in subset.iterrows():
        chemin = row["Chemin"]
        acces = row["Accès"]
        raw = row["Groupe"]

        codes = [c.strip() for c in re.split(r"[,\t]+", raw) if c.strip()]
        people = set()
        for code in codes:
            people |= expand_deep(code)

        for person in sorted(people):
            data[chemin][acces].add(person)

    c = canvas.Canvas(pdf_out, pagesize=A4)
    x0, y, lh = 50, 800, 15
    STYLES = {
        "chemin": ("Helvetica-Bold", 12),
        "group": ("Helvetica-Bold", 10),
        "user": ("Helvetica", 10),
    }

    def write_line(txt, indent, style):
        nonlocal y
        line = txt.strip()
        font, size = STYLES[style]
        c.setFont(font, size)
        c.drawString(x0 + 20 * indent, y, line)
        y -= lh
        if y < 50:
            c.showPage()
            y = 800

    for chemin in sorted(data):
        write_line(chemin, 0, "chemin")
        for acces, users in sorted(data[chemin].items()):
            write_line(f"Accès: {acces}", 1, "group")
            for u in users:
                write_line(u, 2, "user")
        c.setLineWidth(0.5)
        c.setStrokeGray(0.7)
        y -= 4
        c.line(x0, y, 550, y)
        y -= lh
        c.setStrokeGray(0)

    c.save()
    print(f"PDF saved to {pdf_out}")

if __name__ == "__main__":
    excel_file = "C:\\Users\\JGH\\Documents\\doc_test\\StructureActuel.xlsx"
    pdf_file = "visualisation_notrigram.pdf"
    build_tab_view_pdf_without_trigram(excel_file, pdf_file)