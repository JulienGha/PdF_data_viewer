import pandas as pd
from collections import defaultdict
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

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
    Next Chemin ...
    """

    df = pd.read_excel(excel_path)

    # Ensure required columns exist
    required_cols = [
        "Chemin", "Groupe", "Accès", "Nom", "Utilisateurs",
        "NomG", "UtilisateursG",
        "Trigramme", "FullName"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in the Excel file.")

    # -----------------------------
    # 1) Build "nom -> set_of_users" for main usage
    # -----------------------------
    nom_to_users = defaultdict(set)
    for _, row in df.iterrows():
        nom = str(row["Nom"]).strip()
        utilisateurs_str = str(row["Utilisateurs"]).strip()
        users = [u.strip() for u in utilisateurs_str.split(",") if u.strip()]
        nom_to_users[nom].update(users)

    # -----------------------------
    # 2) Build "nomg -> set_of_subgroup_members"
    #    for multi-level subgroup expansions
    # -----------------------------
    nomg_to_usersg = defaultdict(set)
    for _, row in df.iterrows():
        nomg = str(row["NomG"]).strip()
        utilisateursg_str = str(row["UtilisateursG"]).strip()
        subgroup_users = [u.strip() for u in utilisateursg_str.split(",") if u.strip()]
        nomg_to_usersg[nomg].update(subgroup_users)

    # -----------------------------
    # 3) Build "trigram -> full_name"
    # -----------------------------
    trigram_to_full = {}
    for _, row in df.iterrows():
        trigram = str(row["Trigramme"]).strip()
        fullname = str(row["FullName"]).strip()
        if trigram:
            trigram_to_full[trigram] = fullname

    # -----------------------------
    # Recursive function for multi-level subgroup expansion
    # -----------------------------
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
            # Already visited this subgroup -> cycle -> skip
            return set()
        visited.add(user)

        if user not in subgroup_map:
            # Not a subgroup => final user
            visited.remove(user)
            return {user}
        else:
            # user is a subgroup -> expand each member
            final_users = set()
            for member in subgroup_map[user]:
                final_users |= fully_expand_subgroup(member, subgroup_map, visited)
            visited.remove(user)
            return final_users

    # -----------------------------
    # Helper to expand an entire set of base users
    # -----------------------------
    def expand_subgroups_multi_level(base_users, subgroup_map):
        expanded = set()
        for u in base_users:
            expanded |= fully_expand_subgroup(u, subgroup_map, set())
        return expanded

    # -----------------------------
    # Helper to replace any trigram with its full name
    # -----------------------------
    def replace_trigrams(user_set, trigram_map):
        replaced = set()
        for u in user_set:
            if u in trigram_map:
                replaced.add(trigram_map[u])
            else:
                replaced.add(u)
        return replaced

    # -----------------------------
    # 4) Build data[chemin][(groupe, acces)] = final expanded set
    # -----------------------------
    data = defaultdict(lambda: defaultdict(set))

    for _, row in df.iterrows():
        chemin = str(row["Chemin"]).strip()
        groupe = str(row["Groupe"]).strip()
        acces  = str(row["Accès"]).strip()

        # Base users from 'nom_to_users[groupe]'
        base_users = nom_to_users[groupe]

        # Expand subgroups (multi-level)
        multi_expanded = expand_subgroups_multi_level(base_users, nomg_to_usersg)

        # Replace any trigram with a full name
        final_users = replace_trigrams(multi_expanded, trigram_to_full)

        # Store in data structure
        data[chemin][(groupe, acces)].update(final_users)

    # -----------------------------
    # 5) Create the PDF (tab-indented)
    # -----------------------------
    c = canvas.Canvas(pdf_out, pagesize=A4)
    x_margin = 50
    y_position = 800
    line_height = 15

    # Font styles
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

    # -----------------------------
    # 6) Print hierarchy
    # -----------------------------
    for chemin in sorted(data.keys()):
        # Chemin line
        write_line(chemin, indent_level=0, style="chemin")

        # Each (groupe, acces)
        for (grp, acces), users in sorted(data[chemin].items()):
            group_line = f"Groupe: {grp} (Accès: {acces})"
            write_line(group_line, indent_level=1, style="group")

            # Each user in normal style
            for user in sorted(users):
                write_line(user, indent_level=2, style="user")

        # Horizontal line after each Chemin
        c.setLineWidth(0.5)
        c.setStrokeGray(0.7)
        y_line = y_position - 4
        c.line(x_margin, y_line, 550, y_line)
        y_position = y_line - line_height
        c.setStrokeGray(0)

    c.save()
    print(f"PDF with multi-level subgroup expansion & trigram replacement saved to: {pdf_out}")




if __name__ == "__main__":
    # Update path to your Excel file
    excel_file = "C:\\Users\\JGH\\Documents\\doc_test\\my_data.xlsx"
    pdf_file = "visualisation.pdf"
    build_tab_view_pdf(excel_file, pdf_file)