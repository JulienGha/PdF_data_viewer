import pandas as pd
from graphviz import Digraph


def build_tree_with_more_height(excel_path, output_name="my_tall_tree"):
    df = pd.read_excel(excel_path)

    dot = Digraph(comment="Chemin -> Nom -> Utilisateurs + Access depuis Chemin->Groupe")

    dot.attr(rankdir='TB')
    dot.attr(ranksep='10')
    dot.attr(nodesep='2')

    dot.attr(size="8,200!")

    dot.attr(ratio="expand")

    dot.node_attr.update(shape='box', style='rounded,filled', fillcolor='lightgray')
    dot.edge_attr.update(color='black', fontsize='10')

    created_nodes = set()
    created_edges = set()

    for _, row in df.iterrows():
        chemin = str(row["Chemin"])
        groupe = str(row["Groupe"])
        acces  = str(row["Accès"])
        nom    = str(row["Nom"])
        utilisateurs_str = str(row["Utilisateurs"])
        utilisateurs = [u.strip() for u in utilisateurs_str.split(",") if u.strip()]

        if chemin not in created_nodes:
            dot.node(chemin, label=chemin)
            created_nodes.add(chemin)

        if groupe not in created_nodes:
            dot.node(groupe, label=groupe)
            created_nodes.add(groupe)

        edge_chemin_groupe = (chemin, groupe, acces)
        if edge_chemin_groupe not in created_edges:
            dot.edge(chemin, groupe, label=acces)
            created_edges.add(edge_chemin_groupe)

        if nom not in created_nodes:
            dot.node(nom, label=nom)
            created_nodes.add(nom)

        edge_chemin_nom = (chemin, nom)
        if edge_chemin_nom not in created_edges:
            dot.edge(chemin, nom)
            created_edges.add(edge_chemin_nom)

        for user in utilisateurs:
            if user not in created_nodes:
                dot.node(user, label=user)
                created_nodes.add(user)

            edge_nom_user = (nom, user)
            if edge_nom_user not in created_edges:
                dot.edge(nom, user)
                created_edges.add(edge_nom_user)

    dot.render(output_name, view=False, format="pdf")
    print(f"Tall diagram saved as {output_name}.pdf")


if __name__ == "__main__":
    build_tree_with_more_height("C:\\Users\\JGH\\Documents\\doc_test\\my_data.xlsx", output_name="diagramme")
