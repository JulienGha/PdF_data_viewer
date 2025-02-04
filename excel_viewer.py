import pandas as pd
from graphviz import Digraph


def build_tree_with_more_height(excel_path, output_name="my_tall_tree"):
    df = pd.read_excel(excel_path)

    dot = Digraph(comment="Chemin -> Nom -> Utilisateurs + Access depuis Chemin->Groupe")

    # -------------------------------------------------------------------------------------
    # 1. Increase vertical space between ranks (rows)
    #    Example: '1.5' or '2' or even higher if you have many nodes in each row
    # -------------------------------------------------------------------------------------
    dot.attr(rankdir='TB')      # top-to-bottom
    dot.attr(ranksep='10')       # more space between ranks
    dot.attr(nodesep='2')     # horizontal space between nodes (optional)

    # -------------------------------------------------------------------------------------
    # 2. Specify a larger page size in inches, and add '!' so Graphviz doesn't shrink it.
    #    For example:  width=8 inches, height=20 inches, forced. You can adjust as needed.
    # -------------------------------------------------------------------------------------
    # dot.attr(size="8,20!")  # 8 inches wide, 20 inches tall, no auto-shrink
    #
    # If you have a *lot* of nodes, you can increase the second number even more, e.g. 8,40!
    #
    # Alternatively, if you want a REALLY tall diagram:
    dot.attr(size="8,200!")  # for demonstration: 8 wide, 50 tall

    # Optional: let the diagram expand to fill the size
    dot.attr(ratio="expand")

    # -------------------------------------------------------------------------------------
    # 3. Add styling for clarity (optional)
    # -------------------------------------------------------------------------------------
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

        # ----- Chemin node
        if chemin not in created_nodes:
            dot.node(chemin, label=chemin)
            created_nodes.add(chemin)

        # ----- Groupe node
        if groupe not in created_nodes:
            dot.node(groupe, label=groupe)
            created_nodes.add(groupe)

        # Edge Chemin -> Groupe (label = Accès)
        edge_chemin_groupe = (chemin, groupe, acces)
        if edge_chemin_groupe not in created_edges:
            dot.edge(chemin, groupe, label=acces)
            created_edges.add(edge_chemin_groupe)

        # ----- Nom node
        if nom not in created_nodes:
            dot.node(nom, label=nom)
            created_nodes.add(nom)

        # Edge Chemin -> Nom
        edge_chemin_nom = (chemin, nom)
        if edge_chemin_nom not in created_edges:
            dot.edge(chemin, nom)
            created_edges.add(edge_chemin_nom)

        # ----- Users
        for user in utilisateurs:
            if user not in created_nodes:
                dot.node(user, label=user)
                created_nodes.add(user)

            # Edge Nom -> user
            edge_nom_user = (nom, user)
            if edge_nom_user not in created_edges:
                dot.edge(nom, user)
                created_edges.add(edge_nom_user)

    dot.render(output_name, view=False, format="pdf")
    print(f"Tall diagram saved as {output_name}.pdf")


# -----------------------------------------------------------
# Example usage:
#   1) Put this script in build_tree.py
#   2) Call the function with your Excel filename:
# -----------------------------------------------------------
if __name__ == "__main__":
    build_tree_with_more_height("C:\\Users\\JGH\\Documents\\doc_test\\my_data.xlsx", output_name="diagramme")
