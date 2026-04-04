"""
Générateur de signalements synthétiques pour Municip'All
Produit 300 signalements réalistes pour le cold start Qdrant + PostgreSQL
"""

import json
import random
from datetime import datetime, timedelta

# --- TEMPLATES PAR CATÉGORIE ---
# Chaque template a des {slots} qui seront remplis aléatoirement

TEMPLATES = {
    "voirie": [
        "Nid de poule {taille} {localisation}, {consequence}",
        "Trottoir {defaut} devant le {numero} {rue}, {consequence}",
        "Le revêtement de la chaussée est {defaut} {localisation}",
        "Plaque d'égout {probleme_plaque} {localisation}, {consequence}",
        "Fissure {taille} sur la route {localisation}, {consequence}",
        "Bordure de trottoir {defaut} {localisation}",
        "Passage piéton effacé {localisation}, {consequence}",
        "Ralentisseur abîmé {localisation}, on voit plus la signalisation",
        "Route inondée {localisation} à chaque pluie, {consequence}",
    ],
    "proprete": [
        "Dépôt sauvage {type_dechet} {localisation}",
        "Poubelle {probleme_poubelle} {localisation}",
        "Tags et graffitis {localisation}, {ampleur}",
        "Déjections canines partout {localisation}, {consequence_proprete}",
        "Encombrants abandonnés {localisation} depuis {duree}",
        "Sac poubelle éventré {localisation}, {consequence_proprete}",
        "La rue est vraiment sale {localisation}, {consequence_proprete}",
        "Mégots partout {localisation}, {complement_proprete}",
    ],
    "eclairage": [
        "Lampadaire éteint {localisation} depuis {duree}, {consequence_eclairage}",
        "Éclairage public qui clignote {localisation}, {complement_eclairage}",
        "Plusieurs lampadaires en panne {localisation}",
        "Le lampadaire devant le {numero} {rue} est cassé, {consequence_eclairage}",
        "Pas d'éclairage dans le passage {localisation}, {consequence_eclairage}",
        "Ampoule grillée sur le lampadaire {localisation}",
        "Éclairage trop faible {localisation}, {consequence_eclairage}",
    ],
    "espaces_verts": [
        "Arbre {probleme_arbre} {localisation}, {consequence_arbre}",
        "Branche {taille} qui dépasse sur le trottoir {localisation}",
        "Pelouse du parc {parc} complètement {etat_pelouse}",
        "Haie non taillée {localisation}, {consequence_arbre}",
        "Arbre mort {localisation}, {consequence_arbre}",
        "Mauvaises herbes envahissent le trottoir {localisation}",
        "Un tronc d'arbre est tombé {localisation}, {consequence_arbre}",
    ],
    "mobilier_urbain": [
        "Banc cassé {localisation}, {consequence_mobilier}",
        "Poteau de signalisation {probleme_poteau} {localisation}",
        "Barrière {defaut} {localisation}, {consequence_mobilier}",
        "Panneau d'affichage {probleme_panneau} {localisation}",
        "Abribus {probleme_abribus} à l'arrêt {arret}",
        "Borne à vélos hors service {localisation}",
        "Fontaine à eau en panne {localisation} depuis {duree}",
    ],
    "bruit": [
        "Travaux très bruyants {localisation} {horaire_bruit}",
        "Tapage nocturne récurrent {localisation}, {complement_bruit}",
        "Chantier qui commence {horaire_bruit} {localisation}",
        "Nuisances sonores {localisation} à cause {source_bruit}",
        "Alarme qui sonne en continu {localisation} depuis {duree}",
    ],
    "stationnement": [
        "Voiture ventouse {localisation} depuis {duree}",
        "Véhicule garé sur le trottoir {localisation}, {consequence_stationnement}",
        "Stationnement gênant {localisation}, {consequence_stationnement}",
        "Épave de voiture abandonnée {localisation} depuis {duree}",
        "Camion garé en double file {localisation}, {consequence_stationnement}",
        "Véhicule qui bloque l'accès pompiers {localisation}",
    ],
    "eau_assainissement": [
        "Fuite d'eau {localisation}, {ampleur_fuite}",
        "Bouche d'égout bouchée {localisation}, {consequence_eau}",
        "Eau stagnante {localisation} depuis {duree}",
        "Odeur d'égout très forte {localisation}",
        "Canalisation qui fuit {localisation}, {ampleur_fuite}",
    ],
}

# --- SLOTS DE REMPLISSAGE ---

QUARTIERS = [
    "Centre-ville", "Val-d'Argent", "Orgemont", "Les Coteaux",
    "Val-Notre-Dame", "Joliot-Curie", "La Colonie", "Mazagran",
    "Val-Sud", "Les Champioux", "Le Marais", "Saint-Denis"
]

RUES = [
    "rue Paul Vaillant-Couturier", "avenue Gabriel Péri", "rue de Calais",
    "boulevard Héloïse", "rue des Jardins", "avenue Jean Jaurès",
    "rue Henri Barbusse", "rue de Sartrouville", "avenue du Général Leclerc",
    "rue Antonin Artaud", "boulevard du Général Delambre", "rue Victor Hugo",
    "rue Pierre Semard", "avenue de Stalingrad", "rue Marcel Cachin",
    "rue Gustave Eiffel", "rue Raspail", "avenue de la République",
    "rue Émile Zola", "rue du 8 Mai 1945", "rue des Peupliers",
    "allée des Tilleuls", "impasse des Lilas", "rue de la Gare",
]

PARCS = [
    "parc des Berges", "parc du Moulin", "square Jean Macé",
    "parc des Impressionnistes", "jardin des Coteaux", "square Héloïse",
]

ARRETS = [
    "Jean Jaurès", "Mairie d'Argenteuil", "Les Coteaux", "Gabriel Péri",
    "Val-d'Argent", "Gare d'Argenteuil", "Joliot-Curie",
]

SLOTS = {
    "taille": ["assez gros", "énorme", "de taille moyenne", "dangereux", "profond"],
    "localisation": [
        "au niveau du {numero} {rue}",
        "à l'angle de la {rue} et du boulevard",
        "devant l'école {rue}",
        "près de l'arrêt de bus {rue}",
        "dans le quartier {quartier}",
        "{rue}, côté {quartier}",
        "à côté du {parc}",
        "en face de la pharmacie {rue}",
        "juste après le carrefour {rue}",
    ],
    "numero": [str(n) for n in range(1, 120)],
    "consequence": [
        "c'est dangereux pour les vélos",
        "les voitures font des écarts",
        "impossible de passer en fauteuil roulant",
        "risque de chute pour les piétons",
        "ça gêne la circulation",
        "les poussettes ne passent plus",
        "c'est un vrai danger",
        "mon voisin s'est déjà fait mal",
    ],
    "defaut": [
        "complètement défoncé", "fissuré", "affaissé", "en très mauvais état",
        "cassé", "abîmé depuis longtemps", "soulevé par les racines",
    ],
    "probleme_plaque": [
        "manquante", "qui bouge et fait du bruit", "enfoncée",
        "cassée", "mal fixée", "descellée",
    ],
    "type_dechet": [
        "de gravats et de vieux meubles", "d'encombrants", "de sacs poubelles",
        "de cartons et emballages", "de matelas et sommiers",
        "de déchets de chantier", "de pneus et ferraille",
    ],
    "probleme_poubelle": [
        "qui déborde depuis 3 jours", "cassée", "jamais vidée",
        "renversée par le vent", "pleine à craquer", "sans couvercle",
    ],
    "ampleur": [
        "ça couvre tout le mur", "c'est sur plusieurs mètres",
        "ils sont vraiment partout", "c'est devenu invivable visuellement",
    ],
    "consequence_proprete": [
        "c'est vraiment dégueulasse", "ça attire les rats",
        "les enfants jouent à côté", "ça pue surtout en été",
        "c'est pas digne du quartier", "ça fait fuir les passants",
    ],
    "complement_proprete": [
        "il n'y a aucun cendrier à proximité",
        "c'est devant une école en plus",
        "ça donne une image déplorable",
    ],
    "duree": [
        "plus d'une semaine", "plusieurs jours", "au moins 3 semaines",
        "presque un mois", "quelques jours", "2 semaines",
        "depuis le mois dernier", "une bonne dizaine de jours",
    ],
    "consequence_eclairage": [
        "c'est dangereux le soir", "on n'y voit rien la nuit",
        "ma fille a peur de rentrer seule", "il y a eu des incidents",
        "ça craint pour les personnes âgées",
        "le passage est vraiment sombre",
    ],
    "complement_eclairage": [
        "c'est très gênant", "ça fait plusieurs semaines",
        "j'ai déjà signalé mais rien n'a changé",
    ],
    "probleme_arbre": [
        "dangereux qui penche", "avec des branches mortes",
        "qui soulève le trottoir avec ses racines",
        "malade qui perd son écorce",
    ],
    "consequence_arbre": [
        "ça bloque le passage", "c'est dangereux en cas de vent",
        "les branches touchent les câbles", "les feuilles bouchent les gouttières",
        "impossible de passer avec une poussette",
    ],
    "etat_pelouse": [
        "abîmée", "sèche et jaune", "pleine de trous",
        "envahie de mauvaises herbes", "défoncée par le passage",
    ],
    "consequence_mobilier": [
        "c'est dangereux pour les enfants",
        "quelqu'un va se blesser",
        "il manque des parties",
        "c'est inutilisable",
    ],
    "probleme_poteau": [
        "tordu", "arraché", "penché dangereusement", "tagué et illisible",
    ],
    "probleme_panneau": [
        "arraché", "illisible", "tourné dans le mauvais sens", "cassé",
    ],
    "probleme_abribus": [
        "avec la vitre cassée", "tagué", "sans banc", "dont le toit fuit",
    ],
    "horaire_bruit": [
        "dès 6h du matin", "tous les soirs après 22h",
        "même le dimanche matin", "à des heures pas possibles",
    ],
    "complement_bruit": [
        "ça dure depuis des semaines", "impossible de dormir",
        "les voisins en ont tous marre", "j'ai des enfants en bas âge",
    ],
    "source_bruit": [
        "d'un bar qui ferme tard", "de motos qui font des rodéos",
        "d'un chantier non déclaré", "de musique à fond",
    ],
    "consequence_stationnement": [
        "les piétons doivent marcher sur la route",
        "impossible de sortir de mon garage",
        "ça bloque la visibilité au carrefour",
        "les pompiers ne pourraient pas passer",
    ],
    "ampleur_fuite": [
        "ça coule en continu", "il y a une flaque permanente",
        "c'est un vrai gaspillage", "la chaussée est inondée",
    ],
    "consequence_eau": [
        "ça déborde quand il pleut", "odeur insupportable",
        "l'eau stagne sur le trottoir", "risque d'inondation",
    ],
}

URGENCE_WEIGHTS = {
    "voirie": {"haute": 0.3, "moyenne": 0.5, "basse": 0.2},
    "proprete": {"haute": 0.2, "moyenne": 0.5, "basse": 0.3},
    "eclairage": {"haute": 0.3, "moyenne": 0.5, "basse": 0.2},
    "espaces_verts": {"haute": 0.1, "moyenne": 0.4, "basse": 0.5},
    "mobilier_urbain": {"haute": 0.15, "moyenne": 0.45, "basse": 0.4},
    "bruit": {"haute": 0.25, "moyenne": 0.5, "basse": 0.25},
    "stationnement": {"haute": 0.2, "moyenne": 0.5, "basse": 0.3},
    "eau_assainissement": {"haute": 0.4, "moyenne": 0.4, "basse": 0.2},
}


def fill_slot(slot_name: str) -> str:
    """Remplit un slot avec une valeur aléatoire"""
    if slot_name == "quartier":
        return random.choice(QUARTIERS)
    if slot_name == "rue":
        return random.choice(RUES)
    if slot_name == "parc":
        return random.choice(PARCS)
    if slot_name == "arret":
        return random.choice(ARRETS)
    if slot_name in SLOTS:
        value = random.choice(SLOTS[slot_name])
        # Résoudre les sous-slots récursivement
        while "{" in value:
            import re
            sub_slots = re.findall(r"\{(\w+)\}", value)
            for sub in sub_slots:
                value = value.replace("{" + sub + "}", fill_slot(sub), 1)
        return value
    return slot_name


def generate_report(categorie: str, date: str) -> dict:
    """Génère un signalement synthétique"""
    template = random.choice(TEMPLATES[categorie])

    # Remplir tous les slots
    text = template
    import re
    while "{" in text:
        slots_found = re.findall(r"\{(\w+)\}", text)
        for slot in slots_found:
            text = text.replace("{" + slot + "}", fill_slot(slot), 1)

    # Déterminer le quartier (extraire de la localisation ou en choisir un)
    quartier = random.choice(QUARTIERS)

    # Urgence pondérée par catégorie
    weights = URGENCE_WEIGHTS[categorie]
    urgence = random.choices(
        list(weights.keys()),
        weights=list(weights.values()),
        k=1
    )[0]

    return {
        "text": text,
        "categorie": categorie,
        "quartier": quartier,
        "date": date,
        "urgence": urgence,
    }


def generate_dataset(n: int = 300) -> list:
    """Génère n signalements répartis sur ~12 semaines"""
    reports = []
    categories = list(TEMPLATES.keys())

    # Pondération réaliste des catégories
    cat_weights = {
        "voirie": 0.20,
        "proprete": 0.22,
        "eclairage": 0.13,
        "espaces_verts": 0.10,
        "mobilier_urbain": 0.10,
        "bruit": 0.08,
        "stationnement": 0.10,
        "eau_assainissement": 0.07,
    }

    # Période : 12 semaines en arrière
    end_date = datetime(2025, 10, 17)
    start_date = end_date - timedelta(weeks=12)
    total_days = (end_date - start_date).days

    for i in range(n):
        cat = random.choices(
            list(cat_weights.keys()),
            weights=list(cat_weights.values()),
            k=1
        )[0]

        # Date aléatoire sur la période
        day_offset = random.randint(0, total_days)
        date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        report = generate_report(cat, date)
        reports.append(report)

    # Trier par date
    reports.sort(key=lambda r: r["date"])
    return reports


if __name__ == "__main__":
    dataset = generate_dataset(300)

    with open("synthetic_reports.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Stats
    from collections import Counter
    cats = Counter(r["categorie"] for r in dataset)
    urgences = Counter(r["urgence"] for r in dataset)
    quartiers = Counter(r["quartier"] for r in dataset)

    print(f"✅ {len(dataset)} signalements générés → synthetic_reports.json")
    print(f"\nRépartition par catégorie:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")
    print(f"\nRépartition par urgence:")
    for urg, count in urgences.most_common():
        print(f"  {urg}: {count}")
    print(f"\nRépartition par quartier:")
    for q, count in quartiers.most_common():
        print(f"  {q}: {count}")