import pytest

from intelligence_layer.connectors.retrievers.document_index_retriever import (
    DocumentIndexRetriever,
)

QUERY = "Who likes pizza?"
TEXTS = [
    "Gegenwart \nDurch italienische Auswanderer verbreitete sich die Pizza gegen Ende des 19. Jahrhunderts auch in den USA. Im Oktober 1937 wurde in Frankfurt am Main erstmals eine Pizza auf dem damaligen Festhallengelände im Rahmen der 7. Internationalen Kochkunst-Ausstellung bei der Messe Frankfurt zubereitet. Nach dem Zweiten Weltkrieg wurde Pizza auch in Europa außerhalb Italiens bekannter. Die erste Pizzeria in Deutschland wurde von Nicolino di Camillo (1921–2015) im März 1952 in Würzburg unter dem Namen Sabbie di Capri eröffnet. Von hier aus begann der Siegeszug der Pizza in Deutschland. Die erste Pizzeria in Wien wurde 1975 von Pasquale Tavella eröffnet. Neben Spaghetti ist die Pizza heute das bekannteste italienische Nationalgericht, sie wird weltweit angeboten.\n\nZubereitung \nZur Zubereitung wird zuerst ein einfacher Hefeteig aus Mehl, Wasser, wenig Hefe, Salz und eventuell etwas Olivenöl hergestellt, gründlich durchgeknetet und nach einer Gehzeit von mindestens einer Stunde bei Zimmertemperatur (bzw. über Nacht im oberen Fach des Kühlschranks) ausgerollt oder mit den bemehlten Händen dünn ausgezogen. Geübte Pizzabäcker ziehen den Teig über den Handrücken und weiten ihn durch Kreisenlassen in der Luft.\n\nDann wird der Teig mit den Zutaten je nach Rezept nicht zu üppig belegt, meist mit passierten Dosentomaten oder Salsa pizzaiola (einer vorher gekochten, sämigen Tomatensauce, die mit Oregano, Basilikum, Knoblauch und anderem kräftig gewürzt ist). Es folgen der Käse (z. B. Mozzarella, Parmesan oder Pecorino) und die übrigen Zutaten, zum Abschluss etwas Olivenöl.\n\nSchließlich wird die Pizza bei einer möglichst hohen Temperatur von 400 bis 500 °C für wenige Minuten kurz gebacken. Dies geschieht in einer möglichst niedrigen Kammer. Ein Stapeln in Einschüben oder separat schaltbare Unter- und Oberhitze ist daher nicht üblich. Der traditionelle Kuppelofen ist gemauert und die Hitze wird über ein Feuer direkt im Backraum erzeugt. Moderne Pizzaöfen werden mit Gas oder Strom beheizt.",
    "Verbreitet in Italien ist auch die Pizza bianca (weiße Pizza), jegliche Pizza-Variation, die ohne Tomatensoße zubereitet wird.\n\nEine Calzone (italienisch für „Hose“) ist eine Pizza, bei welcher der Teigfladen vor dem Backen über dem Belag zusammengeklappt wird. Die traditionelle Füllung besteht aus Ricotta, rohem Schinken, Pilzen, Mozzarella, Parmesan und Oregano. Ursprünglich wurde die Calzone nicht im Ofen, sondern in einer Pfanne in Schmalz oder Öl gebacken, wie es als Pizza fritta in Neapel üblich ist.\n\nIn ganz Italien verbreitet ist die Pizza al taglio („Pizza am Stück“), die auf einem rechteckigen Blech gebacken und in kleineren rechteckigen Stücken verkauft wird. Angeboten wird sie häufig nicht nur in Pizzerien, sondern auch beim Bäcker.\n\nEine neuartige Abwandlung der Pizza ist die Pinsa, die rechteckig und aus einem lockeren Teig gebacken wird.\n\nUS-amerikanische Pizza \nIn den USA sind zwei Typen weit verbreitet, „Chicago-style“ und „New York-style“ Pizza. Während die New Yorker Variante mit ihrem sehr dünnen Boden der italienischen Variante ähnelt, steht die Variante aus Chicago Kopf: Der Teig bildet eine Schüsselform, wird mit Mozzarellascheiben ausgelegt und mit weiteren Zutaten gefüllt. Zum Schluss wird das ganze von oben mit zerkleinerten Tomaten bestrichen und mit Parmesan und Oregano bestreut.\n\nAuch die Pizza Hawaii mit Kochschinken und Ananas ist wahrscheinlich nordamerikanischen Ursprungs.\n\nIn Deutschland ist eine weitere Variante als „American Pizza“ populär, die sich vor allem durch einen dicken, luftigen Boden auszeichnet und u. a. durch die Restaurantkette Pizza Hut bekannt ist.\n\nKoschere Pizza",
]


@pytest.mark.internal
def test_document_index_retriever(
    document_index_retriever: DocumentIndexRetriever,
) -> None:
    documents = document_index_retriever.get_relevant_documents_with_scores(QUERY)
    assert documents[0].document_chunk.text[0:30] in TEXTS[0]
    assert documents[1].document_chunk.text[0:30] in TEXTS[1]
    document_path = documents[0].id
    assert document_path.collection_path == document_index_retriever._collection_path
    assert document_path.document_name == "Pizza"
