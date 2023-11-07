from aleph_alpha_client import Client

from intelligence_layer.core.complete import (
    FewShotConfig,
    FewShotExample,
)
from intelligence_layer.core.detect_language import (
    Language,
)
from intelligence_layer.use_cases.summarize.long_context_few_shot_summarize import (
    LongContextFewShotSummarize,
)


FEW_SHOT_CONFIGS = {
    Language("en"): FewShotConfig(
        instruction="Summarize each text in one to three sentences.",
        examples=[
            FewShotExample(
                input="The startup ecosystem also represents a key success factor for Baden-Württemberg as a business location. It is currently characterized above all by a large number of very active locations such as Mannheim, Karlsruhe and Stuttgart (Kollmann et al. 2020, Bundesverband Deutsche Startups e.V. 2021a). However, in Baden-Württemberg in particular, traditional industries still account for around 30% of gross domestic product, which means that, like other regions, it is massively threatened by structural change (Statistisches Landesamt Baden-Württemberg 2021).",
                response="Since Baden-Württemberg is heavily dependent on traditional industries, startups are all the more important. Although startups are very active in Mannheim, Karlsruhe and Stuttgart in particular, traditional industries still account for 30% of the gross domestic product.",
            ),
            FewShotExample(
                input="For political reasons, the study avoids the terms country or state, but breaks down by national economies. In 185 economies, the authors were able to survey prices for a pure data mobile product that meets the minimum requirements mentioned for 2020 as well as 2021. The results show that the global average price has fallen by two percent to USD 9.30 per month. The development varies greatly from region to region. In the Commonwealth of Independent States, the average price has risen by almost half in just one year, to the equivalent of 5.70 US dollars. In the Americas, prices have risen by ten percent to an average of $14.70. While consumers in wealthy economies enjoyed an average price reduction of 13 percent to USD 15.40, there was no cost reduction for consumers in less wealthy countries.",
                response="Prices for data-only mobile products fell by an average of two percent globally to USD 9.30 in 2021. However, developments differed, sometimes significantly, by region. For example, prices in the Americas increased by 10%. Affluent regions benefited the most (13% savings). On average, there was no cost reduction in poorer regions.",
            ),
            FewShotExample(
                input="Huawei suffered a 29 percent drop in sales in 2021. No wonder, since the Chinese company is subject to U.S. sanctions and cannot obtain new hardware or software from the U.S. and some other Western countries. Exports are also restricted. Nevertheless, the company's headquarters in Shenzhen reported a lower debt ratio, a 68 percent increase in operating profit and even a 76 percent increase in net profit for 2021. Will Huawei's record profit go down in the history books as the Miracle of Shenzhen despite the slump in sales? Will the brave Chinese be able to outsmart the tech-hungry Americans? Are export bans proving to be a boomerang, hurting the U.S. export business while Huawei prints more money than ever before?",
                response="Although Huawei suffered a 29% drop in revenue in 2021, net profit rose 76% and its debt ratio fell. The likely reason is believed to be U.S. sanctions that restrict the Chinese group's trade and exports. However, the sanctions could now prove to be a boomerang.",
            ),
        ],
        input_prefix="Text",
        response_prefix="Summary",
    ),
    Language("de"): FewShotConfig(
        instruction="Fasse jeden Text in ein bis drei Sätzen zusammen.",
        examples=[
            FewShotExample(
                input="Auch für den Wirtschaftsstandort Baden-Württemberg stellt das Start-up-Ökosystem einen zentralen Erfolgsfaktor dar. Es zeichnet sich aktuell vor allem durch eine Vielzahl von sehr aktiven Standorten wie Mannheim, Karlsruhe und Stuttgart aus (Kollmann et al. 2020, Bundesverband Deutsche Startups e.V. 2021a). Allerdings entfallen gerade in Baden-Württemberg noch immer rund 30 % des Bruttoinlandsprodukts auf traditionelle Industrien, womit es wie auch andere Regionen massiv durch den Strukturwandel bedroht ist (Statistisches Landesamt Baden-Württemberg 2021).",
                response="Da Baden-Württemberg stark von traditionellen Industrien abhängig ist, sind Start-Ups umso wichtiger. Obwohl Startups gerade in Mannheim, Karlsruhe und Stuttgart sehr aktiv sind, entfallen 30% des Bruttoinlandsprodukts weiterhin auf traditionelle Industrien.",
            ),
            FewShotExample(
                input="Aus politischen Gründen vermeidet die Studie die Begriffe Land oder Staat, sondern gliedert nach Volkswirtschaften. In 185 Volkswirtschaften konnten die Autoren für 2020 sowie 2021 Preise für ein reines Daten-Mobilfunkprodukt erheben, das die genannten Minimalvoraussetzungen erfüllt. Dabei zeigt sich, dass der globale Durchschnittspreis um zwei Prozent auf 9,30 US-Dollar pro Monat gesunken ist. Die Entwicklung ist regional höchst unterschiedlich. In der Gemeinschaft Unabhängiger Staaten ist der Durchschnittspreis in nur einem Jahr um fast die Hälfte gestiegen, auf umgerechnet 5,70 US-Dollar. Auf den Amerika-Kontinenten gab es einen Preisanstieg um immerhin zehn Prozent auf durchschnittlich 14,70 US-Dollar. Während Verbraucher in wohlhabenden Volkswirtschaften sich über eine durchschnittliche Preissenkung von 13 Prozent auf 15,40 US-Dollar freuen durften, gab es für Konsumenten in weniger wohlhabenden Ländern keine Kostensenkung.",
                response="Die Preise für reine Daten-Mobilfunkprodukte sind im Jahr 2021 weltweit um durchschnittlich zwei Prozent auf 9,30 USD gefallen. Jedoch unterscheiden sich die Entwicklungen regional teils stark. Beispielsweise stiegen die Preise auf den amerikanischen Kontinenten um 10% an. Am stärksten profitieren wohlhabende Regionen (13% Ersparnis). In ärmeren Regionen gab es im Durchschnitt keine Kostensenkung.",
            ),
            FewShotExample(
                input="Einen Umsatzeinbruch von 29 Prozent musste Huawei 2021 hinnehmen. Kein Wunder, ist der chinesische Konzern doch US-Sanktionen ausgesetzt und kann keine neue Hard- noch Software aus den USA und einigen anderen westlichen Ländern beziehen. Auch die Exporte sind eingeschränkt. Dennoch meldet die Konzernzentrale in Shenzhen eine geringere Schuldenquote, um 68 Prozent gestiegenen Betriebsgewinn und gar um 76 Prozent höheren Reingewinn für 2021. Geht Huaweis Rekordgewinn trotz Umsatzeinbruchs als Wunder von Shenzhen in die Geschichtsbücher ein? Schlagen die wackeren Chinesen den technik-geizigen Amis ein Schnippchen? Erweisen sich die Exportverbote als Bumerang, der das US-Exportgeschäft schädigt, während Huawei gleichzeitig mehr Geld druckt als je zuvor?",
                response="Obwohl Huawei im Jahr 2021 einen Umsatzrückgang von 29 Prozent hinnehmen musste, stieg der Reingewinn um 76% und die Schuldenquote sank. Als Grund gelten wahrscheinlich die US-Sanktionen, die Handel und Exporte des chinesischen Konzerns einschränken. Die Sanktionen könnten sich nun jedoch als Bumerang erweisen.",
            ),
        ],
        input_prefix="Text",
        response_prefix="Zusammenfassung",
    ),
    Language("es"): FewShotConfig(
        instruction="Resuma cada texto en una o tres frases.",
        examples=[
            FewShotExample(
                input="El ecosistema de las start-ups es también un factor clave para el éxito de Baden-Württemberg como lugar de negocios. Actualmente se caracteriza sobre todo por un gran número de localidades muy activas como Mannheim, Karlsruhe y Stuttgart (Kollmann et al. 2020, Bundesverband Deutsche Startups e.V. 2021a). Sin embargo, en Baden-Württemberg, en particular, las industrias tradicionales siguen representando alrededor del 30% del producto interior bruto, lo que significa que, al igual que otras regiones, está muy amenazada por el cambio estructural (Statistisches Landesamt Baden-Württemberg 2021).",
                response="Dado que Baden-Württemberg depende en gran medida de las industrias tradicionales, las empresas de nueva creación son aún más importantes. Aunque las start-ups son muy activas en Mannheim, Karlsruhe y Stuttgart en particular, las industrias tradicionales siguen representando el 30% del producto interior bruto.",
            ),
            FewShotExample(
                input="Por razones políticas, el estudio evita los términos país o estado, sino que se desglosa por economías nacionales. En 185 economías, los autores pudieron estudiar los precios de un producto móvil de datos puros para 2020 y 2021 que cumple los requisitos mínimos mencionados. Los resultados muestran que el precio medio global ha bajado un 2%, hasta los 9,30 dólares mensuales. El desarrollo varía mucho de una región a otra. En la Comunidad de Estados Independientes, el precio medio ha subido casi la mitad en sólo un año, hasta el equivalente a 5,70 dólares estadounidenses. En el continente americano, el precio subió un diez por ciento, hasta una media de 14,70 dólares. Mientras que los consumidores de las economías ricas disfrutaron de una reducción media de precios del 13%, hasta los 15,40 dólares, los consumidores de los países menos ricos no vieron ninguna reducción de costes.",
                response="Los precios de los productos de telefonía móvil de sólo datos bajaron en todo el mundo una media del 2%, hasta los 9,30 dólares en 2021. Sin embargo, la evolución difiere a nivel regional, a veces de forma significativa. Por ejemplo, los precios en América aumentaron un 10%. Las regiones ricas son las más beneficiadas (13% de ahorro). Por término medio, no hubo reducción de costes en las regiones más pobres.",
            ),
            FewShotExample(
                input="Huawei sufrió una caída del 29% en sus ventas en 2021. No es de extrañar, ya que la empresa china está sometida a sanciones estadounidenses y no puede obtener nuevo hardware o software de EE.UU. y algunos otros países occidentales. Las exportaciones también están restringidas. Sin embargo, la sede central de la empresa en Shenzhen informó de un menor ratio de endeudamiento, un aumento del 68% en el beneficio operativo e incluso un incremento del 76% en el beneficio neto para 2021. ¿Pasará el récord de beneficios de Huawei a los libros de historia como el Milagro de Shenzhen a pesar del desplome de las ventas? ¿Serán los valientes chinos capaces de burlar a los expertos en tecnología estadounidenses? ¿Las prohibiciones a las exportaciones están siendo un bumerán, perjudicando al negocio de las exportaciones estadounidenses, mientras que al mismo tiempo Huawei está imprimiendo más dinero que nunca?",
                response="Aunque Huawei sufrió un descenso del 29% en sus ingresos en 2021, el beneficio neto aumentó un 76% y su ratio de endeudamiento se redujo. La razón es probablemente las sanciones de Estados Unidos, que restringen el comercio y las exportaciones del grupo chino. Sin embargo, las sanciones podrían resultar ahora un bumerán.",
            ),
        ],
        input_prefix="Texto",
        response_prefix="Resumen",
    ),
    Language("fr"): FewShotConfig(
        instruction="Résume chaque texte en une à trois phrases.",
        examples=[
            FewShotExample(
                input="L'écosystème des start-ups représente également un facteur de succès central pour le site économique du Bade-Wurtemberg. Il se caractérise actuellement surtout par un grand nombre de sites très actifs comme Mannheim, Karlsruhe et Stuttgart (Kollmann et al. 2020, Bundesverband Deutsche Startups e.V. 2021a). Cependant, dans le Bade-Wurtemberg en particulier, les industries traditionnelles représentent encore environ 30 % du produit intérieur brut, ce qui fait que le Bade-Wurtemberg, comme d'autres régions, est massivement menacé par le changement structurel (Statistisches Landesamt Baden-Württemberg 2021).",
                response="Le Bade-Wurtemberg étant fortement dépendant des industries traditionnelles, les start-ups sont d'autant plus importantes. Bien que les start-ups soient très actives, notamment à Mannheim, Karlsruhe et Stuttgart, les industries traditionnelles représentent toujours 30% du produit intérieur brut.",
            ),
            FewShotExample(
                input="Pour des raisons politiques, l'étude évite les termes de pays ou d'État, mais s'articule autour des économies nationales. Dans 185 économies, les auteurs ont pu relever les prix pour 2020 et 2021 d'un produit de téléphonie mobile de données pur qui remplit les conditions minimales mentionnées. Il en ressort que le prix moyen mondial a baissé de 2 % pour atteindre 9,30 dollars US par mois. L'évolution varie fortement d'une région à l'autre. Dans la Communauté des États indépendants, le prix moyen a augmenté de près de moitié en un an seulement, pour atteindre l'équivalent de 5,70 dollars américains. Sur le continent américain, les prix ont augmenté de 10 % pour atteindre 14,70 dollars en moyenne. Alors que les consommateurs des économies prospères ont pu se réjouir d'une baisse moyenne des prix de 13% à 15,40 dollars US, les consommateurs des pays moins prospères n'ont pas connu de baisse des coûts.",
                response="Les prix des produits mobiles de données pures ont baissé en moyenne de 2 % dans le monde en 2021, pour atteindre 9,30 dollars. Toutefois, les évolutions varient fortement d'une région à l'autre. Par exemple, les prix ont augmenté de 10% sur le continent américain. Ce sont les régions riches qui en profitent le plus (13% d'économies). Dans les régions plus pauvres, il n'y a eu en moyenne aucune réduction des coûts.",
            ),
            FewShotExample(
                input="Huawei a subi une chute de 29 pour cent de son chiffre d'affaires en 2021. Rien d'étonnant à cela, puisque le groupe chinois est soumis à des sanctions américaines et ne peut plus acheter de nouveaux matériels ou logiciels aux États-Unis et dans certains autres pays occidentaux. Les exportations sont également limitées. Malgré cela, le siège du groupe à Shenzhen annonce un taux d'endettement plus faible, un bénéfice d'exploitation en hausse de 68 pour cent et même un bénéfice net en hausse de 76 pour cent pour 2021. Le bénéfice record de Huawei malgré la chute du chiffre d'affaires entrera-t-il dans les livres d'histoire comme le miracle de Shenzhen ? Les braves Chinois déjouent-ils les ambitions technologiques des Américains ? Les interdictions d'exportation se révèlent-elles être un boomerang qui nuit au commerce d'exportation américain, alors que dans le même temps Huawei imprime plus d'argent que jamais ?",
                response="Bien que Huawei ait vu son chiffre d'affaires baisser de 29% en 2021, son bénéfice net a augmenté de 76% et son taux d'endettement a baissé. Les sanctions américaines, qui limitent le commerce et les exportations du groupe chinois, en sont probablement la raison. Toutefois, ces sanctions pourraient désormais avoir un effet boomerang.",
            ),
        ],
        input_prefix="Texte",
        response_prefix="Résumé",
    ),
    Language("it"): FewShotConfig(
        instruction="Riassumete ogni testo in una o tre frasi.",
        examples=[
            FewShotExample(
                input="Anche l'ecosistema delle start-up è un fattore chiave di successo per il Baden-Württemberg come sede d'affari. Attualmente è caratterizzata soprattutto da un gran numero di località molto attive come Mannheim, Karlsruhe e Stoccarda (Kollmann et al. 2020, Bundesverband Deutsche Startups e.V. 2021a). Tuttavia, nel Baden-Württemberg in particolare, le industrie tradizionali rappresentano ancora circa il 30% del prodotto interno lordo, il che significa che, come altre regioni, è fortemente minacciato dal cambiamento strutturale (Statistisches Landesamt Baden-Württemberg 2021).",
                response="Poiché il Baden-Württemberg è fortemente dipendente dalle industrie tradizionali, le start-up sono ancora più importanti. Sebbene le start-up siano molto attive soprattutto a Mannheim, Karlsruhe e Stoccarda, le industrie tradizionali rappresentano ancora il 30% del prodotto interno lordo.",
            ),
            FewShotExample(
                input="Per ragioni politiche, lo studio evita i termini paese o stato, ma suddivide per economie nazionali. In 185 economie, gli autori sono stati in grado di rilevare i prezzi di un prodotto mobile di dati puro per il 2020 e il 2021 che soddisfa i requisiti minimi citati. I risultati mostrano che il prezzo medio globale è sceso del 2% a 9,30 dollari USA al mese. Lo sviluppo varia notevolmente da regione a regione. Nella Comunità degli Stati Indipendenti, il prezzo medio è aumentato di quasi la metà in un solo anno, fino all'equivalente di 5,70 dollari USA. Nelle Americhe, i prezzi sono aumentati fino al 10%, raggiungendo una media di 14,70 dollari USA. Mentre i consumatori delle economie ricche hanno beneficiato di una riduzione media dei prezzi del 13%, passando a 15,40 dollari USA, i consumatori dei Paesi meno ricchi non hanno registrato alcuna riduzione dei costi.",
                response="I prezzi dei prodotti di telefonia mobile solo dati sono diminuiti in media del 2% a livello globale, raggiungendo i 9,30 dollari nel 2021. Tuttavia, gli sviluppi differiscono a livello regionale, a volte in modo significativo. Ad esempio, i prezzi nelle Americhe sono aumentati del 10%. Le regioni ricche sono quelle che ne beneficiano di più (13% di risparmio). Nelle regioni più povere non si è registrata in media alcuna riduzione dei costi.",
            ),
            FewShotExample(
                input="Huawei ha subito un crollo delle vendite del 29% nel 2021. Non c'è da stupirsi, visto che l'azienda cinese è soggetta a sanzioni statunitensi e non può ottenere nuovo hardware o software dagli Stati Uniti e da alcuni altri Paesi occidentali. Anche le esportazioni sono limitate. Ciononostante, la sede centrale dell'azienda a Shenzhen ha registrato un rapporto di indebitamento più basso, un aumento del 68% dell'utile operativo e addirittura un aumento del 76% dell'utile netto per il 2021. Il profitto record di Huawei entrerà nei libri di storia come il miracolo di Shenzhen nonostante il crollo delle vendite? Riusciranno i coraggiosi cinesi a superare in astuzia gli americani esperti di tecnologia? I divieti all'esportazione si stanno rivelando un boomerang, danneggiando l'attività di esportazione degli Stati Uniti, mentre allo stesso tempo Huawei sta stampando più denaro che mai?",
                response="Sebbene Huawei abbia subito un calo dei ricavi del 29% nel 2021, l'utile netto è aumentato del 76% e il rapporto di indebitamento è diminuito. Il motivo è probabilmente da ricercare nelle sanzioni statunitensi, che limitano il commercio e le esportazioni del gruppo cinese. Tuttavia, le sanzioni potrebbero ora rivelarsi un boomerang.",
            ),
        ],
        input_prefix="Testo",
        response_prefix="Sintesi",
    ),
}


class LongContextLowCompressionSummarize(LongContextFewShotSummarize):
    """Condenses a text into a long summary.

    Leverages few-shot prompting to generate a summary.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = LongContextLowCompressionSummarize(client)
        >>> input = SummarizeInput(
                text="This is a story about pizza. Tina hates pizza. However, Mike likes it. Pete strongly believes that pizza is the best thing to exist."
            )
        >>> logger = InMemoryLogger(name="LongContextLowCompressionSummarize")
        >>> output = task.run(input, logger)
        >>> print(output.summary)
        Tina doesn't like pizza, but Mike and Pete do.
    """

    _client: Client

    def __init__(
        self, client: Client, fallback_language: Language = Language("en")
    ) -> None:
        super().__init__(
            client=client,
            few_shot_configs=FEW_SHOT_CONFIGS,
            model="luminous-extended",
            max_generated_tokens=160,
            max_tokens_per_chunk=200,
            allowed_languages=[
                Language(lang) for lang in ["en", "de", "es", "fr", "it"]
            ],
            fallback_language=fallback_language,
        )
