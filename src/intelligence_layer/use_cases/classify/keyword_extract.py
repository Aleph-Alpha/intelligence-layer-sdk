from typing import Mapping

from aleph_alpha_client import Client
from pydantic import BaseModel

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.complete import (
    FewShot,
    FewShotConfig,
    FewShotExample,
    FewShotInput,
)
from intelligence_layer.core.detect_language import Language, LanguageNotSupportedError
from intelligence_layer.core.task import Task
from intelligence_layer.core.tracer import TaskSpan

FEW_SHOT_CONFIGS = {
    Language("de"): FewShotConfig(
        instruction="Erkenne das Thema des Texts.",
        examples=[
            FewShotExample(
                input="Das Gruppenspiel der deutschen Nationalmannschaft gegen Costa Rica hat in Deutschland die bislang höchste TV-Zuschauerzahl erreicht. Den bisherigen Einschaltquoten-Rekord der DFB-Frauen bei der EM 2022 gegen England knackten die Männer allerdings nicht.",
                response="Fußball, Fernsehen, Nationalmannschaft",
            ),
            FewShotExample(
                input="Machine Learning und Deep Learning sind nicht immer die beste Wahl um ein Vorhersageproblem zu lösen. Oft mal kann ein simples lineares Modell schon ausreichen.",
                response="KI, Technolgie, Machine Learning",
            ),
            FewShotExample(
                input='Das Restaurant hat fünf Köche und sechs Servicekräfte, braucht aber ein oder zwei weitere Köche und zwei weitere Servicekräfte. Es ist nicht das einzige Restaurant, das von einer "existenziellen Bedrohung" für das Gastgewerbe betroffen ist.',
                response="Restaurant, Personalmangel, Gastgewerbe",
            ),
            FewShotExample(
                input="Es ist das natürliche Verhalten von Tieren, ihren Instinkten zu folgen. So gehen sie beispielsweise auf die Jagd obwohl sie nicht mal Hunger haben.",
                response="Biologie, Natur, Tiere",
            ),
        ],
        input_prefix="Text",
        response_prefix="Thema",
    ),
    Language("en"): FewShotConfig(
        instruction="Identify the theme of the text.",
        examples=[
            FewShotExample(
                input="The German national team's group match against Costa Rica achieved the highest TV viewership to date in Germany. However, the men did not break the previous ratings record set by the DFB women against England at the 2022 European Championship.",
                response="soccer, television, national team",
            ),
            FewShotExample(
                input="Machine learning and deep learning are not always the best choice to solve a prediction problem. Often times, a simple linear model can suffice.",
                response="AI, Technology, Machine Learning",
            ),
            FewShotExample(
                input='The restaurant has five cooks and six servers, but needs one or two more cooks and two more servers. It\'s not the only restaurant facing an "existential threat" to the hospitality industry.',
                response="restaurant, staff shortage, hospitality industry",
            ),
            FewShotExample(
                input="It is the natural behavior of animals to follow their instincts. For example, they go hunting even though they are not even hungry.",
                response="Biology, Nature, Animals",
            ),
        ],
        input_prefix="Text",
        response_prefix="Topic",
    ),
    Language("es"): FewShotConfig(
        instruction="Identificar el tema del texto.",
        examples=[
            FewShotExample(
                input="El partido de grupo de la selección alemana contra Costa Rica ha alcanzado la mayor audiencia televisiva en Alemania hasta la fecha. Sin embargo, los hombres no batieron el anterior récord de audiencia establecido por las mujeres de la DFB contra Inglaterra en la Eurocopa de 2022.",
                response="Fútbol, Televisión, Selección Nacional",
            ),
            FewShotExample(
                input="El aprendizaje automático y el aprendizaje profundo no siempre son la mejor opción para resolver un problema de predicción. A menudo basta con un modelo lineal simple.",
                response="IA, Tecnología, Aprendizaje automático",
            ),
            FewShotExample(
                input='El restaurante tiene cinco cocineros y seis camareros, pero necesita uno o dos cocineros y dos camareros más. No es el único restaurante afectado por una "amenaza existencial" para la hostelería.',
                response="restaurante, escasez de personal, hostelería",
            ),
            FewShotExample(
                input="El comportamiento natural de los animales es seguir sus instintos. Por ejemplo, salen a cazar aunque ni siquiera tengan hambre.",
                response="Biología, Naturaleza, Animales",
            ),
        ],
        input_prefix="Texto",
        response_prefix="Tema",
    ),
    Language("fr"): FewShotConfig(
        instruction="Identifie le thème du texte.",
        examples=[
            FewShotExample(
                input="Le match de groupe de l'équipe nationale allemande contre le Costa Rica a atteint en Allemagne le plus grand nombre de téléspectateurs à ce jour. Les hommes n'ont toutefois pas battu le record d'audience détenu jusqu'à présent par les femmes de la DFB lors de l'Euro 2022 contre l'Angleterre.",
                response="Football, Télévision, Équipe nationale",
            ),
            FewShotExample(
                input="Le Machine Learning et le Deep Learning ne sont pas toujours le meilleur choix pour résoudre un problème de prédiction. Souvent, un simple modèle linéaire peut suffire.",
                response="IA, Technologie, Machine Learning",
            ),
            FewShotExample(
                input="Le restaurant compte cinq cuisiniers et six serveurs, mais il a besoin d'un ou deux cuisiniers et de deux serveurs supplémentaires. Ce n'est pas le seul restaurant touché par une \"menace existentielle\" pour le secteur de la restauration.",
                response="restaurant, manque de personnel, hôtellerie et restauration",
            ),
            FewShotExample(
                input="C'est le comportement naturel des animaux de suivre leurs instincts. Ils partent par exemple à la chasse alors qu'ils n'ont même pas faim.",
                response="Biologie, Nature, Animaux",
            ),
        ],
        input_prefix="Texte",
        response_prefix="Thème",
    ),
    Language("it"): FewShotConfig(
        instruction="Identificare il tema del testo.",
        examples=[
            FewShotExample(
                input="La partita di girone della nazionale tedesca contro la Costa Rica ha raggiunto il più alto numero di spettatori televisivi in Germania fino ad ora. Tuttavia, gli uomini non hanno superato il precedente record di valutazione stabilito dalle donne della DFB contro l'Inghilterra agli Europei del 2022.",
                response="Calcio, Televisione, Squadra nazionale",
            ),
            FewShotExample(
                input="L'apprendimento automatico e l'apprendimento profondo non sono sempre la scelta migliore per risolvere un problema di previsione. Spesso può bastare un semplice modello lineare.",
                response="AI, Tecnologia, Apprendimento automatico",
            ),
            FewShotExample(
                input="Il ristorante ha cinque cuochi e sei camerieri, ma ha bisogno di uno o due cuochi e due camerieri in più. Non è l'unico ristorante colpito da una \"minaccia esistenziale\" per l'industria dell'ospitalità.",
                response="ristorante, carenza di personale, industria dell'ospitalità",
            ),
            FewShotExample(
                input="Seguire l'istinto è il comportamento naturale degli animali. Per esempio, vanno a caccia anche se non hanno nemmeno fame.",
                response="Biologia, Natura, Animali",
            ),
        ],
        input_prefix="Testo",
        response_prefix="Argomento",
    ),
}


class KeywordExtractInput(BaseModel):
    chunk: Chunk
    language: Language


class KeywordExtractOutput(BaseModel):
    keywords: frozenset[str]


class KeywordExtract(Task[KeywordExtractInput, KeywordExtractOutput]):
    def __init__(
        self,
        client: Client,
        few_shot_configs: Mapping[Language, FewShotConfig] = FEW_SHOT_CONFIGS,
        model: str = "luminous-base",
        maximum_tokens: int = 32,
    ) -> None:
        self._few_shot_configs = few_shot_configs
        self._few_shot = FewShot(client)
        self._model = model
        self._maximum_tokens = maximum_tokens

    def do_run(
        self, input: KeywordExtractInput, task_span: TaskSpan
    ) -> KeywordExtractOutput:
        config = self._few_shot_configs.get(input.language)
        if config is None:
            raise LanguageNotSupportedError(
                f"{input.language} not in ({', '.join(self._few_shot_configs.keys())})"
            )
        result = self._few_shot.run(
            FewShotInput(
                few_shot_config=config,
                input=input.chunk,
                model=self._model,
                maximum_response_tokens=self._maximum_tokens,
            ),
            task_span,
        )
        return KeywordExtractOutput(
            keywords=frozenset(s.strip() for s in result.response.split(","))
        )
