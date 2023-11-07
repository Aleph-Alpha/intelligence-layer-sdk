from aleph_alpha_client import Client
from pytest import fixture

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.complete import FewShotConfig, FewShotExample
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.logger import NoOpDebugLogger
from intelligence_layer.use_cases.summarize.single_chunk_few_shot_summarize import (
    SingleChunkFewShotSummarize,
)
from intelligence_layer.use_cases.summarize.summarize import SingleChunkSummarizeInput


@fixture
def single_chunk_few_shot_summarize(client: Client) -> SingleChunkFewShotSummarize:
    few_shot_configs = {
        Language("en"): FewShotConfig(
            instruction="Summarize each text in one to three sentences.",
            examples=[
                FewShotExample(
                    input="The startup ecosystem also represents a key success factor for Baden-Württemberg as a business location. It is currently characterized above all by a large number of very active locations such as Mannheim, Karlsruhe and Stuttgart (Kollmann et al. 2020, Bundesverband Deutsche Startups e.V. 2021a). However, in Baden-Württemberg in particular, traditional industries still account for around 30% of gross domestic product, which means that, like other regions, it is massively threatened by structural change (Statistisches Landesamt Baden-Württemberg 2021).",
                    response="Startups are gaining importance in Baden-Württemberg.",
                ),
                FewShotExample(
                    input="For political reasons, the study avoids the terms country or state, but breaks down by national economies. In 185 economies, the authors were able to survey prices for a pure data mobile product that meets the minimum requirements mentioned for 2020 as well as 2021. The results show that the global average price has fallen by two percent to USD 9.30 per month. The development varies greatly from region to region. In the Commonwealth of Independent States, the average price has risen by almost half in just one year, to the equivalent of 5.70 US dollars. In the Americas, prices have risen by ten percent to an average of $14.70. While consumers in wealthy economies enjoyed an average price reduction of 13 percent to USD 15.40, there was no cost reduction for consumers in less wealthy countries.",
                    response="While mobile prices are falling globally, they are rising regionally, in some cases sharply.",
                ),
                FewShotExample(
                    input="Huawei suffered a 29 percent drop in sales in 2021. No wonder, since the Chinese company is subject to U.S. sanctions and cannot obtain new hardware or software from the U.S. and some other Western countries. Exports are also restricted. Nevertheless, the company's headquarters in Shenzhen reported a lower debt ratio, a 68 percent increase in operating profit and even a 76 percent increase in net profit for 2021. Will Huawei's record profit go down in the history books as the Miracle of Shenzhen despite the slump in sales? Will the brave Chinese be able to outsmart the tech-hungry Americans? Are export bans proving to be a boomerang, hurting the U.S. export business while Huawei prints more money than ever before?",
                    response="In 2021, Huawei's sales fell, but it made more profit.",
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
                    response="Startups gewinnen in Baden-Württemberg an Bedeutung.",
                ),
                FewShotExample(
                    input="Aus politischen Gründen vermeidet die Studie die Begriffe Land oder Staat, sondern gliedert nach Volkswirtschaften. In 185 Volkswirtschaften konnten die Autoren für 2020 sowie 2021 Preise für ein reines Daten-Mobilfunkprodukt erheben, das die genannten Minimalvoraussetzungen erfüllt. Dabei zeigt sich, dass der globale Durchschnittspreis um zwei Prozent auf 9,30 US-Dollar pro Monat gesunken ist. Die Entwicklung ist regional höchst unterschiedlich. In der Gemeinschaft Unabhängiger Staaten ist der Durchschnittspreis in nur einem Jahr um fast die Hälfte gestiegen, auf umgerechnet 5,70 US-Dollar. Auf den Amerika-Kontinenten gab es einen Preisanstieg um immerhin zehn Prozent auf durchschnittlich 14,70 US-Dollar. Während Verbraucher in wohlhabenden Volkswirtschaften sich über eine durchschnittliche Preissenkung von 13 Prozent auf 15,40 US-Dollar freuen durften, gab es für Konsumenten in weniger wohlhabenden Ländern keine Kostensenkung.",
                    response="Während Mobilfunkpreise global sinken, steigen sie regionals teils stark an.",
                ),
                FewShotExample(
                    input="Einen Umsatzeinbruch von 29 Prozent musste Huawei 2021 hinnehmen. Kein Wunder, ist der chinesische Konzern doch US-Sanktionen ausgesetzt und kann keine neue Hard- noch Software aus den USA und einigen anderen westlichen Ländern beziehen. Auch die Exporte sind eingeschränkt. Dennoch meldet die Konzernzentrale in Shenzhen eine geringere Schuldenquote, um 68 Prozent gestiegenen Betriebsgewinn und gar um 76 Prozent höheren Reingewinn für 2021. Geht Huaweis Rekordgewinn trotz Umsatzeinbruchs als Wunder von Shenzhen in die Geschichtsbücher ein? Schlagen die wackeren Chinesen den technik-geizigen Amis ein Schnippchen? Erweisen sich die Exportverbote als Bumerang, der das US-Exportgeschäft schädigt, während Huawei gleichzeitig mehr Geld druckt als je zuvor?",
                    response="Im Jahr 2021 fiel Huaweis Umsatz, doch es wurde mehr Gewinn gemacht.",
                ),
            ],
            input_prefix="Text",
            response_prefix="Zusammenfassung",
        ),
    }
    return SingleChunkFewShotSummarize(
        client, few_shot_configs, "luminous-extended", 128
    )


def test_high_compression_summarize_en(
    single_chunk_few_shot_summarize: SingleChunkFewShotSummarize,
) -> None:
    text = Chunk(
        "The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]"
    )
    input = SingleChunkSummarizeInput(chunk=text, language=Language("en"))
    output = single_chunk_few_shot_summarize.run(input, NoOpDebugLogger())

    assert output.summary
    assert "bear" in output.summary.lower()
    assert len(output.summary) < len(text)


def test_high_compression_summarize_is_language_sensitive(
    single_chunk_few_shot_summarize: SingleChunkFewShotSummarize,
) -> None:
    text = Chunk(
        "The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.[1][3] In North America, the populations of brown bears are called grizzly bears, while the subspecies that inhabits the Kodiak Islands of Alaska is known as the Kodiak bear. It is one of the largest living terrestrial members of the order Carnivora, rivaled in size only by its closest relative, the polar bear (Ursus maritimus), which is much less variable in size and slightly bigger on average.[4][5][6][7][8] The brown bear's range includes parts of Russia, Central Asia, the Himalayas, China, Canada, the United States, Hokkaido, Scandinavia, Finland, the Balkans, the Picos de Europa and the Carpathian region (especially Romania), Iran, Anatolia, and the Caucasus.[1][9] The brown bear is recognized as a national and state animal in several European countries.[10]"
    )
    input_en = SingleChunkSummarizeInput(chunk=text, language=Language("en"))
    input_de = SingleChunkSummarizeInput(chunk=text, language=Language("de"))
    output_en, output_de = single_chunk_few_shot_summarize.run_concurrently(
        [input_en, input_de], NoOpDebugLogger(), concurrency_limit=2
    )

    assert output_en.summary != output_de.summary
