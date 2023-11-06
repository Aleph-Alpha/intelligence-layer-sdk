from aleph_alpha_client import Client

from intelligence_layer.core.complete import (
    FewShot,
    FewShotConfig,
    FewShotExample,
    FewShotInput,
    PromptOutput,
)
from intelligence_layer.core.detect_language import (
    DetectLanguage,
    DetectLanguageInput,
    Language,
)
from intelligence_layer.core.logger import DebugLogger
from intelligence_layer.core.task import Task
from intelligence_layer.use_cases.summarize.few_shot_summarize import FewShotSummarize
from intelligence_layer.use_cases.summarize.summarize import (
    SummarizeInput,
    SummarizeOutput,
)


FEW_SHOT_CONFIGS = {
    Language("en"): FewShotConfig(
        instruction="Summarize each text in one to three sentences.",
        examples=[
            FewShotExample(
                input="The startup ecosystem also represents a key success factor for Baden-Württemberg as a business location. It is currently characterized above all by a large number of very active locations such as Mannheim, Karlsruhe and Stuttgart (Kollmann et al. 2020, Bundesverband Deutsche Startups e.V. 2021a). However, in Baden-Württemberg in particular, traditional industries still account for around 30% of gross domestic product, which means that, like other regions, it is massively threatened by structural change (Statistisches Landesamt Baden-Württemberg 2021).",
                response="Since Baden-Württemberg is heavily dependent on traditional industries, startups are all the more important. Start-ups are very active in Mannheim, Karlsruhe and Stuttgart.",
            ),
            FewShotExample(
                input="For political reasons, the study avoids the terms country or state, but breaks down by national economies. In 185 economies, the authors were able to survey prices for a pure data mobile product that meets the minimum requirements mentioned for 2020 as well as 2021. The results show that the global average price has fallen by two percent to USD 9.30 per month. The development varies greatly from region to region. In the Commonwealth of Independent States, the average price has risen by almost half in just one year, to the equivalent of 5.70 US dollars. In the Americas, prices have risen by ten percent to an average of $14.70. While consumers in wealthy economies enjoyed an average price reduction of 13 percent to USD 15.40, there was no cost reduction for consumers in less wealthy countries.",
                response="Prices for data-only mobile products fell by an average of two percent globally to USD 9.30 in 2021. Affluent regions benefited most (13% savings), poorer regions least (no cost reduction).",
            ),
            FewShotExample(
                input="Huawei suffered a 29 percent drop in sales in 2021. No wonder, since the Chinese company is subject to U.S. sanctions and cannot obtain new hardware or software from the U.S. and some other Western countries. Exports are also restricted. Nevertheless, the company's headquarters in Shenzhen reported a lower debt ratio, a 68 percent increase in operating profit and even a 76 percent increase in net profit for 2021. Will Huawei's record profit go down in the history books as the Miracle of Shenzhen despite the slump in sales? Will the brave Chinese be able to outsmart the tech-hungry Americans? Are export bans proving to be a boomerang, hurting the U.S. export business while Huawei prints more money than ever before?",
                response="Huawei's 2021 revenue fell 29%, but at the same time net profit rose 76% and its debt ratio fell. That's unusual.",
            ),
        ],
        input_prefix="Text",
        response_prefix="Summary",
        model="luminous-extended",
        maximum_response_tokens=128,
    )
}


class MediumCompressionSummarize(FewShotSummarize):
    """Summarises a section into a text of medium length.

    Generate a short body natural language summary.

    Args:
        client: Aleph Alpha client instance for running model related API calls.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = MediumCompressionSummarize(client)
        >>> input = SummarizeInput(
                chunk="This is a story about pizza. Tina hates pizza. However, Mike likes it. Pete strongly believes that pizza is the best thing to exist."
            )
        >>> logger = InMemoryLogger(name="MediumCompressionSummarize")
        >>> output = task.run(input, logger)
        >>> print(output.summary)
        Tina does not like pizza, but Mike and Pete do.
    """

    _client: Client

    def __init__(self, client: Client) -> None:
        super().__init__(client, FEW_SHOT_CONFIGS)
