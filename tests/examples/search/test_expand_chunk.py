from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Optional

import pytest
from pytest import fixture

from intelligence_layer.connectors import (
    BaseRetriever,
    Document,
    DocumentChunk,
    QdrantInMemoryRetriever,
    SearchResult,
)
from intelligence_layer.connectors.limited_concurrency_client import (
    AlephAlphaClientProtocol,
)
from intelligence_layer.connectors.retrievers.qdrant_in_memory_retriever import (
    RetrieverType,
)
from intelligence_layer.core import LuminousControlModel, NoOpTracer
from intelligence_layer.examples import ExpandChunks, ExpandChunksInput


@fixture
def in_memory_retriever_documents() -> Sequence[Document]:
    return [
        Document(
            text="""In the rolling verdant hills of a realm untouched by the passage of modern times, a kingdom thrived under the rule of a benevolent monarch. The king, known for his wisdom and justice, held the loyalty of his people and the respect of his peers. However, beneath the surface of peace, a shadow loomed that would test the mettle of the kingdom's most valiant defenders: the knights.

These knights, clad in gleaming armor and bearing the colors of their liege, were not mere soldiers but champions of the realm's ideals. They were sworn to protect the innocent, uphold justice, and maintain the peace, guided by a chivalric code that was as much a part of them as the swords they wielded. Among these noble warriors, Sir Aelwyn stood prominent, known across the land for his prowess in battle and his unyielding honor.

Sir Aelwyn, the youngest knight ever to be granted the title of Master of the Horse, was a figure of legend. His tales were told in every corner of the kingdom, often embellished with each retelling. From his duel with the Giant of Gormouth to his silent vigil in the Haunted Wood, Aelwyn's life was a tapestry of bravery and adventure. Yet, his greatest challenge lay ahead, whispered in fearful murmurs throughout the castle—the rise of the Dragon of Black Hollow.

The dragon had awoken from a centuries-long slumber, driven by hunger and wrath, laying waste to the villages on the kingdom's fringes. Smoke and despair rose from the once tranquil borders, drawing the attention of the king and his council. With the threat growing each day, the king summoned Sir Aelwyn and tasked him with a quest that could either save the kingdom or doom it forever—to defeat the dragon.

As Sir Aelwyn prepared for his journey, the castle buzzed with activity. Blacksmiths forged new armor and weapons, alchemists concocted potent draughts, and scholars poured over ancient texts seeking any knowledge that might aid him. The knight spent his nights in the chapel, praying for strength and wisdom, and his days in the training yard, honing his skills against opponents both real and imagined.

Accompanying Sir Aelwyn were his loyal companions: Sir Rowan, a strategist known for his cunning and intellect; Lady Elara, a knight whose skill with the bow was unmatched; and Dame Miriel, a warrior-poet whose songs could stir the soul as fiercely as her sword could cleave armor. Together, they represented the kingdom's finest, united under a single cause.

Their journey was fraught with peril. They crossed through the Whispering Forest, where shadows moved with minds of their own, and over the Mountains of Echoes, where the wind carried voices from the past. Each step brought them closer to their quarry, and the signs of the dragon's passage grew ever more ominous—the charred earth, the ruins of once happy homes, and the air heavy with the scent of sulfur.

As they approached Black Hollow, the landscape grew bleak, and the sky darkened. The dragon, coiled atop a pile of gold and bones, awaited them, its scales shimmering like molten rock. The air crackled with the heat of its breath, and its eyes, glowing like coals, fixed on Sir Aelwyn and his companions.

The battle was fierce. Sir Rowan directed their movements with precision, while Lady Elara loosed arrows that found chinks in the dragon's armor. Dame Miriel's voice rose above the clamor, her words bolstering their courage and blinding the beast with bursts of radiant light. Sir Aelwyn faced the dragon head-on, his shield absorbing the flames that poured from its maw, his sword striking with the weight of his oath behind each blow.

Hours seemed like days as the clash continued, the outcome uncertain. Finally, seeing an opening, Sir Aelwyn drove his sword deep into the dragon's heart. With a final roar that shook the heavens, the dragon fell, its reign of terror ended.

The return to the kingdom was triumphant. The people lined the streets, showering the knights with flowers and cheers. The king welcomed them back as heroes, their deeds to be recorded in the annals of history for generations to come. Sir Aelwyn and his companions had not only saved the kingdom but had also reaffirmed the values it stood for: courage, honor, and a steadfast commitment to the protection of the realm.

As the celebrations faded, Sir Aelwyn looked out over the kingdom from the castle's highest tower. The peace they had fought for lay stretched before him, a tapestry of green fields and bustling towns. Yet, he knew that this peace was not permanent but a precious moment to be cherished and protected. For as long as there were threats to the realm, there would be knights to face them, their swords ready and their hearts brave.

In this timeless land, the cycle of challenge and triumph continued, each generation of knights rising to meet the dangers of their times with the same valor and resolve as those who had come before them. And so, the legends grew, each knight adding their thread to the ever-unfolding story of the kingdom and its defenders."""
        )
    ]


def build_expand_chunk_input(
    document: Document, index_ranges: Sequence[tuple[int, int]]
) -> ExpandChunksInput[int]:
    return ExpandChunksInput(
        document_id=0,
        chunks_found=[
            DocumentChunk(
                text=document.text[index_range[0] : index_range[1]],
                start=index_range[0],
                end=index_range[1],
            )
            for index_range in index_ranges
        ],
    )


@fixture
def wholly_included_expand_chunk_input(
    in_memory_retriever_documents: Sequence[Document],
) -> ExpandChunksInput[int]:
    assert len(in_memory_retriever_documents) == 1
    start_index, end_index = (
        int(len(in_memory_retriever_documents[0].text) * 0.5),
        int(len(in_memory_retriever_documents[0].text) * 0.55),
    )

    return build_expand_chunk_input(
        in_memory_retriever_documents[0], [(start_index, end_index)]
    )


@fixture
def overlapping_expand_chunk_input(
    in_memory_retriever_documents: Sequence[Document],
) -> ExpandChunksInput[int]:
    assert len(in_memory_retriever_documents) == 1
    start_index, end_index = (
        int(len(in_memory_retriever_documents[0].text) * 0.2),
        int(len(in_memory_retriever_documents[0].text) * 0.8),
    )

    return build_expand_chunk_input(
        in_memory_retriever_documents[0], [(start_index, end_index)]
    )


@fixture
def multiple_chunks_expand_chunk_input(
    in_memory_retriever_documents: Sequence[Document],
) -> ExpandChunksInput[int]:
    assert len(in_memory_retriever_documents) == 1
    start_index_1, end_index_1 = (
        int(len(in_memory_retriever_documents[0].text) * 0.3),
        int(len(in_memory_retriever_documents[0].text) * 0.4),
    )
    start_index_2, end_index_2 = (
        int(len(in_memory_retriever_documents[0].text) * 0.45),
        int(len(in_memory_retriever_documents[0].text) * 0.6),
    )

    return build_expand_chunk_input(
        in_memory_retriever_documents[0],
        [(start_index_1, end_index_1), (start_index_2, end_index_2)],
    )


@pytest.mark.skip("Flaky test")
def test_expand_chunk_works_for_wholly_included_chunk(
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
    luminous_control_model: LuminousControlModel,
    wholly_included_expand_chunk_input: ExpandChunksInput[int],
    no_op_tracer: NoOpTracer,
) -> None:
    expand_chunk_task = ExpandChunks(
        asymmetric_in_memory_retriever, luminous_control_model, 256
    )
    expand_chunk_output = expand_chunk_task.run(
        wholly_included_expand_chunk_input, no_op_tracer
    )

    assert (
        len(expand_chunk_output.chunks)
        == 1
        == len(wholly_included_expand_chunk_input.chunks_found)
    )
    assert (
        wholly_included_expand_chunk_input.chunks_found[0].text
        in expand_chunk_output.chunks[0].chunk
    )


@pytest.mark.skip("Flaky test")
def test_expand_chunk_works_for_overlapping_chunk(
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
    luminous_control_model: LuminousControlModel,
    overlapping_expand_chunk_input: ExpandChunksInput[int],
    no_op_tracer: NoOpTracer,
) -> None:
    expand_chunk_task = ExpandChunks(
        asymmetric_in_memory_retriever, luminous_control_model, 256
    )
    expand_chunk_output = expand_chunk_task.run(
        overlapping_expand_chunk_input, no_op_tracer
    )

    assert len(expand_chunk_output.chunks) == 4


@pytest.mark.skip("Flaky test")
def test_expand_chunk_works_for_multiple_chunks(
    asymmetric_in_memory_retriever: QdrantInMemoryRetriever,
    luminous_control_model: LuminousControlModel,
    multiple_chunks_expand_chunk_input: ExpandChunksInput[int],
    no_op_tracer: NoOpTracer,
) -> None:
    expand_chunk_task = ExpandChunks(
        asymmetric_in_memory_retriever, luminous_control_model, 256
    )
    expand_chunk_output = expand_chunk_task.run(
        multiple_chunks_expand_chunk_input, no_op_tracer
    )

    assert len(expand_chunk_output.chunks) == 3

    combined_chunks = "".join(chunk.chunk for chunk in expand_chunk_output.chunks)
    for chunk_found in multiple_chunks_expand_chunk_input.chunks_found:
        assert chunk_found.text in combined_chunks


def test_expand_chunk_is_fast_with_large_document(
    client: AlephAlphaClientProtocol,
    luminous_control_model: LuminousControlModel,
    no_op_tracer: NoOpTracer,
) -> None:
    retriever = QdrantInMemoryRetriever(
        [Document(text="""test text\n""" * 100)],
        client=client,
        k=2,
        retriever_type=RetrieverType.ASYMMETRIC,
    )
    expand_chunk_input = ExpandChunksInput(
        document_id=0,
        chunks_found=[
            DocumentChunk(
                text="test text\n" * 10,
                start=50,
                end=60,
            )
        ],
    )
    expand_chunk_task = ExpandChunks(retriever, luminous_control_model, 256)

    time = datetime.now()
    output = expand_chunk_task.run(expand_chunk_input, no_op_tracer)
    elapsed = datetime.now() - time

    assert len(output.chunks) == 1
    assert elapsed < timedelta(seconds=10)


class FakeRetriever(BaseRetriever[str]):
    def __init__(self, result: str) -> None:
        super().__init__()
        self.result = result

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> Sequence[SearchResult[str]]:
        return []

    def get_full_document(self, id: str) -> Optional[Document]:
        return Document(text=self.result)


def test_expand_chunks_works_if_chunk_of_interest_is_outside_first_large_chunk(
    luminous_control_model: LuminousControlModel,
    no_op_tracer: NoOpTracer,
) -> None:
    # given
    task_input = ExpandChunksInput(
        document_id="id",
        chunks_found=[
            DocumentChunk(
                text="",
                start=1500,  # outside of first large chunk boundary, which is ~1200
                end=1505,
            )
        ],
    )
    full_text = " ".join(str(i) for i in range(1000))
    max_chunk_size = 10
    expand_chunk_task = ExpandChunks(
        FakeRetriever(result=full_text),
        luminous_control_model,
        max_chunk_size=max_chunk_size,
    )
    res = expand_chunk_task.run(task_input, no_op_tracer)
    assert len(res.chunks) > 0
    assert len(res.chunks[0].chunk.strip().split(" ")) == max_chunk_size
