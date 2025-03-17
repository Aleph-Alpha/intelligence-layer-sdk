from pytest import fixture

from intelligence_layer.core import (
    ChunkInput,
    ChunkWithIndices,
    LuminousControlModel,
    NoOpTracer,
)


@fixture
def chunk_input() -> ChunkInput:
    return ChunkInput(
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


def test_chunk_with_indices(
    llama_control_model: LuminousControlModel,
    chunk_input: ChunkInput,
    no_op_tracer: NoOpTracer,
) -> None:
    chunk_with_indices = ChunkWithIndices(llama_control_model, max_tokens_per_chunk=128)

    output = chunk_with_indices.do_run(chunk_input, no_op_tracer)

    assert all(
        c.start_index < output.chunks_with_indices[idx + 1].start_index
        for idx, c in enumerate(output.chunks_with_indices[:-1])
    )
    assert all(
        c.end_index == output.chunks_with_indices[idx + 1].start_index
        for idx, c in enumerate(output.chunks_with_indices[:-1])
    )
