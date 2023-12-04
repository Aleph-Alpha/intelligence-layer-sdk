import os
from aleph_alpha_client import Client
from intelligence_layer.core import InMemoryTracer, FileTracer, CompositeTracer, Chunk
from intelligence_layer.use_cases import PromptBasedClassify, ClassifyInput


tracer_1 = InMemoryTracer()
tracer_2 = InMemoryTracer()
tracer = CompositeTracer([tracer_1, tracer_2])
aa_client = Client(os.getenv("AA_TOKEN"))
task = PromptBasedClassify(aa_client)
task.run(ClassifyInput(chunk=Chunk("Cool"), labels=frozenset({"label", "other label"})), tracer)