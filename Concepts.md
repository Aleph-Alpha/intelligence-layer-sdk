# Concepts

## Task

At the heart of the Intelligence Layer is a `Task`. A Task is actually a pretty generic concept that just
transforms an input-parameter to an output like a function in mathematics.

```
Task: Input -> Output
```

In Python this is expressed through an abstract class with type-parameters and the abstract method `do_run`
where the actual transformation is implemented:

```Python
class Task(ABC, Generic[Input, Output]):

    @abstractmethod
    def do_run(self, input: Input, task_span: TaskSpan) -> Output:
        ...
```

`Input` and `Output` are normal Python datatypes that can be serialized from and to JSON. For this the Intelligence
Layer relies on [Pydantic](https://docs.pydantic.dev/). The types that can actually be used are defined in form
of the type-alias [`PydanticSerializable`](src/intelligence_layer/core/tracer.py#L44).

The second parameter `task_span` is used for [tracing](#Trace) which is described below.

`do_run` is the method that needs to be implemented for a concrete Task. The external interface of a
Task is its `run` method:

```Python
class Task(ABC, Generic[Input, Output]):
    @final
    def run(self, input: Input, tracer: Tracer, trace_id: Optional[str] = None) -> Output:
      ...
```

Its signature differs only in the parameters regarding [tracing](#Trace).

### Levels of abstraction

Even though the concept is so generic the main purpose for a Task is of course to make use of an LLM for the
transformation. Tasks are defined at different levels of abstraction. There are higher level Tasks (also called Use Cases)
that reflect a typical user problem and there are lower level Tasks that are more about interfacing
with an LLM on a very generic or even technical level.

Examples for higher level tasks (Use Cases) are:

- Answering a question based on a gievn document: `QA: (Document, Question) -> Answer`
- Generate a summary of a given document: `Summary: Document -> Summary`

Examples for lower level tasks are:

- Let the model generate text based on an instruacton and some context: `Instruct: (Context, Instruction) -> Completion`
- Chunk a text in smaller pieces at optimized boundaries (typically to make it fit into an LLM's context-size): `Chunk: Text -> [Chunk]`

### Composability

Tasks compose. Typically you would build higher level tasks from lower level tasks. Given a task you can draw a dependency graph
that illustrates which sub-tasks it is using and in turn which sub-tasks they are using. This graph typically forms a hierarchy or
more general a directed acyclic graph. The following drawing shows this graph for the Intelligence Layer's `RecursiveSummarize`
Task:

<img src="./assets/RecursiveSummary.drawio.svg">


### Trace

A Task implements a workflow. It processes its input, passes it on to sub-tasks, processes the outputs of sub-tasks
to build its own output. This workflow can be represented in a trace. For this a Task's `run` method takes a `Tracer`
that takes care of storing details on the steps of this workflow like the tasks that have been invoked along with their
input and output and timing information. For this the tracing defines the following concepts:

- A `Tracer` is passed to a Task's `run` method and provides methods for opening `Span`s or `TaskSpan`s.
- A `Span` is a `Tracer` and allows for grouping multiple logs and duration together as a single, logical step in the
  workflow.
- A `TaskSpan` is a `Span` and allows for grouping multiple logs together, as well as the task's specific input, output.
  An opened `TaskSpan` is passed to `Task.do_run`. Since a `TaskSpan` is a `Tracer` a `do_run` implementation can pass
  this instance on to `run` methods of sub-tasks.

The following diagram illustrates their relationship:

<img src="./assets/Tracer.drawio.svg">

Each of these concepts is implemented in form of an abstract base class and the Intelligence Layer provides
several concrete implementations that store the actual traces in different backends. For each backend each of the
three abstract classes `Tracer`, `Span` and `TaskSpan` needs to be implemented. Here only the top-level
`Tracer`-implementations are listed:

- The `NoOpTracer` can be used when tracing information shall not be stored at all.
- The `InMemoryTracer` stores all traces in an in memory data structure and is most helpful in tests or
  Jupyter notebooks.
- The `FileTracer` stores all traces in a jsonl-file.
- The `OpenTelemetryTracer` uses an OpenTelemetry
  [`Tracer`](https://opentelemetry-python.readthedocs.io/en/latest/api/trace.html#opentelemetry.trace.Tracer)
  to store the traces in an OpenTelemetry backend.


## Evaluation

### Dataset

- List of examples (`Input`)

### Run

- Compute `Output`s for Dataset

### Evaluate

- Evaluate a single run to create an results that can be compared
- Compare multiple runs with a single evaluation (e.g. ELO)

### Aggregate

- Aggregate results from a single evaluation
- Aggregate results from multiple compare-evaluations to complete comparison

### Data Storage

- DatasetRepository
- RunRepository
- EvaluationRepository
- AggregationRepository


explainability:
- debug loglevel explain (full prompt vs focus (RAG)) (prompt whisper)
- eval: unexpected result: explain for input (aggregate)
  - run explain only on "failed"

Run:
- scheduled
