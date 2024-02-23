# Concepts

## Task

At the heart of the Intelligence Layer is a `Task`. A task is actually a pretty generic concept that just
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

`do_run` is the method that needs to be implemented for a concrete task. The external interface of a
task is its `run` method:

```Python
class Task(ABC, Generic[Input, Output]):
    @final
    def run(self, input: Input, tracer: Tracer, trace_id: Optional[str] = None) -> Output:
      ...
```

Its signature differs only in the parameters regarding [tracing](#Trace).

### Levels of abstraction

Even though the concept is so generic the main purpose for a task is of course to make use of an LLM for the
transformation. Tasks are defined at different levels of abstraction. There are higher level tasks (also called Use Cases)
that reflect a typical user problem and there are lower level tasks that are more about interfacing
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
task:

<img src="./assets/RecursiveSummary.drawio.svg">


### Trace

A task implements a workflow. It processes its input, passes it on to sub-tasks, processes the outputs of sub-tasks
to build its own output. This workflow can be represented in a trace. For this a task's `run` method takes a `Tracer`
that takes care of storing details on the steps of this workflow like the tasks that have been invoked along with their
input and output and timing information. For this the tracing defines the following concepts:

- A `Tracer` is passed to a task's `run` method and provides methods for opening `Span`s or `TaskSpan`s.
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

An important part of the Intelligence Layer is tooling that helps to evaluate custom tasks. Evaluation helps
to measure how well the implementation of a task performs given real world examples. The outcome of an entire
evaluation process is an aggregated evaluation result that consists out of metrics aggregated over all examples.

The evaluation process helps to:

- optimize a task's implementation by comparing and verifying if changes improve the performance.
- compare the performance of one implementation of a task with that of other (already existing) implementations.
- compare the performance of models for a given task implementation.
- verify how changes to the environment (new model version, new finetuning version) affect the
  performance of a task.


### Dataset

The basis of an evaluation is a set of examples for the specific task-type to be evaluated. A single example
consists out of :

- an instance of the `Input` for the specific task and
- optionally an _expected output_ that can be anything that makes sense in context of the specific evaluation (e.g.
  in case of classification this could contain the correct classification result, in case of QA this could contain
  a _golden answer_, but if an evaluation is only about comparing results with other results of other runs this
  could also be empty)

To enable reproducibility of evaluations datasets are immutable. A single dataset can be used to evaluate all
tasks of the same type, i.e. with the same `Input` and `Output` types.

### Evaluation Process

The Intelligence Layer supports different kinds of evaluation techniques. Most important are:

- Computing absolute metrics for a task where the aggregated result can be compared with results of previous
  result in a way that they can be ordered. Text classification could be a typical use case for this. In that
  case the aggregated result could contain metrics like accuracy which can easily compared with other
  aggregated results.
- Comparing the individual outputs of different runs (all based on the same dataset)
  in a single evaluation process and produce as aggregated result a
  ranking of all runs. This technique is useful when it is hard to come up with an absolute metrics to evaluate
  a single output, but it is easier to compare two different outputs and decide which one is better. An example
  use case could be summarization.

To support these techniques the Intelligence Layer differantiates between 3 consecutive steps:

- Run a task by feeding it all inputs of a dataset and collecting all outputs
- Evaluate the outputs of one or several
  runs and produce an evaluation result for each example. Typically a single run is evaluated if absolute
  metrics can be computed and several runs are evaluated when the outputs of runs shall be compared.
- Aggregate the evaluation results of one or several evaluation runs into a single object containing the aggregated
  metrics. Aggregating over several evaluation runs supports amending a previous comparison result with
  comparisons of new runs without the need to re-execute the previous comparisons again.


### Data Storage

- DatasetRepository
- RunRepository
- EvaluationRepository
- AggregationRepository
