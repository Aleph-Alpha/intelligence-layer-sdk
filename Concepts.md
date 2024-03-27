# Concepts

The main focus of the Intelligence Layer is to enable developers to

- implement their LLM use cases by building upon existing and composing existing functionality and providing insights into
  the runtime behavior of these
- iteratively improve their implementations or compare them to existing implementations by evaluating them against
  a given set of example

Both focus points are described in more detail in the following sections.

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
of the type-alias [`PydanticSerializable`](src/intelligence_layer/core/tracer/tracer.py#L44).

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
input and output and timing information. The following illustration shows the trace of an MultiChunkQa-task:

<img src="./assets/Tracing.drawio.svg">

To represent this tracing defines the following concepts:

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

To support these techniques the Intelligence Layer differentiates between 3 consecutive steps:

1. Run a task by feeding it all inputs of a dataset and collecting all outputs
2. Evaluate the outputs of one or several
  runs and produce an evaluation result for each example. Typically a single run is evaluated if absolute
  metrics can be computed and several runs are evaluated when the outputs of runs shall be compared.
3. Aggregate the evaluation results of one or several evaluation runs into a single object containing the aggregated
  metrics. Aggregating over several evaluation runs supports amending a previous comparison result with
  comparisons of new runs without the need to re-execute the previous comparisons again.

The following table shows how these three steps are represented in code:

| Step    | Executor | Custom Logic | Repository    |
|---------|----------|--------------|---------------|
| 1. Run  | `Runner` | `Task`       | `RunRepository` |
| 2. Evaluate | `Evaluator` | `EvaluationLogic` | `EvaluationRepository` |
| 3. Aggregate | `Aggregator` | `AggregationLogic` | `AggregationRepository` |

The column
- Executor lists concrete implementations provided by the Intelligence Layer.
- Custom Logic lists abstract classes that need to be implemented with the custom logic.
- Repository lists abstract classes for storing intermediate results. The Intelligence Layer provides
  different implementations for these. See the next section for details.

### Data Storage

During an evaluation process a lot of intermediate data is created before the final aggregated result can be produced.
To avoid that expensive computations have to be repeated if new results should be produced based on previous ones
all intermediate results are persisted. For this the different executor-classes make use of repositories.

There are the following Repositories:

- The `DatasetRepository` offers methods to manage datasets. The `Runner` uses it to read all examples of a dataset to feed
  then to the `Task`.
- The `RunRepository` is responsible for storing a task's output (in form of a `ExampleOutput`) for each example of a dataset
  which are created when a `Runner`
  runs a task using this dataset. At the end of a run a `RunOverview` is stored containing some metadata concerning the run.
  The `Evaluator` reads these outputs given a list of runs it should evaluate to create an evaluation
  result for each example of the dataset.
- The `EvaluationRepository` enables the `Evaluator` to store the individual evaluation result (in form of an `ExampleEvaluation`)
  for each example and an `EvaluationOverview`
  and makes them available to the `Aggregator`.
- The `AggregationRepository` stores the `AggregationOverview` containing the aggregated metrics on request of the `Aggregator`.

The following diagrams illustrate how the different concepts play together in case of the different types of evaluations.

<figure>
<img src="./assets/AbsoluteEvaluation.drawio.svg">
<figcaption>Process of an absolute Evaluation</figcaption>
</figure>

1. The `Runner` reads the `Example`s of a dataset from the `DatasetRepository` and runs a `Task` for each `Example.input` to produce `Output`s.
2. Each `Output` is wrapped in an `ExampleOutput` and stored in the `RunRepository`.
3. The `Evaluator` reads the `ExampleOutput`s for a given run from the
   `RunRepository` and the corresponding `Example` from the `DatasetRepository` and uses the `EvaluationLogic` to compute an `Evaluation`.
4. Each `Evaluation` gets wrapped in an `ExampleEvaluation` and stored in the `EvaluationRepository`.
5. The `Aggregator` reads all `ExampleEvaluation`s for a given evaluation and feeds them to the `AggregationLogic` to produce a `AggregatedEvaluation`.
6. The `AggregatedEvalution` is wrapped in an `AggregationOverview` and stoed in the `AggregationRepository`.

The next diagram illustrates the more complex case of a relative evaluation.

<figure>
<img src="./assets/RelativeEvaluation.drawio.svg">
<figcaption>Process of a relative Evaluation</figcaption>
</figure>

1. Multiple `Runner`s read the same dataset and produce for different `Task`s corresponding `Output`s.
2. For each run all `Output`s are stored in the `RunRepository`.
3. The `Evaluator` gets as input previous evaluations (that were produced on basis of the same dataset, but different `Task`s) and the new runs of the previous step.
4. Given the previous evaluations and the new runs the `Evaluator` can read the `ExampleOutput`s of both the new runs
   and the runs associated to previous evaluations, collect all that belong to a single `Example` and pass them
   along with the `Example` to the `EvaluationLogic` to compute an `Evaluation`.
5. Each `Evaluation` gets wrapped in an `ExampleEvaluation` and is stored in the `EvaluationRepository`.
6. The `Aggregator` reads all `ExampleEvaluation` from all involved evaluations
   and feeds them to the `AggregationLogic` to produce a `AggregatedEvaluation`.
7. The `AggregatedEvalution` is wrapped in an `AggregationOverview` and stoed in the `AggregationRepository`.
