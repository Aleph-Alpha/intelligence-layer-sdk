class TestTask(Task[str, str]):
    def run(self, input: str, logger: DebugLogger) -> str:
        return "output"


@fixture
def log_based_debug_log(tmp_path: Path) -> LogBasedDebugLogger:
    return LogBasedDebugLogger(tmp_path / "log.log")


def test_log_based_debug_logger(log_based_debug_log: LogBasedDebugLogger) -> None:
    input = "input"
    output = TestTask().run(input, log_based_debug_log)
    log_tree = parse_log(log_based_debug_log.log_file_path)
    assert log_tree == TaskSpan(
        name="", start_timestamp=None, end_timestamp=None, input=input, output=output
    )


class StartTask(BaseModel):
    uuid: UUID
    name: str
    start: datetime
    input: Any


class EndTask(BaseModel):
    end: datetime
    output: Any


class LogLineMetadata(BaseModel):
    entry_type: str
    entry: Mapping[str, Any]
    parent: Optional[UUID] = None


def parse_log(log_path: Path) -> InMemoryDebugLogger:
    tree_builder = TreeBuilder()
    with log_path.open("r") as f:
        for line in f:
            json_line = loads(line)
            line_metadata = LogLineMetadata.model_validate(json_line)
            if line_metadata.entry_type == type(StartTask).__name__:
                tree_builder.start_task(line_metadata)


class Span(BaseModel):
    name: str
    start_timestamp: datetime
    end_timestamp: Optional[datetime] = None


class TaskSpan(Span):
    input: SerializeAsAny[Any]
    output: Optional[SerializeAsAny[Any]] = None


class TreeBuilder:
    tasks: dict[UUID, InMemoryTaskSpan]

    def start_task(self, log_line: LogLineMetadata) -> None:
        start_task = StartTask.model_validate(log_line.entry)
        self.tasks[start_task.uuid] = TaskSpan(
            name=start_task.name,
            start_timestamp=start_task.start,
            input=start_task.input,
        )
