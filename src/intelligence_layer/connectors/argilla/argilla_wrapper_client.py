import itertools
import logging
import os
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Optional,
)

import argilla as rg  # type: ignore

#from argilla.client.feedback.schemas.types import (  # type: ignore
#    AllowedFieldTypes,
#    AllowedQuestionTypes,
#)

from intelligence_layer.connectors.argilla.argilla_client import (
    ArgillaClient,
    ArgillaEvaluation,
    Record,
    RecordData,
)


class ArgillaWrapperClient(ArgillaClient):
    pass
