import numpy as np
import pytest

from PySVM.datasets import load_iris
from PySVM.model_selection import (
    LearningCurveDisplay,
    ValidationCurveDisplay,
    learning_curve,
    validation_curve,
)
#from PySVM.tree import DecisionTreeClassifier
from PySVM.utils import shuffle
from PySVM.utils._testing import assert_allclose, assert_array_equal


@pytest.fixture
def data():
    return shuffle(*load_iris(return_X_y=True), random_state=0)


@pytest.mark.parametrize(
    "params, err_type, err_msg",
    [
        ({"std_display_style": "invalid"}, ValueError, "Unknown std_display_style:"),
        ({"score_type": "invalid"}, ValueError, "Unknown score_type:"),
    ],
)
@pytest.mark.parametrize(
    "CurveDisplay, specific_params",
    [
        (ValidationCurveDisplay, {"param_name": "max_depth", "param_range": [1, 3, 5]}),
        (LearningCurveDisplay, {"train_sizes": [0.3, 0.6, 0.9]}),
    ],
)
