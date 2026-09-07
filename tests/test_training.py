from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from customer_churn.models.training import make_oversampled_rf, make_oversampled_xgb


def test_make_oversampled_rf_pipeline_structure():
    pipeline = make_oversampled_rf(random_state=42)

    assert isinstance(pipeline, ImbPipeline)
    assert [name for name, _ in pipeline.steps] == ["oversample", "clf"]
    assert isinstance(pipeline.named_steps["oversample"], RandomOverSampler)
    assert isinstance(pipeline.named_steps["clf"], RandomForestClassifier)


def test_make_oversampled_rf_forwards_params_to_classifier_only():
    pipeline = make_oversampled_rf(random_state=42, n_estimators=250, max_depth=10)

    clf = pipeline.named_steps["clf"]
    assert clf.get_params()["n_estimators"] == 250
    assert clf.get_params()["max_depth"] == 10


def test_make_oversampled_xgb_pipeline_structure():
    pipeline = make_oversampled_xgb(random_state=42, eval_metric="logloss")

    assert isinstance(pipeline, ImbPipeline)
    assert [name for name, _ in pipeline.steps] == ["oversample", "clf"]
    assert isinstance(pipeline.named_steps["oversample"], RandomOverSampler)
    assert isinstance(pipeline.named_steps["clf"], XGBClassifier)


def test_make_oversampled_xgb_forwards_params_to_classifier_only():
    pipeline = make_oversampled_xgb(random_state=42, n_estimators=350, max_depth=5)

    clf = pipeline.named_steps["clf"]
    assert clf.get_params()["n_estimators"] == 350
    assert clf.get_params()["max_depth"] == 5
