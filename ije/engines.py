"""Framework adapters using the same inner holdout and macro-F1 objective.

    search: fit inner-training examples and select using inner validation;
    refit: selected classifier on all outer-training examples, with the same
           frozen inner-training feature map. Outer validation stays untouched.
"""
from pathlib import Path
from time import perf_counter
import random
import numpy as np


def build_optuna(trial, seed, threads):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import LinearSVC
    name = trial.suggest_categorical("classifier", ["logreg_l1", "logreg_l2", "linearsvc",
                                                   "sgd", "random_forest", "extra_trees", "hist_gbm"])
    if name in ("logreg_l1", "logreg_l2"):
        penalty = "l1" if name.endswith("l1") else "l2"
        return LogisticRegression(penalty=penalty, C=trial.suggest_float("lr_C", 1e-3, 1e2, log=True),
                                  solver="saga" if penalty == "l1" else "lbfgs",
                                  max_iter=2000, random_state=seed)
    if name == "linearsvc":
        return CalibratedClassifierCV(LinearSVC(C=trial.suggest_float("svc_C", 1e-3, 1e2, log=True),
                                                max_iter=3000, random_state=seed), cv=3, n_jobs=1)
    if name == "sgd":
        # Hinge needs calibration to provide the same probability interface to LIME.
        loss = trial.suggest_categorical("sgd_loss", ["hinge", "modified_huber", "log_loss"])
        model = SGDClassifier(loss=loss, alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
                              max_iter=1000, random_state=seed)
        return CalibratedClassifierCV(model, cv=3, n_jobs=1) if loss == "hinge" else model
    if name in ("random_forest", "extra_trees"):
        cls = RandomForestClassifier if name == "random_forest" else ExtraTreesClassifier
        return cls(n_estimators=trial.suggest_int("n_estimators", 50, 300),
                   max_depth=trial.suggest_int("max_depth", 5, 30),
                   min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                   n_jobs=threads, random_state=seed)
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 1e-3, .3, log=True),
        max_iter=trial.suggest_int("max_iter", 50, 300),
        max_depth=trial.suggest_int("max_depth", 3, 10), random_state=seed)


def frame(matrix, labels=None):
    import pandas as pd
    df = pd.DataFrame(matrix, columns=[f"f{i}" for i in range(matrix.shape[1])])
    if labels is not None:
        df["sentiment"] = np.asarray(labels, dtype=int)
    return df


class ModelHandle:
    def __init__(self, kind, model=None, directory=None, name=None):
        self.kind, self.model, self.directory, self.name = kind, model, directory, name
        self._predictor = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_predictor"] = None
        return state

    def predictor(self, artifact_dir):
        if self._predictor is None:
            from autogluon.tabular import TabularPredictor
            self._predictor = TabularPredictor.load(str(Path(artifact_dir) / self.directory))
        return self._predictor

    def predict(self, matrix, artifact_dir):
        if self.kind == "autogluon":
            return self.predictor(artifact_dir).predict(frame(matrix), model=self.name).to_numpy(dtype=int)
        return np.asarray(self.model.predict(matrix), dtype=int)

    def proba(self, matrix, artifact_dir):
        if self.kind == "autogluon":
            values = self.predictor(artifact_dir).predict_proba(frame(matrix), model=self.name)
            result = values.loc[:, [0, 1, 2]].to_numpy()
        else:
            if not hasattr(self.model, "predict_proba"):
                raise RuntimeError("Selected estimator has no probability interface; no substitute model allowed")
            classes = list(self.model.classes_)
            result = self.model.predict_proba(matrix)[:, [classes.index(c) for c in (0, 1, 2)]]
        result = np.asarray(result, dtype=float)
        if result.shape != (len(matrix), 3) or not np.isfinite(result).all():
            raise ValueError("Invalid prediction probabilities")
        if (result < -1e-8).any() or not np.allclose(result.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Probability values fail validation")
        return result


def train_engine(framework, x_train, y_train, x_valid, y_valid, x_refit, y_refit,
                 config, seed, artifact_dir):
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from threadpoolctl import threadpool_limits
    random.seed(seed)
    np.random.seed(seed)
    threads = config["cpu_threads"]
    budget = config["search_budget_seconds"]
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    timing = {}
    metadata = {"framework": framework, "objective": "f1_macro", "seed": seed,
                "budget_seconds": budget, "budget_policy": config["budget_policy"]}
    with threadpool_limits(limits=threads):
        start = perf_counter()
        if framework == "flaml":
            from flaml import AutoML
            search = AutoML()
            search.fit(X_train=x_train, y_train=y_train, X_val=x_valid, y_val=y_valid,
                       task="classification", metric="macro_f1", eval_method="holdout",
                       estimator_list=["lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree", "lrl1", "lrl2"],
                       time_budget=budget, seed=seed, n_jobs=threads,
                       retrain_full=False, sample=False,
                       log_file_name=str(artifact_dir / "search.log"), verbose=1)
            timing["search_s"] = perf_counter() - start
            # Clone the actual selected estimator, preserving all selected parameters.
            estimator = clone(search.model.estimator)
            metadata.update({"selected_model": search.best_estimator,
                             "selected_parameters": search.best_config,
                             "inner_macro_f1": float(1 - search.best_loss)})
        elif framework == "optuna":
            import optuna
            def objective(trial):
                estimator = build_optuna(trial, seed, threads)
                estimator.fit(x_train, y_train)
                return float(f1_score(y_valid, estimator.predict(x_valid), labels=[0, 1, 2],
                                      average="macro", zero_division=0))
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=config["optuna_trials"], timeout=budget)
            timing["search_s"] = perf_counter() - start
            estimator = build_optuna(optuna.trial.FixedTrial(study.best_params), seed, threads)
            metadata.update({"selected_model": study.best_params["classifier"],
                             "selected_parameters": study.best_params,
                             "inner_macro_f1": float(study.best_value),
                             "completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)})
            from .storage import write_json
            write_json(artifact_dir / "trials.json", [{"number": t.number, "state": t.state.name,
                                                       "value": t.value, "params": t.params} for t in study.trials])
        elif framework == "autogluon":
            from autogluon.tabular import TabularPredictor
            predictor = TabularPredictor(label="sentiment", eval_metric="f1_macro", problem_type="multiclass",
                                         path=str(artifact_dir / "autogluon"), verbosity=2)
            predictor.fit(train_data=frame(x_train, y_train), tuning_data=frame(x_valid, y_valid),
                          time_limit=budget, presets="medium_quality", num_bag_folds=0, num_stack_levels=0,
                          dynamic_stacking=False, num_cpus=threads, num_gpus=0,
                          refit_full=False, set_best_to_refit_full=False)
            timing["search_s"] = perf_counter() - start
            best_name = predictor.model_best
            leaderboard = predictor.leaderboard(silent=True)
            score = float(leaderboard.set_index("model").loc[best_name, "score_val"])
            metadata.update({"selected_model": best_name, "inner_macro_f1": score,
                             "preset": "medium_quality", "bagging": False, "stacking": False,
                             "model_seed_policy": "AutoGluon_1.5_model_defaults_python_and_numpy_seeded_separately"})
            leaderboard.to_csv(artifact_dir / "leaderboard.csv", index=False)
            start = perf_counter()
            mapping = predictor.refit_full(model=best_name, set_best_to_refit_full=True,
                                           num_cpus=threads, num_gpus=0)
            timing["refit_s"] = perf_counter() - start
            if best_name not in mapping:
                raise RuntimeError("AutoGluon did not provide the requested refitted model")
            handle = ModelHandle("autogluon", directory="autogluon", name=mapping[best_name])
            # Cache for immediate outer scoring; persistence uses the relative directory.
            handle._predictor = predictor
        else:
            raise ValueError(framework)
        if framework != "autogluon":
            start = perf_counter()
            estimator.fit(x_refit, y_refit)
            timing["refit_s"] = perf_counter() - start
            handle = ModelHandle("sklearn", model=estimator)
    timing["search_overrun_s"] = max(0.0, timing["search_s"] - budget)
    return handle, metadata, timing
