# 셀 6: Hyperparameter Tuning (Optuna)
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train_selected, y_train)

    y_prob = model.predict_proba(X_test_selected)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    y_pred = model.predict(X_test_selected)
    recall = recall_score(y_test, y_pred)

    harmonic_mean = 2 / (1/auc + 1/recall)
    return harmonic_mean

# SQLite 데이터베이스를 사용하여 Optuna study 생성
study = optuna.create_study(
    study_name="bank_marketing_optimization",
    storage="sqlite:///bank_marketing_optuna.db",
    load_if_exists=True,
    direction='maximize'
)
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
