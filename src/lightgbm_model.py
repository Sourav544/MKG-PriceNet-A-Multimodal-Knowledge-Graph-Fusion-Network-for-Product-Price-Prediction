# Phase 9: LightGBM Training Loop
# ==========================================
def train_lgb(X, y, X_test, n_splits=5, num_boost_round=15000, early_stop_rounds=1000):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\nLGB Fold {fold}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": RANDOM_STATE,
            "verbosity": -1
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stop_rounds, verbose=False),
            lgb.log_evaluation(period=200) 
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            callbacks=callbacks
        )

        best_iter = model.best_iteration if hasattr(model, "best_iteration") else num_boost_round
        
        val_pred = model.predict(X_val, num_iteration=best_iter)
        oof[val_idx] = val_pred
        test_preds += model.predict(X_test, num_iteration=best_iter) / n_splits

        model.save_model(str(MODEL_DIR / f"lgb_fold{fold}.txt"))
        del model, train_data, val_data
        gc.collect()

    return oof, test_preds

lgb_oof, lgb_test_preds = train_lgb(X_train, y_train, X_test)
print("LGB OOF SMAPE:", smape(np.expm1(y_train), np.expm1(lgb_oof)))

