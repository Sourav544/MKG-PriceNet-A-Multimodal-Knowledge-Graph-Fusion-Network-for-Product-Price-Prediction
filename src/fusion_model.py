class DynamicFusionModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden=1024,
        dropout=0.2
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden),

            nn.LayerNorm(hidden),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden, hidden//2),

            nn.LayerNorm(hidden//2),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden//2,1)
        )

    def forward(self,x):

        return self.net(x).squeeze(-1)

def train_fusion_nn_fast(
        X,
        y,
        X_test,
        n_splits=5,
        epochs=40,
        bs=128,
        lr=3e-4,
        weight_decay=1e-3
):

    X = X.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y = y.astype(np.float32)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    X_test_tensor = torch.from_numpy(X_test)

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    oof = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    for fold,(tr_idx,val_idx) in enumerate(kf.split(X),1):

        print(f"\n=== Fold {fold} ===")

        train_ds = TensorDataset(
            X_tensor[tr_idx],
            y_tensor[tr_idx]
        )

        val_ds = TensorDataset(
            X_tensor[val_idx],
            y_tensor[val_idx]
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=bs*2
        )

        model = DynamicFusionModel(
            input_dim=X.shape[1]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )

        criterion = nn.SmoothL1Loss()

        best_smape = 999999

        best_path = f"dynamic_fold_{fold}.pth"

        for epoch in range(epochs):

            model.train()

            train_losses = []

            for xb,yb in train_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                preds = model(xb)

                loss = criterion(preds,yb)

                loss.backward()

                optimizer.step()

                scheduler.step()

                train_losses.append(
                    loss.item()
                )

            model.eval()

            val_preds = []

            with torch.no_grad():

                for xb,yb in val_loader:

                    xb = xb.to(device)

                    val_preds.append(
                        model(xb).cpu().numpy()
                    )

            val_preds = np.concatenate(val_preds)

            fold_smape = smape(
                np.expm1(y[val_idx]),
                np.expm1(val_preds)
            )

            print(
                f"Epoch {epoch+1} "
                f"SMAPE={fold_smape:.4f}"
            )

            if fold_smape < best_smape:

                best_smape = fold_smape

                torch.save(
                    model.state_dict(),
                    best_path
                )

        model.load_state_dict(
            torch.load(best_path)
        )

        model.eval()

        val_preds = []

        with torch.no_grad():

            for xb,yb in val_loader:

                xb = xb.to(device)

                val_preds.append(
                    model(xb).cpu().numpy()
                )

        oof[val_idx] = np.concatenate(
            val_preds
        )

        del model

        gc.collect()

        torch.cuda.empty_cache()

    return oof,test_preds

nn_oof, nn_test_preds = train_fusion_nn_fast(X_train, y_train, X_test)
print("NN OOF SMAPE:", smape(np.expm1(y_train), np.expm1(nn_oof)))