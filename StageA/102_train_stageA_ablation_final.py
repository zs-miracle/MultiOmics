#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from full_gnn_transformer_utils_ablation_final import GRL, FullGNNTransformerEncoder, set_seed


class StageAAblationModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_pathways: int,
        num_classes: int,
        d_model: int,
        num_heads: int,
        num_gnn_layers: int,
        num_transformer_layers: int,
        dropout: float,
        ablation_mode: str,
    ):
        super().__init__()

        use_gnn = ablation_mode in {'full', 'gnn_only', 'no_graph'}
        use_transformer = ablation_mode in {'full', 'transformer_only', 'no_graph'}

        self.ablation_mode = ablation_mode
        self.encoder = FullGNNTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            use_gnn=use_gnn,
            use_transformer=use_transformer,
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.pathway_head = nn.Linear(d_model, num_pathways)
        self.cancer_head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        grl_lambda: float = 0.5,
    ):
        if self.ablation_mode == 'no_graph':
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            edge_weight = torch.zeros((0,), dtype=torch.float32, device=x.device)

        tok, pooled = self.encoder(x, edge_index, edge_weight)
        recon = self.decoder(tok).squeeze(-1)
        pathway = self.pathway_head(pooled)
        cancer = self.cancer_head(GRL.apply(pooled, grl_lambda))
        return recon, pathway, cancer, pooled


def build_modality_graph(features: list[str], workdir: Path, top_k: int = 8):
    membership = pd.read_csv(workdir / 'feature_pathway_membership.csv')
    membership = membership[membership['feature'].astype(str).isin(features)].copy()

    idx = {f: i for i, f in enumerate(features)}
    groups = {
        str(r['feature']): set(str(r['pathways']).split(';')) - {''}
        for _, r in membership.iterrows()
    }

    edge_rows = []
    feats = list(groups.keys())
    for a in feats:
        neigh = []
        for b in feats:
            if a == b:
                continue
            inter = len(groups[a] & groups[b])
            if inter > 0 and a in idx and b in idx:
                neigh.append((b, inter))
        neigh = sorted(neigh, key=lambda x: x[1], reverse=True)[:top_k]
        for b, inter in neigh:
            edge_rows.append((idx[a], idx[b], float(inter)))

    if not edge_rows:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

    edge_index = torch.tensor(
        [[x[0] for x in edge_rows], [x[1] for x in edge_rows]],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([x[2] for x in edge_rows], dtype=torch.float32)
    return edge_index, edge_weight


def stratified_split_indices(labels: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        if len(cls_idx) == 1:
            train_idx.extend(cls_idx.tolist())
            continue

        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        if n_val >= len(cls_idx):
            n_val = len(cls_idx) - 1

        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError('分层切分失败：训练集或验证集为空，请检查样本量和类别分布。')

    return train_idx, val_idx


def batch_feature_pearson(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    cors = []
    for i in range(pred.shape[0]):
        x = pred[i]
        y = target[i]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        c = np.corrcoef(x, y)[0, 1]
        if np.isfinite(c):
            cors.append(float(c))
    return float(np.mean(cors)) if cors else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description='StageA ablation for GNN + Transformer')

    parser.add_argument('--workdir', type=str, required=True)
    parser.add_argument('--modality', choices=['rna', 'protein', 'metabolomics'], required=True)
    parser.add_argument('--ablation-mode', choices=['full', 'transformer_only', 'gnn_only', 'no_graph'], required=True)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=0.2)

    parser.add_argument('--max-features', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--top-k-graph', type=int, default=8)

    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-gnn-layers', type=int, default=2)
    parser.add_argument('--num-transformer-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--grl-lambda', type=float, default=0.5)

    parser.add_argument('--mask-rate', type=float, default=0.15)
    parser.add_argument('--noise-std', type=float, default=0.05)
    parser.add_argument('--disable-noise', action='store_true')

    parser.add_argument('--sample-id-col', type=str, default='sample_id')
    parser.add_argument('--label-col', type=str, default='cancer_type')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError('--val-ratio 必须在 0 和 1 之间。')

    set_seed(args.seed)
    device = torch.device('cpu' if args.force_cpu or not torch.cuda.is_available() else 'cuda')
    print(f'[Init] ablation_mode={args.ablation_mode}, device={device}')

    root = Path(args.workdir).resolve() / 'data_processed' / 'stageA' / args.modality

    X = pd.read_csv(root / f'{args.modality}_pretrain_matrix.csv', index_col=0)
    X.index = X.index.astype(str)

    Yp_raw = pd.read_csv(root / 'pathway_target_matrix.csv', index_col=0)
    Yp_raw.index = Yp_raw.index.astype(str)
    missing_in_yp = X.index[~X.index.isin(Yp_raw.index)]
    if len(missing_in_yp) > 0:
        raise ValueError(f'pathway_target_matrix.csv 缺少样本，前 10 个: {missing_in_yp[:10].tolist()}')
    Yp = Yp_raw.loc[X.index]

    sm = pd.read_csv(root / f'{args.modality}_sample_manifest.csv')
    if args.sample_id_col not in sm.columns:
        raise ValueError(f'sample_manifest.csv 缺少样本 ID 列: {args.sample_id_col}')
    if args.label_col not in sm.columns:
        raise ValueError(f'sample_manifest.csv 缺少标签列: {args.label_col}')

    sm[args.sample_id_col] = sm[args.sample_id_col].astype(str)
    sm = sm.set_index(args.sample_id_col)
    if sm.index.duplicated().any():
        dup_ids = sm.index[sm.index.duplicated()].unique().tolist()
        raise ValueError(f'sample_manifest.csv 中样本 ID 有重复，前 10 个: {dup_ids[:10]}')

    missing_in_sm = X.index[~X.index.isin(sm.index)]
    if len(missing_in_sm) > 0:
        raise ValueError(f'sample_manifest.csv 缺少样本，前 10 个: {missing_in_sm[:10].tolist()}')

    sm = sm.loc[X.index]
    labels = sm[args.label_col].astype(str).fillna('unknown')
    classes = sorted(labels.unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = labels.map(label_to_idx).to_numpy(dtype=np.int64)

    tr, va = stratified_split_indices(y, val_ratio=args.val_ratio, seed=args.seed)

    if X.shape[1] > args.max_features:
        train_var = X.iloc[tr].var(axis=0)
        keep = train_var.sort_values(ascending=False).head(args.max_features).index.tolist()
        X = X[keep]

    x_np = X.to_numpy(dtype=np.float32)
    yp_np = Yp.to_numpy(dtype=np.float32)

    edge_index, edge_weight = build_modality_graph(list(X.columns.astype(str)), root, top_k=args.top_k_graph)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # 保持和 101 一样：数据集留在 CPU，batch 再搬到 GPU
    x_all = torch.tensor(x_np, dtype=torch.float32)
    yp_all = torch.tensor(yp_np, dtype=torch.float32)
    yt_all = torch.tensor(y, dtype=torch.long)

    tr_t = torch.tensor(tr, dtype=torch.long)
    va_t = torch.tensor(va, dtype=torch.long)

    train_dataset = TensorDataset(x_all[tr_t], yp_all[tr_t], yt_all[tr_t])
    val_dataset = TensorDataset(x_all[va_t], yp_all[va_t], yt_all[va_t])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    model = StageAAblationModel(
        num_features=X.shape[1],
        num_pathways=Yp.shape[1],
        num_classes=len(classes),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        dropout=args.dropout,
        ablation_mode=args.ablation_mode,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    run_name = (
        f'{args.ablation_mode}_bs{args.batch_size}_feat{X.shape[1]}_topk{args.top_k_graph}'
        f'_dm{args.d_model}_g{args.num_gnn_layers}_t{args.num_transformer_layers}_seed{args.seed}'
    )
    out = root / 'ablation_outputs_stageA' / run_name
    out.mkdir(parents=True, exist_ok=True)

    history = []
    best = None
    best_epoch = 0
    best_loss = float('inf')
    wait = 0
    stopped_early = False
    best_record: dict[str, float] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, ypb, ytb in train_loader:
            xb = xb.to(device, non_blocking=True)
            ypb = ypb.to(device, non_blocking=True)
            ytb = ytb.to(device, non_blocking=True)

            opt.zero_grad()

            if args.disable_noise:
                model_input = xb
            else:
                mask = (torch.rand_like(xb) > args.mask_rate).float()
                model_input = xb * mask + args.noise_std * torch.randn_like(xb)

            recon, path_pred, cancer_pred, _ = model(
                model_input,
                edge_index,
                edge_weight,
                grl_lambda=args.grl_lambda,
            )

            recon_loss = F.mse_loss(recon, xb)
            path_loss = F.mse_loss(path_pred, ypb)
            adv_loss = F.cross_entropy(cancer_pred, ytb)
            loss = recon_loss + 0.2 * path_loss + 0.1 * adv_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            model.eval()

            val_loss_sum = 0.0
            recon_mse_sum = 0.0
            path_mse_sum = 0.0
            adv_ce_sum = 0.0
            cancer_correct = 0.0
            total_count = 0

            recon_pearson_weighted = 0.0
            pathway_pearson_weighted = 0.0

            for xb, ypb, ytb in val_loader:
                xb = xb.to(device, non_blocking=True)
                ypb = ypb.to(device, non_blocking=True)
                ytb = ytb.to(device, non_blocking=True)

                recon_v, path_v, cancer_v, _ = model(
                    xb,
                    edge_index,
                    edge_weight,
                    grl_lambda=args.grl_lambda,
                )

                recon_mse = F.mse_loss(recon_v, xb)
                path_mse = F.mse_loss(path_v, ypb)
                adv_ce = F.cross_entropy(cancer_v, ytb)
                val = recon_mse + 0.2 * path_mse + 0.1 * adv_ce

                bs = xb.size(0)
                total_count += bs

                val_loss_sum += float(val.detach().cpu()) * bs
                recon_mse_sum += float(recon_mse.detach().cpu()) * bs
                path_mse_sum += float(path_mse.detach().cpu()) * bs
                adv_ce_sum += float(adv_ce.detach().cpu()) * bs
                cancer_correct += float((cancer_v.argmax(dim=1) == ytb).sum().detach().cpu())

                recon_p = batch_feature_pearson(recon_v, xb)
                path_p = batch_feature_pearson(path_v, ypb)
                recon_pearson_weighted += recon_p * bs
                pathway_pearson_weighted += path_p * bs

            record = {
                'epoch': epoch,
                'train_total_loss': float(np.mean(train_losses)),
                'val_total_loss': val_loss_sum / total_count,
                'val_recon_mse': recon_mse_sum / total_count,
                'val_path_mse': path_mse_sum / total_count,
                'val_adv_ce': adv_ce_sum / total_count,
                'val_cancer_acc': cancer_correct / total_count,
                'val_recon_feature_pearson': recon_pearson_weighted / total_count,
                'val_pathway_pearson': pathway_pearson_weighted / total_count,
            }

        history.append(record)

        if record['val_total_loss'] < best_loss - args.early_stopping_min_delta:
            best_loss = record['val_total_loss']
            best_epoch = epoch
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_record = record.copy()
            wait = 0
        else:
            wait += 1

        print(
            f"mode={args.ablation_mode} epoch={epoch} "
            f"train_loss={record['train_total_loss']:.6f} "
            f"val_loss={record['val_total_loss']:.6f} "
            f"best_val={best_loss:.6f} wait={wait}/{args.early_stopping_patience}"
        )

        if wait >= args.early_stopping_patience:
            stopped_early = True
            print(f'Early stopping at epoch {epoch}. Best epoch = {best_epoch}, best val loss = {best_loss:.6f}')
            break

    if best is None:
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_record = history[-1].copy()
        best_epoch = history[-1]['epoch']
        best_loss = history[-1]['val_total_loss']

    torch.save(best, out / 'best_model.pt')
    pd.DataFrame(history).to_csv(out / 'training_curve.csv', index=False)

    metrics = {
        'ablation_mode': args.ablation_mode,
        'modality': args.modality,
        'samples': int(X.shape[0]),
        'features_used': int(X.shape[1]),
        'pathways': int(Yp.shape[1]),
        'edge_count': int(edge_index.shape[1]),
        'best_epoch': int(best_epoch),
        'stopped_early': bool(stopped_early),
        'batch_size': int(args.batch_size),
        'top_k_graph': int(args.top_k_graph),
        'd_model': int(args.d_model),
        'num_heads': int(args.num_heads),
        'num_gnn_layers': int(args.num_gnn_layers),
        'num_transformer_layers': int(args.num_transformer_layers),
        'dropout': float(args.dropout),
        'mask_rate': float(args.mask_rate),
        'noise_std': float(args.noise_std),
        'noise_disabled': bool(args.disable_noise),
        'device': str(device),
        'num_classes': int(len(classes)),
        **{k: v for k, v in best_record.items() if k != 'epoch'},
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
