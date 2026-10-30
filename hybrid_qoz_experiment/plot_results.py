"""
plot_results.py

Generate per-dataset Rate-PSNR and Rate-SSIM plots + CR comparison table,
matching the style and structure of the HPEZ paper (QoZ paper).

Usage:
    python plot_results.py [--results results/results.json]
                           [--config config.yaml]
                           [--out-dir plots/]
                           [--datasets rtm cesm ...]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── plot style ───────────────────────────────────────────────────────────────

ALG_STYLE = {
    'hybrid_qoz_gp': dict(label='Hybrid QoZ+GP', color='#02C39A', marker='o',   ls='-',  lw=2.0, ms=7),
    'qoz_hpez':      dict(label='HPEZ (QoZ -q4)',     color='#1A6B8A', marker='s',   ls='--', lw=1.6, ms=6),
    'sz3_interp':    dict(label='SZ3-Interp',          color='#F0A500', marker='^',   ls='--', lw=1.6, ms=6),
    'sz3_lorenzo':   dict(label='SZ3-Lorenzo+Interp',  color='#E05C5C', marker='v',   ls='--', lw=1.6, ms=6),
    # anchor_frac sweep family — green gradient, solid lines
    'hybrid_f005':   dict(label='Hybrid 5% anchors',   color='#B7E4C7', marker='o', ls='-',  lw=1.4, ms=5),
    'hybrid_f010':   dict(label='Hybrid 10% anchors',  color='#74C69D', marker='o', ls='-',  lw=1.4, ms=5),
    'hybrid_f025':   dict(label='Hybrid 25% anchors',  color='#40916C', marker='o', ls='-',  lw=1.6, ms=6),
    'hybrid_f050':   dict(label='Hybrid 50% anchors',  color='#1B4332', marker='o', ls='-',  lw=1.6, ms=6),
    'hybrid_f100':   dict(label='Hybrid 100% anchors (step=64)', color='#02C39A', marker='o', ls='-',  lw=2.0, ms=7),
    # max_step sweep family — blue gradient, dashed lines
    'hybrid_s016':   dict(label='Hybrid step=16 (100% anchors)', color='#AED6F1', marker='D', ls='-.', lw=1.4, ms=5),
    'hybrid_s032':   dict(label='Hybrid step=32 (100% anchors)', color='#2E86C1', marker='D', ls='-.', lw=1.6, ms=6),
    'hybrid_s064':   dict(label='Hybrid step=64 (100% anchors)', color='#1A5276', marker='D', ls='-.', lw=2.0, ms=7),
}

FALLBACK_COLORS = ['#9B59B6', '#E74C3C', '#27AE60', '#F39C12',
                   '#2980B9', '#D35400', '#7F8C8D', '#16A085']


def _style(alg_id: str, idx: int = 0) -> dict:
    if alg_id in ALG_STYLE:
        return ALG_STYLE[alg_id]
    c = FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
    return dict(label=alg_id, color=c, marker='D', ls=':', lw=1.4, ms=5)


# ── data helpers ──────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    return list(raw.values())


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def group_by(records: list[dict], keys: list[str]) -> dict:
    """Group list of dicts by a tuple of key values."""
    out = defaultdict(list)
    for r in records:
        gk = tuple(r.get(k, '') for k in keys)
        out[gk].append(r)
    return dict(out)


def average_field_records(records: list[dict]) -> list[dict]:
    """
    Average multiple field/file records at the same (algorithm, eb_rel) point.
    Used so per-dataset RD curves show the mean over all fields/files.
    """
    by_alg_eb = defaultdict(list)
    for r in records:
        by_alg_eb[(r['algorithm'], r.get('eb_rel', 0))].append(r)

    averaged = []
    for (alg, eb), recs in by_alg_eb.items():
        avg = {
            'algorithm': alg,
            'eb_rel':    eb,
            'bit_rate':  np.mean([r['bit_rate'] for r in recs]),
            'psnr':      np.mean([r['psnr']     for r in recs]),
            'ssim':      np.mean([r['ssim']     for r in recs]),
            'cr':        np.mean([r['cr']        for r in recs]),
        }
        averaged.append(avg)
    return averaged


# ── RD curve plot ─────────────────────────────────────────────────────────────

def plot_rd_curves(records: list[dict], dataset_label: str,
                  out_dir: Path, dataset_id: str) -> None:
    """One figure with two subplots: Rate-PSNR and Rate-SSIM."""
    averaged = average_field_records(records)
    by_alg   = defaultdict(list)
    for r in averaged:
        by_alg[r['algorithm']].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{dataset_label} — Rate-Distortion Curves',
                 fontsize=13, fontweight='bold')

    metrics = [
        ('psnr', 'PSNR (dB)',           axes[0]),
        ('ssim', 'SSIM (approx.)',      axes[1]),
    ]

    for idx, (alg_id, recs) in enumerate(sorted(by_alg.items())):
        recs_s = sorted(recs, key=lambda r: r['bit_rate'])
        br = [r['bit_rate'] for r in recs_s]
        st = _style(alg_id, idx)
        for metric, ylabel, ax in metrics:
            yv = [r[metric] for r in recs_s]
            ax.plot(br, yv,
                    color=st['color'], marker=st['marker'],
                    ls=st['ls'], lw=st['lw'], ms=st['ms'],
                    label=st['label'])

    for metric, ylabel, ax in metrics:
        ax.set_xlabel('Bit-rate (bpv)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        if metric == 'ssim':
            ax.set_ylim(top=1.0)

    plt.tight_layout()
    out_path = out_dir / f'{dataset_id}_rd.pdf'
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ── CR table ──────────────────────────────────────────────────────────────────

def make_cr_table(all_records: list[dict], cfg: dict, out_dir: Path) -> None:
    """
    Print and save a Compression-Ratio comparison table (matching paper Table 3).
    Uses the three CR-table error bounds from config.
    """
    cr_ebs = cfg.get('error_bounds_cr_table', [1e-2, 1e-3, 1e-4])
    # snap eb_rel to nearest table eb
    tol = 0.05  # within 5 % of log10 value

    def nearest_cr_eb(eb):
        for ceb in cr_ebs:
            if abs(math.log10(eb) - math.log10(ceb)) < tol:
                return ceb
        return None

    # Build: {(ds, field, alg, cr_eb): cr}
    data = defaultdict(dict)
    for r in all_records:
        ceb = nearest_cr_eb(r.get('eb_rel', 0))
        if ceb is None:
            continue
        key = (r['dataset'], r['field'], r['algorithm'])
        data[key][ceb] = r['cr']

    # Collect all unique datasets and algorithms
    datasets = sorted(set(k[0] for k in data))
    algs     = sorted(set(k[2] for k in data))

    lines = []
    header = ('Dataset', 'Field') + tuple(f'{a}_{e:.0e}' for a in algs for e in cr_ebs)
    lines.append('\t'.join(header))

    for ds in datasets:
        fields = sorted(set(k[1] for k in data if k[0] == ds))
        for field in fields:
            row = [ds, field]
            for alg in algs:
                key = (ds, field, alg)
                for ceb in cr_ebs:
                    val = data.get(key, {}).get(ceb, None)
                    row.append(f'{val:.2f}' if val is not None else 'N/A')
            lines.append('\t'.join(row))

    table_path = out_dir / 'cr_table.tsv'
    table_path.write_text('\n'.join(lines))
    print(f'  CR table saved: {table_path}')

    # Also pretty-print to stdout
    print('\n── Compression Ratio Table ──────────────────────────────────────────')
    for line in lines[:25]:    # cap output
        print(line)
    if len(lines) > 25:
        print(f'  ... ({len(lines)-25} more rows — see {table_path})')


# ── per-dataset summary ───────────────────────────────────────────────────────

def plot_dataset_summary_bar(all_records: list[dict], cfg: dict,
                              out_dir: Path) -> None:
    """
    Bar chart: mean CR at eb=1e-3 across all algorithms, one bar per dataset.
    Quick at-a-glance summary figure.
    """
    target_eb = 1e-3
    tol       = 0.05

    def is_target(eb):
        return abs(math.log10(eb) - math.log10(target_eb)) < tol

    by_ds_alg = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        if is_target(r.get('eb_rel', 0)):
            by_ds_alg[r['dataset']][r['algorithm']].append(r['cr'])

    datasets = sorted(by_ds_alg)
    algs     = sorted(set(a for d in by_ds_alg.values() for a in d))
    x        = np.arange(len(datasets))
    w        = 0.8 / max(len(algs), 1)

    fig, ax = plt.subplots(figsize=(max(6, 2*len(datasets)), 5))

    for i, alg in enumerate(algs):
        vals = []
        for ds in datasets:
            crs = by_ds_alg[ds].get(alg, [])
            vals.append(np.mean(crs) if crs else 0.0)
        st = _style(alg, i)
        offset = (i - len(algs)/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w * 0.9,
                      label=st['label'], color=st['color'],
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([cfg['datasets'].get(d, {}).get('label', d)
                        for d in datasets], fontsize=9)
    ax.set_ylabel('Compression Ratio (CR)', fontsize=11)
    ax.set_title(f'Mean CR at eb=1e-3 by Dataset and Algorithm', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = out_dir / 'summary_cr_bar.pdf'
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot Hybrid QoZ+GP results')
    parser.add_argument('--results', default='results/results.json')
    parser.add_argument('--config',  default='config.yaml')
    parser.add_argument('--out-dir', default='plots/')
    parser.add_argument('--datasets', nargs='+', metavar='DS',
                        help='Only plot these dataset IDs')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_results(args.results)
    cfg         = load_config(args.config)

    ds_filter = set(args.datasets) if args.datasets else None
    if ds_filter:
        all_records = [r for r in all_records if r['dataset'] in ds_filter]

    if not all_records:
        print('No results to plot.')
        return

    # Per-dataset RD curves
    by_ds = defaultdict(list)
    for r in all_records:
        by_ds[r['dataset']].append(r)

    for ds_id, recs in sorted(by_ds.items()):
        ds_cfg = cfg['datasets'].get(ds_id, {})
        label  = ds_cfg.get('label', ds_id)
        print(f'\nPlotting {label} ({len(recs)} result records) …')
        plot_rd_curves(recs, label, out_dir, ds_id)

    # CR table
    print('\nBuilding CR table …')
    make_cr_table(all_records, cfg, out_dir)

    # Summary bar chart
    print('\nPlotting summary bar chart …')
    plot_dataset_summary_bar(all_records, cfg, out_dir)

    print(f'\nAll plots written to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
