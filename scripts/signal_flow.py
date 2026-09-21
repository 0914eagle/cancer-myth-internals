"""Signal flow (chat 9/19): where the premise-truth signal lives between the premise
span and the answer position, and whether moving it moves the answer. Judge calls 0.

    R=$SUITE_DIR/mechanism/mech_v1/rows.jsonl      # rows from mechanism_scan extract (FPQ/twin/fpara/NFP with spans)
    CUDA_VISIBLE_DEVICES=0 python scripts/signal_flow.py fit   --suite-dir $SUITE_DIR --config configs/qwen25_7b.yaml --rows $R --name flow_v1
    CUDA_VISIBLE_DEVICES=0 python scripts/signal_flow.py scan  --suite-dir $SUITE_DIR --config configs/qwen25_7b.yaml --name flow_v1
    CUDA_VISIBLE_DEVICES=0 python scripts/signal_flow.py patch --suite-dir $SUITE_DIR --config configs/qwen25_7b.yaml --name flow_v1
    python scripts/signal_flow.py eval --suite-dir $SUITE_DIR --name flow_v1

Prompt context is the DIRECT Yes/No question by default (--prompt plain for the
bare question), so "answer position" means the token that emits Yes/No.
fit:   one forward per row; caches span-mean and last-token residuals per layer;
       fits d_l per layer, 2-fold by origin (src/signal_flow.py).
scan:  one forward per row; projects EVERY token on its held-out d_l; caches (L, T).
patch: per FPQ-twin pair and per layer L, adds (twin span mean - FPQ span mean)
       to the FPQ span tokens at L and records the last-token projections at every
       layer plus P(Yes). ~326 x 29 forwards.
eval:  A delivery curve (span vs last vs natural, per layer); B layer x bin heatmap;
       C transfer fraction per patch layer at readout layers, and P(Yes).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src import signal_flow as sf
from src.jsonl import read_jsonl
from src.pilot import frozen_json, output_lock

DIRECT = "Does the following question rest on a false premise? Answer only Yes or No.\n\nQuestion: {question}"


def prompt_text(mode, text):
    return DIRECT.format(question=text) if mode == "direct" else text


def _runtime(config_path):
    from src import baseline_generation as bg
    from src.config import load_config
    return bg.make_runtime(load_config(config_path))


def encode_row(tok, mode, row):
    from src.pilot_model import render
    rendered = render(tok, prompt_text(mode, row["text"]))
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    q0, q1 = sf.content_positions(rendered, row["text"])
    span_abs = (q0 + int(row["span"][0]), q0 + int(row["span"][1])) if row.get("span") else None
    roles = sf.token_roles(enc["offset_mapping"], q0, q1, span_abs)
    span_idx = [i for i, r in enumerate(roles) if r == sf.ROLE_SPAN]
    return enc["input_ids"], roles, span_idx


def forward_states(runtime, ids, hook=None):
    """(hidden (L, T, d) float32 numpy, p_yes float). hook: optional (block index -> callable) registered once."""
    torch, model, tok = runtime.torch, runtime.model, runtime.tokenizer
    x = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
    yes, no = (tok(s, add_special_tokens=False)["input_ids"][0] for s in ("Yes", "No"))
    with torch.inference_mode():
        out = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, output_hidden_states=True, return_dict=True)
        hid = torch.stack(out.hidden_states[1:], 0)[:, 0].float().cpu().numpy()
        lg = out.logits[0, -1, [yes, no]].float().cpu()
        p_yes = float(torch.softmax(lg, 0)[0])
    del out
    return hid, p_yes


def load_rows(out):
    return list(read_jsonl(out / "rows.jsonl"))


def stage_fit(args, out):
    rows = list(read_jsonl(args.rows))
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    runtime = _runtime(args.config)
    tok = runtime.tokenizer
    frozen_json(out / "identity.json", {**runtime.identity, "prompt": args.prompt, "version": "signal-flow-v1"})
    cache = out / "cache_fit"; cache.mkdir(exist_ok=True)
    with output_lock(out / "fit"):
        for i, r in enumerate(rows, 1):
            path = cache / f"{r['id']}.npz"
            if path.exists():
                continue
            ids, roles, span_idx = encode_row(tok, args.prompt, r)
            hid, p_yes = forward_states(runtime, ids)
            span_mean = hid[:, span_idx, :].mean(1) if span_idx else np.full(hid.shape[::2], np.nan, dtype=np.float32)
            q_idx = [k for k, ro in enumerate(roles) if ro in (sf.ROLE_Q_BEFORE, sf.ROLE_SPAN, sf.ROLE_Q_AFTER)]
            np.savez_compressed(path, span_mean=span_mean.astype(np.float16), last=hid[:, -1, :].astype(np.float16),
                                q_mean=hid[:, q_idx, :].mean(1).astype(np.float16), p_yes=np.asarray(p_yes),
                                n_span=np.asarray(len(span_idx)), n_tokens=np.asarray(len(ids)))
            if i % 50 == 0 or i == len(rows):
                print(f"[fit] {i}/{len(rows)}", flush=True)
        # directions per layer, 2-fold by origin

        def load(r):
            return np.load(cache / f"{r['id']}.npz")
        pos = [r for r in rows if r.get("span") and r["label"] == 1]
        neg = [r for r in rows if r.get("span") and r["label"] == 0 and r["kind"] == "twin"]
        L, d = load(pos[0])["span_mean"].shape
        D = np.zeros((2, L, d), np.float32); Cc = np.zeros((2, L, d), np.float32)
        for fold in range(2):
            tr_pos = [r for r in pos if sf.fold_of(r["origin"]) != fold]
            tr_neg = [r for r in neg if sf.fold_of(r["origin"]) != fold]
            Xp = np.stack([load(r)["span_mean"].astype(np.float32) for r in tr_pos])
            Xn = np.stack([load(r)["span_mean"].astype(np.float32) for r in tr_neg])
            for l in range(L):
                D[fold, l], Cc[fold, l] = sf.fit_direction(Xp[:, l], Xn[:, l], C=args.C)
            print(f"[fit] fold {fold}: directions from {len(tr_pos)} false / {len(tr_neg)} true spans", flush=True)
        np.savez_compressed(out / "directions.npz", d=D, center=Cc)
    print(f"Written to {out}. Judge calls: 0.")


def _proj(vec, D, Cc, fold):
    """vec (L, d) -> (L,) projections with the fold's per-layer direction."""
    return np.einsum("ld,ld->l", vec.astype(np.float32) - Cc[fold], D[fold])


def _check_prompt(args, out):
    saved = json.loads((out / "identity.json").read_text()).get("prompt")
    if saved != args.prompt:
        raise ValueError(f"fit ran with --prompt {saved}; pass the same prompt (got {args.prompt})")


def stage_scan(args, out):
    _check_prompt(args, out)
    rows = load_rows(out)
    dirs = np.load(out / "directions.npz"); D, Cc = dirs["d"], dirs["center"]
    runtime = _runtime(args.config)
    tok = runtime.tokenizer
    cache = out / "cache_scan"; cache.mkdir(exist_ok=True)
    with output_lock(out / "scan"):
        for i, r in enumerate(rows, 1):
            path = cache / f"{r['id']}.npz"
            if path.exists():
                continue
            ids, roles, _ = encode_row(tok, args.prompt, r)
            hid, _ = forward_states(runtime, ids)
            fold = sf.fold_of(r["origin"])
            proj = np.einsum("ltd,ld->lt", hid - Cc[fold][:, None, :], D[fold])  # (L, T)
            np.savez_compressed(path, proj=proj.astype(np.float32), roles=np.asarray(roles, dtype=np.int8))
            if i % 50 == 0 or i == len(rows):
                print(f"[scan] {i}/{len(rows)}", flush=True)
    print(f"Written to {out}. Judge calls: 0.")


def stage_patch(args, out):
    _check_prompt(args, out)
    rows = load_rows(out)
    by_origin = defaultdict(dict)
    for r in rows:
        by_origin[r["origin"]][r["kind"]] = r
    pairs = [(d["natural"], d["twin"]) for d in by_origin.values() if "natural" in d and "twin" in d and d["natural"]["set"] == "fpq"]
    if args.limit:
        pairs = pairs[:args.limit]
    dirs = np.load(out / "directions.npz"); D, Cc = dirs["d"], dirs["center"]

    def fit(r):
        return np.load(out / "cache_fit" / f"{r['id']}.npz")
    runtime = _runtime(args.config)
    tok, torch, model = runtime.tokenizer, runtime.torch, runtime.model
    from src.modeling import decoder_layers
    blocks = decoder_layers(model)
    layers = args.layers or list(range(1, len(blocks) + 1))
    cache = out / "cache_patch"; cache.mkdir(exist_ok=True)
    state = {"delta": None, "idx": None}

    def make_hook():
        def hook(_m, _inp, output):
            if state["delta"] is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            h[:, state["idx"], :] = h[:, state["idx"], :] + state["delta"].to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    with output_lock(out / "patch"):
        for i, (f, t) in enumerate(pairs, 1):
            path = cache / f"{f['id']}.npz"
            if path.exists():
                continue
            ids, roles, span_idx = encode_row(tok, args.prompt, f)
            if not span_idx:
                continue
            fold = sf.fold_of(f["origin"])
            zf, zt = fit(f), fit(t)
            base_f = _proj(zf["last"], D, Cc, fold); base_t = _proj(zt["last"], D, Cc, fold)
            delta_all = zt["span_mean"].astype(np.float32) - zf["span_mean"].astype(np.float32)  # (L, d)
            patched = np.full((len(blocks), len(blocks)), np.nan, np.float32)  # [patch layer-1, readout layer-1]
            patched_last = np.zeros((len(blocks), len(blocks), delta_all.shape[1]), np.float16)  # full last-token states
            p_yes = np.full(len(blocks), np.nan, np.float32)
            device = model.get_input_embeddings().weight.device
            for L in layers:
                handle = blocks[L - 1].register_forward_hook(make_hook())
                try:
                    state["delta"] = torch.tensor(delta_all[L - 1], device=device)
                    state["idx"] = span_idx
                    hid, py = forward_states(runtime, ids)
                finally:
                    state["delta"] = None
                    handle.remove()
                patched[L - 1] = _proj(hid[:, -1, :], D, Cc, fold)
                patched_last[L - 1] = hid[:, -1, :].astype(np.float16)
                p_yes[L - 1] = py
            np.savez_compressed(path, base_fpq=base_f, base_twin=base_t, p_yes_fpq=zf["p_yes"], p_yes_twin=zt["p_yes"],
                                patched=patched, patched_last=patched_last, p_yes_patched=p_yes, layers=np.asarray(layers),
                                fold=np.asarray(fold), last_fpq=zf["last"], last_twin=zt["last"])
            if i % 10 == 0 or i == len(pairs):
                print(f"[patch] {i}/{len(pairs)} pairs", flush=True)
    print(f"Written to {out}. Judge calls: 0.")


def stage_eval(args, out):
    rows = load_rows(out)
    dirs = np.load(out / "directions.npz"); D, Cc = dirs["d"], dirs["center"]
    L = D.shape[1]

    def fit(r):
        return np.load(out / "cache_fit" / f"{r['id']}.npz")
    by_origin = defaultdict(dict)
    for r in rows:
        by_origin[r["origin"]][r["kind"]] = r
    pairs = [(d["natural"], d["twin"]) for d in by_origin.values() if "natural" in d and "twin" in d and d["natural"]["set"] == "fpq"]
    nfp = [r for r in rows if r["kind"] == "natural" and r["set"] == "nfp"]
    fpq = [a for a, _ in pairs]
    lines = [f"# Signal flow ({args.name}, prompt={json.loads((out / 'identity.json').read_text()).get('prompt')})", "",
             f"pairs {len(pairs)}; NFP {len(nfp)}; directions fitted on span means, 2-fold by origin; every score below uses the held-out fold's direction.", ""]
    # A. delivery curve
    lines += ["## A. Per-layer AUROC of the projection on d_l (twins: FPQ vs true twin; natural: FPQ vs NFP)", "",
              "| layer | span mean (twins) | last token (twins) | question mean (natural) | last token (natural) | P(Yes) FPQ / twin / NFP |", "|---|---:|---:|---:|---:|---|"]
    def projs(rs, key):
        return np.stack([_proj(fit(r)[key], D, Cc, sf.fold_of(r["origin"])) for r in rs])
    P = {k: {key: projs(rs, key) for key in ("span_mean", "last", "q_mean")} for k, rs in (("fpq", fpq), ("twin", [t for _, t in pairs]), ("nfp", nfp))}
    py = {k: float(np.mean([float(fit(r)["p_yes"]) for r in rs])) for k, rs in (("fpq", fpq), ("twin", [t for _, t in pairs]), ("nfp", nfp))}
    for l in range(L):
        y_tw = [1] * len(fpq) + [0] * len(pairs)
        a_span = sf.auroc(y_tw, np.concatenate([P["fpq"]["span_mean"][:, l], P["twin"]["span_mean"][:, l]]))
        a_last = sf.auroc(y_tw, np.concatenate([P["fpq"]["last"][:, l], P["twin"]["last"][:, l]]))
        y_nat = [1] * len(fpq) + [0] * len(nfp)
        a_q = sf.auroc(y_nat, np.concatenate([P["fpq"]["q_mean"][:, l], P["nfp"]["q_mean"][:, l]]))
        a_ln = sf.auroc(y_nat, np.concatenate([P["fpq"]["last"][:, l], P["nfp"]["last"][:, l]]))
        def f(v):
            return "NA" if v is None else f"{v:.3f}"
        lines.append(f"| L{l + 1} | {f(a_span)} | {f(a_last)} | {f(a_q)} | {f(a_ln)} | {py['fpq']:.2f} / {py['twin']:.2f} / {py['nfp']:.2f} |")
    # B. heatmap from scan
    scan_dir = out / "cache_scan"
    if scan_dir.exists() and any(scan_dir.iterdir()):
        def binned(r):
            z = np.load(scan_dir / f"{r['id']}.npz")
            return sf.bin_means(z["proj"], sf.bin_indices(list(z["roles"])))
        B_f = [binned(a) for a, _ in pairs]; B_t = [binned(t) for _, t in pairs]; B_n = [binned(r) for r in nfp]
        table = {}
        for l in range(L):
            for b in sf.BINS:
                s_f = [m[b][l] for m in B_f if b in m]; s_t = [m[b][l] for m in B_t if b in m]
                table[(l, b)] = sf.auroc([1] * len(s_f) + [0] * len(s_t), s_f + s_t) if s_f and s_t else None
        lines += [""] + sf.heatmap_lines([f"L{l + 1}" for l in range(L)], list(sf.BINS), table,
                                         "B. Layer x position heatmap: AUROC FPQ vs true twin of the mean projection in each token bin")
        nat = {}
        for l in range(L):
            for b in ("pre", "q_before", "post", "last"):
                s_f = [m[b][l] for m in B_f if b in m]; s_n = [m[b][l] for m in B_n if b in m]
                nat[(l, b)] = sf.auroc([1] * len(s_f) + [0] * len(s_n), s_f + s_n) if s_f and s_n else None
        lines += [""] + sf.heatmap_lines([f"L{l + 1}" for l in range(L)], ["pre", "q_before", "post", "last"], nat,
                                         "B2. Same direction, natural FPQ vs NFP (NFP has no span: q_before = whole question)")
    # last-token directions (FPQ vs twin at the answer position), 2-fold by origin: the "0.78-type" probe
    E = np.zeros_like(D); Ec = np.zeros_like(Cc)
    for fold in range(2):
        tr = [(a, t) for a, t in pairs if sf.fold_of(a["origin"]) != fold]
        Xp = np.stack([fit(a)["last"].astype(np.float32) for a, _ in tr]); Xn = np.stack([fit(t)["last"].astype(np.float32) for _, t in tr])
        for l in range(L):
            E[fold, l], Ec[fold, l] = sf.fit_direction(Xp[:, l], Xn[:, l], C=0.01)
    lines += ["", "## A2. Last-token direction e_l (fitted at the answer position, 2-fold): held-out AUROC per layer", "",
              "| layer | last token (twins) | last token (natural FPQ vs NFP) |", "|---|---:|---:|"]
    PE = {k: np.stack([_proj(fit(r)["last"], E, Ec, sf.fold_of(r["origin"])) for r in rs])
          for k, rs in (("fpq", fpq), ("twin", [t for _, t in pairs]), ("nfp", nfp))}
    for l in range(L):
        a1 = sf.auroc([1] * len(fpq) + [0] * len(pairs), np.concatenate([PE["fpq"][:, l], PE["twin"][:, l]]))
        a2 = sf.auroc([1] * len(fpq) + [0] * len(nfp), np.concatenate([PE["fpq"][:, l], PE["nfp"][:, l]]))
        lines.append(f"| L{l + 1} | {'NA' if a1 is None else f'{a1:.3f}'} | {'NA' if a2 is None else f'{a2:.3f}'} |")
    # C. patch
    patch_dir = out / "cache_patch"
    if patch_dir.exists() and any(patch_dir.iterdir()):
        Z, TF2 = [], []  # TF2: per pair (Lp, R) transfer along the last-token direction, computed once per file
        for pth in sorted(patch_dir.glob("*.npz")):
            with np.load(pth) as z:
                keep = {k: z[k] for k in ("base_fpq", "base_twin", "p_yes_fpq", "p_yes_twin", "patched", "p_yes_patched", "layers")}
                if "patched_last" in z.files:
                    fold = int(z["fold"])
                    pl = z["patched_last"].astype(np.float32)                      # (Lp, R, d)
                    pr = np.einsum("prd,rd->pr", pl - Ec[fold][None], E[fold])     # (Lp, R)
                    bf = np.einsum("rd,rd->r", z["last_fpq"].astype(np.float32) - Ec[fold], E[fold])
                    bt = np.einsum("rd,rd->r", z["last_twin"].astype(np.float32) - Ec[fold], E[fold])
                    TF2.append(sf.transfer_fraction(pr, bf[None, :], bt[None, :]))
                Z.append(keep)
        layers = list(Z[0]["layers"])
        readouts = [r for r in (12, 17, 20, 24, 28) if r <= L]
        lines += ["", f"## C. Span patching (n={len(Z)} pairs): transfer fraction of the last-token projection, and P(Yes)", "",
                  "transfer = (patched - FPQ) / (twin - FPQ), median over pairs, at readout layer R >= patch layer L. 1.0 = the answer position now looks like the twin's.", "",
                  "| patch L | " + " | ".join(f"R{r}" for r in readouts) + " | P(Yes) FPQ -> patched (twin) |", "|---|" + "---:|" * len(readouts) + "---|"]
        pf = float(np.mean([float(z["p_yes_fpq"]) for z in Z])); pt = float(np.mean([float(z["p_yes_twin"]) for z in Z]))
        for Lp in layers:
            cells = []
            for R in readouts:
                if R < Lp:
                    cells.append("-"); continue
                tf = [sf.transfer_fraction(z["patched"][Lp - 1, R - 1], z["base_fpq"][R - 1], z["base_twin"][R - 1]) for z in Z]
                tf = [v for v in tf if np.isfinite(v)]
                cells.append(f"{np.median(tf):.2f}" if tf else "NA")
            ppatched = float(np.nanmean([z["p_yes_patched"][Lp - 1] for z in Z]))
            lines.append(f"| L{Lp} | " + " | ".join(cells) + f" | {pf:.2f} -> {ppatched:.2f} ({pt:.2f}) |")
        if TF2:
            T2 = np.stack(TF2)  # (pairs, Lp, R)
            lines += ["", "## C2. Same patching, read along the LAST-TOKEN direction e_R (the axis that separates FPQ from twin at the answer position)", "",
                      "| patch L | " + " | ".join(f"R{r}" for r in readouts) + " |", "|---|" + "---:|" * len(readouts)]
            for Lp in layers:
                cells = []
                for R in readouts:
                    if R < Lp:
                        cells.append("-"); continue
                    v = T2[:, Lp - 1, R - 1]; v = v[np.isfinite(v)]
                    cells.append(f"{np.median(v):.2f}" if v.size else "NA")
                lines.append(f"| L{Lp} | " + " | ".join(cells) + " |")
    lines += ["", "Read: A shows how much of the span's truth signal is linearly present at the answer position per layer. "
              "B shows where along the prompt it lives and where it fades. C shows whether writing the twin's span state into the FPQ at layer L "
              "moves the answer position toward the twin (transfer near 1 at some R) and whether the Yes/No readout follows."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Written to {out / 'report.md'}. Judge calls: 0.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for stage in ("fit", "scan", "patch", "eval"):
        p = sub.add_parser(stage)
        p.add_argument("--suite-dir", required=True); p.add_argument("--name", default="flow_v1")
        if stage != "eval":
            p.add_argument("--config", required=True); p.add_argument("--prompt", choices=("direct", "plain"), default="direct")
        if stage == "fit":
            p.add_argument("--rows", required=True, help="mechanism_scan rows.jsonl"); p.add_argument("--C", type=float, default=0.01)
        if stage == "patch":
            p.add_argument("--layers", nargs="+", type=int, default=None); p.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.suite_dir).resolve() / "signal_flow" / args.name
    out.mkdir(parents=True, exist_ok=True)
    {"fit": stage_fit, "scan": stage_scan, "patch": stage_patch, "eval": stage_eval}[args.stage](args, out)


if __name__ == "__main__":
    main()
