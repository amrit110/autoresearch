"""
Generate a stylish HTML progress report from experiments.tsv (or results.tsv).

Usage:
    uv run progress.py              # writes progress.html
    uv run progress.py out.html     # custom output path
"""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).parent


def load_rows() -> list[dict]:
    """Load rows from experiments.tsv, falling back to results.tsv."""
    for tsv in [REPO_ROOT / "experiments.tsv", REPO_ROOT / "results.tsv"]:
        if tsv.exists():
            with open(tsv) as f:
                return list(csv.DictReader(f, delimiter="\t"))
    return []


def safe_float(v: object) -> float | None:
    """Parse v as float, returning None for non-numeric, NaN, or inf."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def status_badge(status: str) -> str:
    """Return an inline-styled HTML badge for the given experiment status."""
    palette = {
        "keep": ("#4ade80", "rgba(74,222,128,0.12)", "rgba(74,222,128,0.25)"),
        "discard": ("#fbbf24", "rgba(251,191,36,0.12)", "rgba(251,191,36,0.25)"),
        "crash": ("#f87171", "rgba(248,113,113,0.12)", "rgba(248,113,113,0.25)"),
    }
    color, bg, border = palette.get(status, ("#94a3b8", "rgba(148,163,184,0.1)", "rgba(148,163,184,0.2)"))
    style = (
        f"color:{color};background:{bg};border:1px solid {border};"
        "padding:2px 9px;border-radius:5px;font-size:0.68rem;"
        "font-family:'JetBrains Mono',monospace;font-weight:500;white-space:nowrap"
    )
    return f'<span style="{style}">{status}</span>'


def render(rows: list[dict]) -> str:
    """Build the full HTML report string from a list of experiment rows."""
    annotated = []
    for i, r in enumerate(rows):
        tps = safe_float(r.get("tokens_per_sec"))
        bpb = safe_float(r.get("bpb"))
        annotated.append(
            {
                "idx": i,
                "session": r.get("session", ""),
                "commit": (r.get("commit") or "")[:7],
                "tps": tps,
                "bpb": bpb,
                "status": r.get("status", ""),
                "description": r.get("description", ""),
            }
        )

    kept = [r for r in annotated if r["status"] == "keep"]
    discarded = [r for r in annotated if r["status"] == "discard"]
    crashed = [r for r in annotated if r["status"] == "crash"]

    baseline_tps = kept[0]["tps"] if kept else None
    best_tps = max((r["tps"] for r in kept if r["tps"] is not None), default=None)
    baseline_bpb = kept[0]["bpb"] if kept else None

    running_best: list[float | None] = []
    current_best: float | None = None
    for r in annotated:
        if r["status"] == "keep" and r["tps"] is not None:
            current_best = max(r["tps"], current_best) if current_best is not None else r["tps"]
        running_best.append(current_best)

    improvement_pct = (
        (best_tps - baseline_tps) / baseline_tps * 100 if (baseline_tps and best_tps and baseline_tps > 0) else None
    )

    # ── Table rows ────────────────────────────────────────────────────────
    table_rows_html = ""
    for r in annotated:
        tps_str = f"{r['tps']:.1f}" if r["tps"] is not None else "—"
        bpb_str = f"{r['bpb']:.4f}" if r["bpb"] is not None else "—"
        opacity = " style='opacity:0.4'" if r["status"] == "crash" else ""
        table_rows_html += f"""
          <tr{opacity}>
            <td class="mono muted">{r["idx"] + 1}</td>
            <td class="mono muted">{r["session"] or "—"}</td>
            <td class="mono">{r["commit"] or "—"}</td>
            <td class="mono hi">{tps_str}</td>
            <td class="mono muted">{bpb_str}</td>
            <td>{status_badge(r["status"])}</td>
            <td class="desc">{r["description"] or "—"}</td>
          </tr>"""

    # ── Stats ─────────────────────────────────────────────────────────────
    n_total = len(annotated)
    baseline_str = f"{baseline_tps:.1f}" if baseline_tps is not None else "—"
    best_str = f"{best_tps:.1f}" if best_tps is not None else "—"
    impr_str = f"+{improvement_pct:.1f}%" if improvement_pct is not None else "—"
    impr_color = "#4ade80" if (improvement_pct is not None and improvement_pct > 0) else "#94a3b8"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Chart data (embedded as JSON; no f-string brace escaping needed) ──
    chart_json = json.dumps(
        {
            "experiments": annotated,
            "runningBest": running_best,
            "baselineTps": baseline_tps,
            "baselineBpb": baseline_bpb,
        }
    )

    empty_note = "" if n_total > 0 else '<p class="empty">No experiments recorded yet.</p>'
    charts_html = (
        ""
        if n_total == 0
        else """
      <div class="card chart-card">
        <p class="card-label">tokens / sec per experiment</p>
        <canvas id="tpsChart"></canvas>
      </div>
      <div class="card chart-card chart-card-sm">
        <p class="card-label">bits per byte — kept experiments</p>
        <canvas id="bpbChart"></canvas>
      </div>"""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>autoresearch · progress</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:       #07090f;
    --surface:  rgba(255,255,255,0.035);
    --surface2: rgba(255,255,255,0.06);
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.13);
    --text:     #f1f5f9;
    --muted:    #64748b;
    --dimmer:   #334155;
    --cyan:     #22d3ee;
    --green:    #4ade80;
    --amber:    #fbbf24;
    --red:      #f87171;
    --purple:   #c084fc;
  }}
  html {{ font-size: 15px; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    min-height: 100vh;
    padding: 2.25rem 2.75rem 5rem;
    max-width: 1240px;
    margin: 0 auto;
    line-height: 1.5;
  }}

  /* ── Header ── */
  header {{
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
  }}
  .logo {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 500;
    color: var(--cyan);
    letter-spacing: -0.02em;
  }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; }}
  .ts {{ margin-left: auto; font-size: 0.75rem; color: var(--dimmer); font-family: 'JetBrains Mono', monospace; }}

  /* ── Cards ── */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
  }}
  .stat-card {{
    padding: 1.1rem 1.3rem 1.2rem;
  }}
  .card-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.55rem;
  }}
  .card-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    line-height: 1;
    color: var(--text);
  }}
  .card-value.cyan   {{ color: var(--cyan); }}
  .card-value.green  {{ color: var(--green); }}
  .card-sub {{
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
  }}
  .pills {{
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-top: 0.55rem;
  }}
  .pill {{
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 5px;
  }}
  .pill-g {{ background: rgba(74,222,128,0.12);  color: var(--green); border: 1px solid rgba(74,222,128,0.25); }}
  .pill-a {{ background: rgba(251,191,36,0.12);  color: var(--amber); border: 1px solid rgba(251,191,36,0.25); }}
  .pill-r {{ background: rgba(248,113,113,0.12); color: var(--red);   border: 1px solid rgba(248,113,113,0.25); }}

  /* ── Charts ── */
  .chart-card {{
    padding: 1.4rem 1.5rem 1.25rem;
    margin-bottom: 1.25rem;
  }}
  .chart-card-sm canvas {{ max-height: 180px; }}

  /* ── Table ── */
  .table-card {{
    overflow: hidden;
    margin-top: 0.25rem;
  }}
  .table-head {{
    padding: 0.9rem 1.25rem;
    border-bottom: 1px solid var(--border);
  }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.55rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    background: rgba(255,255,255,0.015);
  }}
  td {{
    font-size: 0.82rem;
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: var(--surface2); }}
  .mono {{ font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; }}
  .muted {{ color: var(--muted); }}
  .hi {{ color: var(--cyan); }}
  .desc {{ max-width: 340px; color: var(--text); }}
  .empty {{ padding: 3rem; text-align: center; color: var(--muted); font-size: 0.9rem; }}
</style>
</head>
<body>

<header>
  <span class="logo">autoresearch</span>
  <span class="subtitle">inference throughput experiments</span>
  <span class="ts">generated {generated_at}</span>
</header>

<div class="stats-grid">
  <div class="card stat-card">
    <p class="card-label">experiments</p>
    <p class="card-value">{n_total}</p>
    <div class="pills">
      <span class="pill pill-g">{len(kept)} kept</span>
      <span class="pill pill-a">{len(discarded)} disc.</span>
      {"" if not crashed else f'<span class="pill pill-r">{len(crashed)} crash</span>'}
    </div>
  </div>
  <div class="card stat-card">
    <p class="card-label">baseline</p>
    <p class="card-value cyan">{baseline_str}</p>
    <p class="card-sub">tok / s</p>
  </div>
  <div class="card stat-card">
    <p class="card-label">best</p>
    <p class="card-value green">{best_str}</p>
    <p class="card-sub">tok / s</p>
  </div>
  <div class="card stat-card">
    <p class="card-label">improvement</p>
    <p class="card-value" style="color:{impr_color}">{impr_str}</p>
    <p class="card-sub">over baseline</p>
  </div>
</div>

{empty_note}
{charts_html}

<div class="card table-card">
  <div class="table-head"><p class="card-label">experiment log</p></div>
  {
        "<p class='empty'>No experiments yet.</p>"
        if n_total == 0
        else f'''
  <div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>#</th><th>session</th><th>commit</th>
        <th>tok/s</th><th>bpb</th><th>status</th><th>description</th>
      </tr>
    </thead>
    <tbody>{table_rows_html}</tbody>
  </table>
  </div>'''
    }
</div>

<script id="chartData" type="application/json">{chart_json}</script>
<script>
(function () {{
  const raw = JSON.parse(document.getElementById('chartData').textContent);
  const exps = raw.experiments;
  const runningBest = raw.runningBest;
  const baselineTps = raw.baselineTps;
  const baselineBpb = raw.baselineBpb;
  if (!exps || exps.length === 0) return;

  const GRID   = 'rgba(255,255,255,0.04)';
  const TICK   = '#475569';
  const MONO   = "'JetBrains Mono', monospace";
  const TOOLTIP_BG = 'rgba(7,9,15,0.95)';

  const barColors = exps.map(e =>
    e.status === 'keep'    ? '#4ade80' :
    e.status === 'discard' ? '#fbbf24' :
    e.status === 'crash'   ? '#f87171' : '#475569'
  );

  const baselineDataset = baselineTps ? [{{
    label: 'baseline',
    data: Array(exps.length).fill(baselineTps),
    type: 'line',
    borderColor: 'rgba(255,255,255,0.18)',
    borderDash: [5, 5],
    borderWidth: 1,
    pointRadius: 0,
    fill: false,
    order: 3,
  }}] : [];

  // ── TPS chart ────────────────────────────────────────────────────────
  new Chart(document.getElementById('tpsChart'), {{
    type: 'bar',
    data: {{
      labels: exps.map((e, i) => e.commit || String(i + 1)),
      datasets: [
        ...baselineDataset,
        {{
          label: 'running best',
          data: runningBest,
          type: 'line',
          borderColor: 'rgba(34,211,238,0.65)',
          borderWidth: 2,
          pointRadius: 0,
          stepped: true,
          fill: false,
          order: 2,
        }},
        {{
          label: 'tok/s',
          data: exps.map(e => e.tps),
          backgroundColor: barColors.map(c => c + 'cc'),
          borderColor: barColors,
          borderWidth: 1,
          borderRadius: 3,
          borderSkipped: false,
          order: 4,
        }},
      ],
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          labels: {{ color: '#64748b', font: {{ size: 11 }}, boxWidth: 14, padding: 16 }},
        }},
        tooltip: {{
          backgroundColor: TOOLTIP_BG,
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          padding: 10,
          callbacks: {{
            afterBody(ctx) {{
              const d = exps[ctx[0].dataIndex];
              return d?.description ? ['', d.description] : [];
            }},
          }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{ color: TICK, font: {{ family: MONO, size: 10 }}, maxRotation: 45 }},
          grid: {{ color: GRID }},
        }},
        y: {{
          ticks: {{ color: TICK, font: {{ family: MONO, size: 10 }} }},
          grid: {{ color: GRID }},
          title: {{ display: true, text: 'tok / s', color: TICK, font: {{ size: 10 }} }},
          beginAtZero: true,
        }},
      }},
    }},
  }});

  // ── BPB chart ────────────────────────────────────────────────────────
  const bpbPoints = exps
    .filter(e => e.status === 'keep' && e.bpb != null)
    .map(e => ({{ x: e.idx, y: e.bpb, desc: e.description, commit: e.commit }}));

  if (bpbPoints.length > 0) {{
    const bpbBaseline = baselineBpb ? [{{
      label: 'baseline bpb',
      data: [{{ x: 0, y: baselineBpb }}, {{ x: exps.length - 1, y: baselineBpb }}],
      type: 'line',
      borderColor: 'rgba(255,255,255,0.18)',
      borderDash: [5, 5],
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
    }}] : [];

    new Chart(document.getElementById('bpbChart'), {{
      type: 'scatter',
      data: {{
        datasets: [
          ...bpbBaseline,
          {{
            label: 'bpb',
            data: bpbPoints.map(p => ({{ x: p.x, y: p.y }})),
            backgroundColor: 'rgba(192,132,252,0.75)',
            borderColor: '#c084fc',
            borderWidth: 1,
            pointRadius: 6,
            pointHoverRadius: 8,
          }},
        ],
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{
            labels: {{ color: '#64748b', font: {{ size: 11 }}, boxWidth: 14, padding: 16 }},
          }},
          tooltip: {{
            backgroundColor: TOOLTIP_BG,
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            titleColor: '#f1f5f9',
            bodyColor: '#94a3b8',
            padding: 10,
            callbacks: {{
              label(ctx) {{
                const p = bpbPoints[ctx.dataIndex];
                return p ? `bpb ${{p.y.toFixed(4)}}  ${{p.commit}}  ${{p.desc}}` : '';
              }},
            }},
          }},
        }},
        scales: {{
          x: {{
            title: {{ display: true, text: 'experiment index', color: TICK, font: {{ size: 10 }} }},
            ticks: {{ color: TICK, font: {{ family: MONO, size: 10 }} }},
            grid: {{ color: GRID }},
          }},
          y: {{
            title: {{ display: true, text: 'bpb', color: TICK, font: {{ size: 10 }} }},
            ticks: {{
              color: TICK,
              font: {{ family: MONO, size: 10 }},
              callback: v => v.toFixed(3),
            }},
            grid: {{ color: GRID }},
          }},
        }},
      }},
    }});
  }}
}})();
</script>

</body>
</html>"""


def main() -> None:
    """Entry point: write the HTML report to progress.html (or a custom path)."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "progress.html"
    rows = load_rows()
    html = render(rows)
    out.write_text(html)
    print(f"wrote {out}  ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
