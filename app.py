"""
llm-bench — Streamlit dashboard
Full-customization inference benchmark suite for vLLM, Ollama, llama.cpp
"""
import sys
import uuid
import time
import threading
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from backends import BACKEND_REGISTRY
from backends.base import GenerateParams
from benchmarks import BENCHMARK_REGISTRY
from core.runner import run_single_request
from core.aggregator import summarize
from core.storage import (
    save_request_results, save_benchmark_results,
    save_summary, load_all_request_results,
    load_all_benchmark_results, list_run_ids,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="llm-bench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
code, .stCode, .stTextInput input, .stSelectbox, .stNumberInput input {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dark header bar */
.bench-header {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
    border-bottom: 2px solid #00d4ff;
    padding: 1.2rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.bench-header h1 {
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.bench-header span {
    color: #888;
    font-size: 0.85rem;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    color: #6b7280;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    color: #00d4ff;
    font-size: 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

/* Status badges */
.badge-ok   { background:#064e3b; color:#6ee7b7; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-family:monospace; }
.badge-err  { background:#7f1d1d; color:#fca5a5; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-family:monospace; }
.badge-warn { background:#78350f; color:#fcd34d; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-family:monospace; }

section[data-testid="stSidebar"] {
    background: #0d1117;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bench-header">
    <h1>⚡ llm-bench</h1>
    <span>vLLM · Ollama · llama.cpp &nbsp;|&nbsp; TPS · TTFT · Prefill · Decode · Memory</span>
</div>
""", unsafe_allow_html=True)

# ── Load config ─────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config" / "defaults.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🖥️ Backends")
    backend_status = {}
    selected_backends = []
    for bname, bcls in BACKEND_REGISTRY.items():
        bcfg = cfg["backends"].get(bname.replace(".", "").replace("llama", "llamacpp"), {})
        host = st.text_input(f"{bname} host", value=bcfg.get("host", "localhost"), key=f"host_{bname}")
        port = st.number_input(f"{bname} port", value=int(bcfg.get("port", 8000)), key=f"port_{bname}", step=1)
        enabled = st.checkbox(f"Enable {bname}", value=True, key=f"en_{bname}")
        if enabled:
            selected_backends.append((bname, bcls, host, int(port)))
        st.divider()

    st.markdown("### 🤖 Model")
    model_name = st.text_input("Model name / path", value="qwen2.5-7b-instruct", help="Must match the model loaded in each backend")
    quant_options = cfg.get("quant_tags", ["fp16", "q8_0", "q4_k_m"])
    quant_label = st.selectbox("Quantization label (for tagging results)", quant_options)
    custom_quant = st.text_input("...or enter custom quant tag", value="")
    quant = custom_quant.strip() if custom_quant.strip() else quant_label

    st.markdown("### ⚙️ Run Parameters")
    n_requests = st.slider("Requests per run", 1, 100, cfg["defaults"]["n_requests"])
    max_tokens = st.slider("Max output tokens", 32, 2048, cfg["defaults"]["max_tokens"])
    temperature = st.slider("Temperature", 0.0, 2.0, float(cfg["defaults"]["temperature"]), step=0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, float(cfg["defaults"]["top_p"]), step=0.05)

    st.markdown("### 📝 Prompt")
    preset_prompts = cfg["defaults"]["prompt_lengths"]
    prompt_preset = st.selectbox("Preset prompt length", ["short", "medium", "long", "custom"])
    if prompt_preset == "custom":
        prompt_text = st.text_area("Custom prompt", value=preset_prompts["short"], height=120)
    else:
        prompt_text = preset_prompts[prompt_preset]
        st.caption(f"_{prompt_text[:80]}..._" if len(prompt_text) > 80 else f"_{prompt_text}_")

    st.markdown("### 🔀 Concurrency Sweep")
    default_levels = cfg["defaults"].get("concurrency_levels", [1, 4, 16])
    concurrency_levels_input = st.text_input(
        "Concurrency levels (comma-separated)",
        value=", ".join(str(x) for x in default_levels),
        help="e.g. 1, 4, 8, 16, 32"
    )
    try:
        concurrency_levels = [int(x.strip()) for x in concurrency_levels_input.split(",") if x.strip()]
    except ValueError:
        concurrency_levels = default_levels
        st.warning("Invalid concurrency levels, using defaults.")
    sweep_requests_per_level = st.slider("Requests per concurrency level", 5, 50, 10)

    st.markdown("### 📊 Metrics to collect")
    col1, col2 = st.columns(2)
    with col1:
        do_tps      = st.checkbox("TPS",           value=True)
        do_ttft     = st.checkbox("TTFT",          value=True)
        do_prefill  = st.checkbox("Prefill time",  value=True)
    with col2:
        do_decode   = st.checkbox("Decode time",   value=True)
        do_latency  = st.checkbox("Total latency", value=True)
        do_memory   = st.checkbox("Memory",        value=True)

    st.markdown("### 🧪 Task Benchmarks")
    selected_benchmarks = []
    for bname in BENCHMARK_REGISTRY:
        if st.checkbox(f"Run {bname} benchmark", value=False, key=f"bench_{bname}"):
            selected_benchmarks.append(bname)
    bench_sample_limit = st.number_input("Max samples per benchmark", min_value=1, max_value=500, value=20)
    bench_max_tokens = st.number_input("Benchmark max tokens", min_value=32, max_value=1024, value=256)

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_run, tab_results, tab_compare, tab_sweep, tab_history = st.tabs([
    "🚀 Run", "📈 Results", "⚖️ Compare", "🔀 Sweep", "🗂️ History"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.markdown("#### Backend Health")

    health_cols = st.columns(len(BACKEND_REGISTRY))
    live_backends = []

    for i, (bname, bcls, host, port) in enumerate(selected_backends):
        instance = bcls(host=host, port=port, model=model_name)
        ok = instance.health_check()
        with health_cols[i % len(health_cols)]:
            badge = "badge-ok" if ok else "badge-err"
            label = "ONLINE" if ok else "OFFLINE"
            st.markdown(f"**{bname}** <span class='{badge}'>{label}</span>", unsafe_allow_html=True)
            if ok:
                live_backends.append((bname, instance))

    if not live_backends:
        st.warning("No backends online. Start vLLM, Ollama, or llama.cpp and refresh.")
    else:
        st.success(f"{len(live_backends)} backend(s) ready: {', '.join(b[0] for b in live_backends)}")

    st.divider()
    run_btn = st.button("⚡ Run Benchmark", type="primary", use_container_width=True, disabled=not live_backends)

    if run_btn:
        run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        st.session_state["last_run_id"] = run_id
        all_request_results = []
        all_benchmark_results = []

        params = GenerateParams(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        for bname, backend in live_backends:
            st.markdown(f"##### Running `{bname}` — {n_requests} requests")
            progress = st.progress(0)
            req_results = []

            for i in range(n_requests):
                result = run_single_request(backend, params, quant=quant)
                req_results.append(result)
                all_request_results.append(result)
                progress.progress((i + 1) / n_requests)

            errors = sum(1 for r in req_results if r.error)
            if errors:
                st.markdown(f"<span class='badge-warn'>{errors} errors</span>", unsafe_allow_html=True)

            summary = summarize(req_results)
            save_summary(summary, f"{run_id}_{bname}")

            # Quick inline metric display
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Output TPS", f"{summary.mean_output_tps:.1f}")
            m2.metric("TTFT p50", f"{summary.ttft_p50*1000:.0f}ms")
            m3.metric("Latency p95", f"{summary.total_latency_p95:.2f}s")
            m4.metric("Peak Mem", f"{summary.peak_memory_mb/1024:.1f}GB")

        # Task benchmarks
        for bench_name in selected_benchmarks:
            bench_cls = BENCHMARK_REGISTRY[bench_name]
            bench = bench_cls()
            for bname, backend in live_backends:
                st.markdown(f"##### Benchmark: `{bench_name}` on `{bname}`")
                with st.spinner("Running..."):
                    b_results = bench.run(
                        backend=backend,
                        quant=quant,
                        max_tokens=bench_max_tokens,
                        sample_limit=bench_sample_limit,
                    )
                all_benchmark_results.extend(b_results)
                if b_results:
                    avg = sum(r.score for r in b_results) / len(b_results)
                    st.metric(f"{bench.score_label} (avg)", f"{avg:.3f}")

        # Persist
        if all_request_results:
            save_request_results(all_request_results, run_id)
        if all_benchmark_results:
            save_benchmark_results(all_benchmark_results, run_id)

        st.success(f"Run `{run_id}` complete. View results in the **Results** tab.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    df = load_all_request_results()
    if df.empty:
        st.info("No results yet. Run a benchmark first.")
    else:
        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            backend_filter = st.multiselect("Backend", df["backend"].unique().tolist(), default=df["backend"].unique().tolist())
        with fc2:
            quant_filter = st.multiselect("Quant", df["quant"].unique().tolist(), default=df["quant"].unique().tolist())
        with fc3:
            model_filter = st.multiselect("Model", df["model"].unique().tolist(), default=df["model"].unique().tolist())

        mask = (
            df["backend"].isin(backend_filter) &
            df["quant"].isin(quant_filter) &
            df["model"].isin(model_filter)
        )
        df_f = df[mask].copy()
        st.caption(f"{len(df_f)} requests shown")

        # Chart selector
        active_metrics = []
        if do_tps:      active_metrics.append("output_tps")
        if do_ttft:     active_metrics.append("ttft")
        if do_prefill:  active_metrics.append("prefill_time")
        if do_decode:   active_metrics.append("decode_time")
        if do_latency:  active_metrics.append("total_time")
        if do_memory:   active_metrics.append("memory_used_mb")

        if not active_metrics:
            st.warning("Select at least one metric in the sidebar.")
        else:
            metric_sel = st.selectbox("Chart metric", active_metrics)
            chart_type = st.radio("Chart type", ["Box", "Line", "Scatter"], horizontal=True)

            if chart_type == "Box":
                fig = px.box(df_f, x="backend", y=metric_sel, color="quant",
                             points="outliers", title=f"{metric_sel} by backend & quant",
                             template="plotly_dark")
            elif chart_type == "Line":
                df_f = df_f.reset_index(drop=True).reset_index().rename(columns={"index": "request_idx"})
                fig = px.line(df_f, x="request_idx", y=metric_sel, color="backend",
                              line_dash="quant", title=f"{metric_sel} over requests",
                              template="plotly_dark")
            else:
                fig = px.scatter(df_f, x="prompt_tokens", y=metric_sel, color="backend",
                                 size="output_tokens", hover_data=["quant"],
                                 title=f"{metric_sel} vs prompt length", template="plotly_dark")

            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                              font_color="#c9d1d9", title_font_color="#00d4ff")
            st.plotly_chart(fig, use_container_width=True)

            # Percentile table
            st.markdown("#### Latency Percentiles")
            pct_df = (
                df_f.groupby(["backend", "quant"])[metric_sel]
                .quantile([0.5, 0.95, 0.99])
                .unstack()
                .round(4)
            )
            pct_df.columns = ["p50", "p95", "p99"]
            st.dataframe(pct_df, use_container_width=True)

        # Raw table (collapsible)
        with st.expander("Raw data"):
            st.dataframe(df_f, use_container_width=True)

    # Benchmark results
    bdf = load_all_benchmark_results()
    if not bdf.empty:
        st.divider()
        st.markdown("#### Task Benchmark Results")
        b1, b2 = st.columns(2)
        with b1:
            bench_type = st.selectbox("Benchmark", bdf["benchmark_name"].unique().tolist())
        bdf_f = bdf[bdf["benchmark_name"] == bench_type]
        avg_scores = bdf_f.groupby(["backend", "quant"])["score"].mean().reset_index()
        fig2 = px.bar(avg_scores, x="backend", y="score", color="quant", barmode="group",
                      title=f"{bench_type} — avg score by backend & quant",
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#c9d1d9")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    df = load_all_request_results()
    if df.empty:
        st.info("No results yet.")
    else:
        st.markdown("#### Side-by-side backend comparison")
        metrics_to_compare = [m for m, flag in [
            ("output_tps", do_tps), ("ttft", do_ttft), ("prefill_time", do_prefill),
            ("decode_time", do_decode), ("total_time", do_latency), ("memory_used_mb", do_memory)
        ] if flag]

        agg = df.groupby(["backend", "quant"])[metrics_to_compare].mean().reset_index().round(4)
        st.dataframe(agg, use_container_width=True)

        # Radar chart
        if len(metrics_to_compare) >= 3:
            st.markdown("#### Radar: normalized mean metrics")
            groups = df.groupby(["backend", "quant"])
            fig_radar = go.Figure()
            for (bname, q), gdf in groups:
                vals = [gdf[m].mean() for m in metrics_to_compare]
                # Normalize 0-1
                max_vals = [df[m].max() for m in metrics_to_compare]
                norm = [v / mx if mx else 0 for v, mx in zip(vals, max_vals)]
                norm += norm[:1]
                cats = metrics_to_compare + [metrics_to_compare[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=norm, theta=cats, fill="toself", name=f"{bname}/{q}"
                ))
            fig_radar.update_layout(
                polar=dict(bgcolor="#111827", radialaxis=dict(color="#6b7280")),
                paper_bgcolor="#0d1117", font_color="#c9d1d9",
                title="Normalized metric comparison (lower = faster for latency metrics)",
                title_font_color="#00d4ff",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sweep:
    from core.sweep import run_concurrency_sweep

    st.markdown("#### Concurrency Sweep — throughput vs latency curve")
    st.caption("Runs the same prompt at each concurrency level to find where each backend saturates.")

    # Re-check live backends for sweep
    sweep_backends = []
    for bname, bcls, host, port in selected_backends:
        instance = bcls(host=host, port=port, model=model_name)
        if instance.health_check():
            sweep_backends.append((bname, instance))

    if not sweep_backends:
        st.warning("No backends online.")
    else:
        st.info(f"Will sweep levels {concurrency_levels} × {sweep_requests_per_level} req/level across: {', '.join(b[0] for b in sweep_backends)}")
        sweep_btn = st.button("▶️ Run Sweep", type="primary", use_container_width=True)

        if sweep_btn:
            import pandas as pd
            sweep_rows = []
            params = GenerateParams(
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            for bname, backend in sweep_backends:
                st.markdown(f"**Sweeping `{bname}`...**")
                sweep_progress = st.progress(0)

                def _progress(done, total, bname=bname):
                    sweep_progress.progress(done / total, text=f"{bname}: level {done}/{total}")

                points = run_concurrency_sweep(
                    backend=backend,
                    params=params,
                    quant=quant,
                    concurrency_levels=concurrency_levels,
                    requests_per_level=sweep_requests_per_level,
                    progress_callback=_progress,
                )
                for pt in points:
                    s = pt.summary
                    sweep_rows.append({
                        "backend": bname,
                        "quant": quant,
                        "concurrency": pt.concurrency,
                        "mean_output_tps": s.mean_output_tps,
                        "ttft_p50_ms": s.ttft_p50 * 1000,
                        "ttft_p95_ms": s.ttft_p95 * 1000,
                        "total_latency_p50_ms": s.total_latency_p50 * 1000,
                        "total_latency_p95_ms": s.total_latency_p95 * 1000,
                        "mean_prefill_ms": s.mean_prefill_time * 1000,
                        "mean_decode_ms": s.mean_decode_time * 1000,
                        "peak_memory_gb": s.peak_memory_mb / 1024,
                        "error_rate": s.error_rate,
                    })
                sweep_progress.progress(1.0, text=f"{bname}: done")

            if sweep_rows:
                sdf = pd.DataFrame(sweep_rows)
                st.session_state["sweep_df"] = sdf

        # Display sweep results (persists after run)
        if "sweep_df" in st.session_state:
            sdf = st.session_state["sweep_df"]

            sweep_metric = st.selectbox(
                "Sweep metric to plot",
                ["mean_output_tps", "ttft_p50_ms", "ttft_p95_ms",
                 "total_latency_p50_ms", "total_latency_p95_ms",
                 "mean_prefill_ms", "mean_decode_ms", "peak_memory_gb"],
            )

            fig_sweep = px.line(
                sdf, x="concurrency", y=sweep_metric, color="backend",
                line_dash="quant", markers=True,
                title=f"{sweep_metric} vs concurrency level",
                template="plotly_dark",
                labels={"concurrency": "Concurrent requests", sweep_metric: sweep_metric},
            )
            fig_sweep.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#c9d1d9", title_font_color="#00d4ff",
                xaxis=dict(tickmode="array", tickvals=sdf["concurrency"].unique().tolist()),
            )
            st.plotly_chart(fig_sweep, use_container_width=True)

            # TPS vs latency tradeoff scatter
            st.markdown("#### Throughput vs Latency tradeoff")
            fig_scatter = px.scatter(
                sdf, x="total_latency_p50_ms", y="mean_output_tps",
                color="backend", size="concurrency", hover_data=["concurrency", "quant"],
                title="TPS vs p50 latency (bubble size = concurrency)",
                template="plotly_dark",
            )
            fig_scatter.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font_color="#c9d1d9", title_font_color="#00d4ff",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            with st.expander("Sweep raw data"):
                st.dataframe(sdf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:  # noqa: F811
    run_ids = list_run_ids()
    if not run_ids:
        st.info("No saved runs yet.")
    else:
        st.markdown(f"**{len(run_ids)} saved run(s)**")
        for rid in reversed(run_ids):
            st.code(rid, language=None)
        st.divider()
        st.markdown("Full historical data loads in **Results** tab automatically.")
