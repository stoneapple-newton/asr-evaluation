"""Typer CLI for the SSS evaluation system."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from kimi.aligner import align
from kimi.chunker import chunk
from kimi.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL
from kimi.embedder import OllamaEmbedder
from kimi.history import diff_results, get_result, load_all_results
from kimi.models import RunMeta, Segment, SSSResult
from kimi.scorer import score_segments, total_sss
from kimi.storage import (
    find_result_by_run_id,
    list_ground_truths,
    list_results,
    list_runs,
    list_transcriptions,
    load_ground_truth,
    load_result,
    load_run_meta,
    load_transcription,
    save_result,
)

app = typer.Typer(help="SSS: Sentence Similarity Score evaluation CLI")
console = Console()


@app.command()
def run(
    ground_truth: str = typer.Option(..., "--gt", help="Ground-truth ID (file stem)"),
    transcription: str = typer.Option(..., "--tx", help="Transcription ID (file stem)"),
    meta: Path = typer.Option(..., "--meta", help="Path to run metadata YAML"),
    model: Optional[str] = typer.Option(None, "--model", help="Ollama model override"),
    strategy: str = typer.Option("sentence", "--strategy", help="Chunk strategy: sentence | paragraph"),
    ollama_url: str = typer.Option(OLLAMA_BASE_URL, "--ollama-url", help="Ollama base URL"),
):
    """Execute an SSS evaluation run."""
    console.print(f"[bold green]Loading metadata[/bold green] {meta}")
    run_meta = load_run_meta(meta.stem)

    console.print(f"[bold green]Loading texts[/bold green]")
    gt_text = load_ground_truth(ground_truth)
    tx_text = load_transcription(transcription)

    console.print(f"[bold green]Chunking[/bold green] (strategy={strategy})")
    gt_chunks = chunk(gt_text, strategy=strategy)
    tx_chunks = chunk(tx_text, strategy=strategy)
    console.print(f"  GT chunks: {len(gt_chunks)}  TX chunks: {len(tx_chunks)}")

    console.print(f"[bold green]Aligning chunks[/bold green]")
    aligned = align(gt_chunks, tx_chunks)
    console.print(f"  Aligned pairs: {len(aligned)}")

    embed_model = model or run_meta.model or OLLAMA_EMBED_MODEL
    embedder = OllamaEmbedder(base_url=ollama_url, model=embed_model)

    console.print(f"[bold green]Embedding[/bold green] with {embed_model}")
    gt_embs = embedder.embed([a[0] for a in aligned])
    tx_embs = embedder.embed([a[1] for a in aligned])

    console.print(f"[bold green]Scoring[/bold green]")
    scores = score_segments(aligned, gt_embs, tx_embs)
    lengths = [len(a[1]) for a in aligned]
    total = total_sss(scores, lengths)

    segments = [
        Segment(
            segment_index=i,
            ground_truth_text=gt,
            transcription_text=tx,
            sss=round(s, 4),
        )
        for i, ((gt, tx), s) in enumerate(zip(aligned, scores))
    ]

    result = SSSResult(
        run_id=run_meta.run_id,
        ground_truth_id=ground_truth,
        transcription_id=transcription,
        model=embed_model,
        chunk_strategy=strategy,
        recorded_at=datetime.utcnow(),
        segments=segments,
        total_sss=round(total, 4),
        metadata=run_meta.model_dump(),
    )

    path = save_result(result)
    console.print(f"[bold green]Result saved[/bold green] {path}")
    console.print(f"[bold cyan]Total SSS:[/bold cyan] {result.total_sss:.4f}")


@app.command("list")
def list_(
    kind: str = typer.Argument("runs", help="What to list: runs | results | ground_truths | transcriptions"),
):
    """List available artifacts."""
    if kind == "runs":
        items = list_runs()
    elif kind == "results":
        items = [p.name for p in list_results()]
    elif kind == "ground_truths":
        items = list_ground_truths()
    elif kind == "transcriptions":
        items = list_transcriptions()
    else:
        console.print(f"[red]Unknown kind: {kind}[/red]")
        raise typer.Exit(1)

    for it in items:
        console.print(it)


@app.command()
def show(run_id: str):
    """Display segment table and total score for a run."""
    result = get_result(run_id)
    table = Table(title=f"Run: {result.run_id} | Total SSS: {result.total_sss:.4f}")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Ground Truth", style="green")
    table.add_column("Transcription", style="yellow")
    table.add_column("SSS", justify="right", style="magenta")

    for seg in result.segments:
        table.add_row(
            str(seg.segment_index),
            seg.ground_truth_text[:60] + ("..." if len(seg.ground_truth_text) > 60 else ""),
            seg.transcription_text[:60] + ("..." if len(seg.transcription_text) > 60 else ""),
            f"{seg.sss:.4f}",
        )
    console.print(table)


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID"),
    run_b: str = typer.Argument(..., help="Second run ID"),
):
    """Compare two runs side-by-side."""
    a = get_result(run_a)
    b = get_result(run_b)
    diffs = diff_results(a, b)

    table = Table(title=f"Compare: {run_a} vs {run_b}")
    table.add_column("#", justify="right", style="cyan")
    table.add_column(f"{run_a} SSS", justify="right", style="green")
    table.add_column(f"{run_b} SSS", justify="right", style="yellow")
    table.add_column("Delta", justify="right", style="magenta")

    for i, sa, sb, delta in diffs:
        delta_str = f"{delta:+.4f}"
        style = "red" if delta < 0 else "green" if delta > 0 else "white"
        table.add_row(str(i), f"{sa:.4f}", f"{sb:.4f}", f"[{style}]{delta_str}[/{style}]")

    console.print(table)
    console.print(f"[bold]Total SSS[/bold]  {run_a}: {a.total_sss:.4f}  {run_b}: {b.total_sss:.4f}")


if __name__ == "__main__":
    app()
