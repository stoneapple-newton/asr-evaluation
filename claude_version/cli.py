"""CLI for computing and comparing Sentence Similarity Score runs."""

from pathlib import Path
from textwrap import dedent
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from claude_version.aligner import align_chunks
from claude_version.chunker import chunk_text
from claude_version.config import (
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_CHARS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TARGET_SENTENCES,
    GROUND_TRUTH_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TRANSCRIPTION_DIR,
)
from claude_version.embedder import OllamaEmbedder
from claude_version.models import RunMetadata, Segment, SSSResult
from claude_version.scorer import cosine_similarity, total_sss
from claude_version.storage import (
    ensure_dirs,
    find_latest_result,
    history_rows,
    list_ids,
    list_result_paths,
    load_result,
    load_run_metadata,
    load_text,
    save_result,
    save_run_metadata,
    utc_now,
)

app = typer.Typer(
    help="claude-sss: evaluate transcription similarity against a ground truth.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init() -> None:
    """Create data folders for ground_truth, transcriptions, runs, and results."""
    ensure_dirs()
    console.print(f"[dim]ground_truth :[/dim] {GROUND_TRUTH_DIR}")
    console.print(f"[dim]transcriptions:[/dim] {TRANSCRIPTION_DIR}")
    console.print(f"[dim]runs          :[/dim] {RUNS_DIR}")
    console.print(f"[dim]results       :[/dim] {RESULTS_DIR}")


@app.command("new-run")
def new_run(
    run_id: Annotated[str, typer.Option("--run-id", "-r", help="Unique run ID, e.g. meeting_v1")],
    ground_truth_id: Annotated[str, typer.Option("--gt", help="Ground truth file stem (no .txt)")],
    transcription_id: Annotated[str, typer.Option("--tx", help="Transcription file stem (no .txt)")],
    model: Annotated[str, typer.Option("--model", help="Ollama embedding model")] = DEFAULT_EMBED_MODEL,
    location: Annotated[str | None, typer.Option("--location", help="Where the audio was recorded")] = None,
    notes: Annotated[str | None, typer.Option("--notes", help="Free-form notes about this version")] = None,
    strategy: Annotated[str, typer.Option("--strategy", help="sentence_window | sentence | paragraph")] = DEFAULT_CHUNK_STRATEGY,
) -> None:
    """Scaffold a YAML metadata file for one transcription version.

    Edit the generated YAML to fill in ASR settings before running.
    """
    ensure_dirs()
    yaml_path = RUNS_DIR / f"{run_id}.yaml"
    if yaml_path.exists():
        console.print(f"[yellow]Already exists:[/yellow] {yaml_path}")
        raise typer.Exit(1)

    content = dedent(f"""\
        # SSS run metadata — edit before running
        run_id: {run_id}
        ground_truth_id: {ground_truth_id}
        transcription_id: {transcription_id}

        # Embedding
        model: {model}
        chunk_strategy: {strategy}   # sentence_window | sentence | paragraph
        target_sentences_per_chunk: {DEFAULT_TARGET_SENTENCES}
        max_chars_per_chunk: {DEFAULT_MAX_CHARS}

        # Context
        location: {location or 'null'}       # e.g. conference_room_A, remote_call
        recorded_at: null                    # ISO8601 timestamp of the recording
        notes: {notes or 'null'}

        # ASR / transcription settings — fill in as needed
        settings:
          asr_model: ''           # e.g. whisper-large-v3
          language: ''            # e.g. en
          prompt: ''              # prompt given to ASR, if any
          audio_source: ''        # e.g. microphone, phone_call, video_conference
    """)
    yaml_path.write_text(content, encoding="utf-8")
    console.print(f"[green]Created[/green] {yaml_path}")
    console.print("[dim]Edit the settings block before running.[/dim]")


@app.command()
def run(
    run_id: Annotated[str, typer.Argument(help="Run ID (matches a file in data/runs/)")],
    ollama_url: Annotated[str, typer.Option("--ollama-url", help="Ollama base URL")] = DEFAULT_OLLAMA_URL,
    model: Annotated[str | None, typer.Option("--model", help="Override metadata model")] = None,
) -> None:
    """Run SSS evaluation for a metadata file and save the result."""
    metadata = load_run_metadata(run_id)
    embed_model = model or metadata.model

    gt_text = load_text(GROUND_TRUTH_DIR, metadata.ground_truth_id)
    tx_text = load_text(TRANSCRIPTION_DIR, metadata.transcription_id)

    gt_chunks = chunk_text(
        gt_text,
        strategy=metadata.chunk_strategy,
        target_sentences=metadata.target_sentences_per_chunk,
        max_chars=metadata.max_chars_per_chunk,
    )
    tx_chunks = chunk_text(
        tx_text,
        strategy=metadata.chunk_strategy,
        target_sentences=metadata.target_sentences_per_chunk,
        max_chars=metadata.max_chars_per_chunk,
    )
    aligned = align_chunks(gt_chunks, tx_chunks)

    console.print(
        f"GT chunks: [cyan]{len(gt_chunks)}[/cyan]  "
        f"TX chunks: [cyan]{len(tx_chunks)}[/cyan]  "
        f"Aligned pairs: [cyan]{len(aligned)}[/cyan]"
    )

    embedder = OllamaEmbedder(base_url=ollama_url, model=embed_model)
    gt_texts = [pair[0] for pair in aligned]
    tx_texts = [pair[1] for pair in aligned]

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Embedding with [bold]{embed_model}[/bold]", total=len(aligned) * 2)
        gt_embeddings = []
        for text in gt_texts:
            gt_embeddings.append(embedder._embed_one(text))
            progress.advance(task)
        tx_embeddings = []
        for text in tx_texts:
            tx_embeddings.append(embedder._embed_one(text))
            progress.advance(task)

    import numpy as np

    scores = [
        cosine_similarity(np.asarray(g, dtype=np.float32), np.asarray(t, dtype=np.float32))
        for g, t in zip(gt_embeddings, tx_embeddings)
    ]
    weights = [max(len(gt), len(tx)) for gt, tx in aligned]

    segments = [
        Segment(
            index=i + 1,
            ground_truth_text=gt,
            transcription_text=tx,
            gt_chars=len(gt),
            tx_chars=len(tx),
            sss=round(score, 6),
        )
        for i, ((gt, tx), score) in enumerate(zip(aligned, scores))
    ]

    result = SSSResult(
        run_id=metadata.run_id,
        ground_truth_id=metadata.ground_truth_id,
        transcription_id=metadata.transcription_id,
        model=embed_model,
        chunk_strategy=metadata.chunk_strategy,
        target_sentences_per_chunk=metadata.target_sentences_per_chunk,
        max_chars_per_chunk=metadata.max_chars_per_chunk,
        recorded_at=utc_now(),
        total_sss=round(total_sss(scores, weights), 6),
        segment_count=len(segments),
        metadata=metadata.model_dump(mode="json"),
        segments=segments,
    )

    path = save_result(result)
    console.print(f"[bold green]Total SSS:[/bold green] {result.total_sss:.6f}  [dim]({result.segment_count} segments)[/dim]")
    console.print(f"[dim]Saved:[/dim] {path}")


@app.command("list")
def list_artifacts(
    kind: Annotated[
        str,
        typer.Argument(help="ground_truth | transcriptions | runs | results"),
    ] = "results",
) -> None:
    """List stored inputs, run metadata files, or results."""
    kind_map = {
        "ground_truth": (GROUND_TRUTH_DIR, ".txt"),
        "transcriptions": (TRANSCRIPTION_DIR, ".txt"),
        "runs": (RUNS_DIR, ".yaml"),
    }
    if kind in kind_map:
        directory, suffix = kind_map[kind]
        for item in list_ids(directory, suffix):
            console.print(item)
    elif kind == "results":
        for path in list_result_paths():
            console.print(path.name)
    else:
        raise typer.BadParameter("kind must be one of: ground_truth, transcriptions, runs, results")


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run ID to display")],
) -> None:
    """Show segment-level SSS table for the latest result of a run."""
    result = find_latest_result(run_id)
    table = Table(
        title=f"[bold]{result.run_id}[/bold]  total SSS = {result.total_sss:.6f}",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("SSS", justify="right", style="magenta", no_wrap=True)
    table.add_column("GT chars", justify="right", style="dim", no_wrap=True)
    table.add_column("TX chars", justify="right", style="dim", no_wrap=True)
    table.add_column("Ground Truth", style="green")
    table.add_column("Transcription", style="yellow")

    for seg in result.segments:
        sss_color = "green" if seg.sss >= 0.8 else "yellow" if seg.sss >= 0.5 else "red"
        table.add_row(
            str(seg.index),
            f"[{sss_color}]{seg.sss:.4f}[/{sss_color}]",
            str(seg.gt_chars),
            str(seg.tx_chars),
            _preview(seg.ground_truth_text),
            _preview(seg.transcription_text),
        )
    console.print(table)


@app.command()
def history(
    gt: Annotated[str | None, typer.Option("--gt", help="Filter by ground truth ID")] = None,
    tx: Annotated[str | None, typer.Option("--tx", help="Filter by transcription ID")] = None,
) -> None:
    """Show all historic runs sorted by recorded time."""
    rows = history_rows()
    if gt:
        rows = [r for r in rows if r["ground_truth_id"] == gt]
    if tx:
        rows = [r for r in rows if r["transcription_id"] == tx]

    if not rows:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title="SSS History", show_lines=False)
    table.add_column("Run ID", style="cyan")
    table.add_column("Ground Truth", style="green")
    table.add_column("Transcription", style="yellow")
    table.add_column("Recorded At", style="dim")
    table.add_column("Model", style="dim")
    table.add_column("Strategy", style="dim")
    table.add_column("Segs", justify="right", style="dim")
    table.add_column("Total SSS", justify="right", style="magenta")

    for row in sorted(rows, key=lambda r: r["recorded_at"]):
        sss_color = "green" if row["total_sss"] >= 0.8 else "yellow" if row["total_sss"] >= 0.5 else "red"
        table.add_row(
            row["run_id"],
            row["ground_truth_id"],
            row["transcription_id"],
            row["recorded_at"].strftime("%Y-%m-%d %H:%M"),
            row["model"],
            row["chunk_strategy"],
            str(row["segments"]),
            f"[{sss_color}]{row['total_sss']:.6f}[/{sss_color}]",
        )
    console.print(table)


@app.command()
def compare(
    baseline: Annotated[str, typer.Argument(help="Baseline run ID")],
    candidate: Annotated[str, typer.Argument(help="Candidate run ID")],
) -> None:
    """Compare two runs segment-by-segment and show deltas."""
    base_result = find_latest_result(baseline)
    cand_result = find_latest_result(candidate)

    table = Table(
        title=f"[bold]{baseline}[/bold] vs [bold]{candidate}[/bold]",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="cyan")
    table.add_column(f"{baseline}", justify="right", style="green")
    table.add_column(f"{candidate}", justify="right", style="yellow")
    table.add_column("Delta", justify="right")

    max_segs = max(len(base_result.segments), len(cand_result.segments))
    for i in range(max_segs):
        base_sss = base_result.segments[i].sss if i < len(base_result.segments) else 0.0
        cand_sss = cand_result.segments[i].sss if i < len(cand_result.segments) else 0.0
        delta = cand_sss - base_sss
        delta_color = "green" if delta > 0.001 else "red" if delta < -0.001 else "dim"
        table.add_row(
            str(i + 1),
            f"{base_sss:.6f}",
            f"{cand_sss:.6f}",
            f"[{delta_color}]{delta:+.6f}[/{delta_color}]",
        )

    console.print(table)

    total_delta = cand_result.total_sss - base_result.total_sss
    delta_color = "green" if total_delta > 0.001 else "red" if total_delta < -0.001 else "dim"
    console.print(
        f"\n[bold]Total SSS[/bold]  "
        f"{baseline}=[green]{base_result.total_sss:.6f}[/green]  "
        f"{candidate}=[yellow]{cand_result.total_sss:.6f}[/yellow]  "
        f"delta=[{delta_color}]{total_delta:+.6f}[/{delta_color}]"
    )


def _preview(text: str, width: int = 70) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= width else f"{normalized[:width - 3]}..."


if __name__ == "__main__":
    app()
