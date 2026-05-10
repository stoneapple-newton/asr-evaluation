"""CLI for computing and comparing Sentence Similarity Score runs."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from codex_version.aligner import align_chunks
from codex_version.chunker import chunk_text
from codex_version.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_URL,
    GROUND_TRUTH_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TRANSCRIPTION_DIR,
)
from codex_version.embedder import OllamaEmbedder
from codex_version.models import AlignmentSegment, RunMetadata, SSSResult
from codex_version.scorer import cosine_similarity, total_sss
from codex_version.storage import (
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

app = typer.Typer(help="SSS: evaluate transcription similarity against a ground truth.")
console = Console()


@app.command()
def init() -> None:
    """Create the data folders used by this version."""
    ensure_dirs()
    console.print(f"Ground truth: {GROUND_TRUTH_DIR}")
    console.print(f"Transcriptions: {TRANSCRIPTION_DIR}")
    console.print(f"Run metadata: {RUNS_DIR}")
    console.print(f"Results: {RESULTS_DIR}")


@app.command("new-run")
def new_run(
    run_id: Annotated[str, typer.Option("--run-id", help="Unique run/version ID.")],
    ground_truth_id: Annotated[str, typer.Option("--gt", help="Ground truth file stem.")],
    transcription_id: Annotated[str, typer.Option("--tx", help="Transcription file stem.")],
    model: Annotated[str, typer.Option("--model", help="Ollama embedding model.")] = DEFAULT_EMBED_MODEL,
    location: Annotated[str | None, typer.Option("--location", help="Where the transcription came from.")] = None,
) -> None:
    """Create a YAML metadata file for one transcription version."""
    metadata = RunMetadata(
        run_id=run_id,
        ground_truth_id=ground_truth_id,
        transcription_id=transcription_id,
        model=model,
        location=location,
        recorded_at=utc_now(),
        settings={"source": "", "asr_model": "", "language": "", "prompt": ""},
        notes="",
    )
    path = save_run_metadata(metadata)
    console.print(f"Created {path}")


@app.command()
def run(
    run_id: Annotated[str, typer.Argument(help="Run metadata ID from data/runs/<run_id>.yaml.")],
    ollama_url: Annotated[str, typer.Option("--ollama-url", help="Ollama base URL.")] = DEFAULT_OLLAMA_URL,
    model: Annotated[str | None, typer.Option("--model", help="Override metadata model.")] = None,
) -> None:
    """Run SSS for one metadata file and save a historic result JSON."""
    metadata = load_run_metadata(run_id)
    embed_model = model or metadata.model
    gt_text = load_text(GROUND_TRUTH_DIR, metadata.ground_truth_id)
    tx_text = load_text(TRANSCRIPTION_DIR, metadata.transcription_id)

    gt_chunks = chunk_text(
        gt_text,
        metadata.chunk_strategy,
        metadata.target_sentences_per_chunk,
        metadata.max_chars_per_chunk,
    )
    tx_chunks = chunk_text(
        tx_text,
        metadata.chunk_strategy,
        metadata.target_sentences_per_chunk,
        metadata.max_chars_per_chunk,
    )
    aligned = align_chunks(gt_chunks, tx_chunks)

    console.print(f"Chunks: ground_truth={len(gt_chunks)} transcription={len(tx_chunks)} aligned={len(aligned)}")
    console.print(f"Embedding with Ollama model: {embed_model}")

    embedder = OllamaEmbedder(base_url=ollama_url, model=embed_model)
    gt_embeddings = embedder.embed([pair[0] for pair in aligned])
    tx_embeddings = embedder.embed([pair[1] for pair in aligned])
    scores = [cosine_similarity(gt, tx) for gt, tx in zip(gt_embeddings, tx_embeddings)]
    weights = [max(len(gt), len(tx)) for gt, tx in aligned]

    segments = [
        AlignmentSegment(
            segment_index=index,
            ground_truth_text=gt,
            transcription_text=tx,
            ground_truth_chars=len(gt),
            transcription_chars=len(tx),
            sss=round(score, 6),
        )
        for index, ((gt, tx), score) in enumerate(zip(aligned, scores), start=1)
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
    console.print(f"Total SSS: {result.total_sss:.6f}")
    console.print(f"Saved: {path}")


@app.command("list")
def list_artifacts(
    kind: Annotated[
        str,
        typer.Argument(help="One of: ground_truth, transcriptions, runs, results"),
    ] = "results",
) -> None:
    """List stored inputs, run metadata, or results."""
    if kind == "ground_truth":
        items = list_ids(GROUND_TRUTH_DIR, ".txt")
    elif kind == "transcriptions":
        items = list_ids(TRANSCRIPTION_DIR, ".txt")
    elif kind == "runs":
        items = list_ids(RUNS_DIR, ".yaml")
    elif kind == "results":
        items = [path.name for path in list_result_paths()]
    else:
        raise typer.BadParameter("kind must be one of: ground_truth, transcriptions, runs, results")
    for item in items:
        console.print(item)


@app.command()
def history(
    ground_truth_id: Annotated[str | None, typer.Option("--gt", help="Filter by ground truth ID.")] = None,
) -> None:
    """Show historic run totals so transcription versions can be compared."""
    rows = [
        row for row in history_rows()
        if ground_truth_id is None or row["ground_truth_id"] == ground_truth_id
    ]
    table = Table(title="SSS History")
    table.add_column("Run")
    table.add_column("Ground Truth")
    table.add_column("Transcription")
    table.add_column("Recorded At")
    table.add_column("Model")
    table.add_column("Segments", justify="right")
    table.add_column("Total SSS", justify="right")
    for row in rows:
        table.add_row(
            row["run_id"],
            row["ground_truth_id"],
            row["transcription_id"],
            row["recorded_at"].isoformat(),
            row["model"],
            str(row["segments"]),
            f"{row['total_sss']:.6f}",
        )
    console.print(table)


@app.command()
def show(run_id: Annotated[str, typer.Argument(help="Run ID to display.")]) -> None:
    """Show segment-level SSS for the latest result of a run."""
    result = find_latest_result(run_id)
    table = Table(title=f"{result.run_id} total={result.total_sss:.6f}")
    table.add_column("#", justify="right")
    table.add_column("SSS", justify="right")
    table.add_column("Ground Truth")
    table.add_column("Transcription")
    for segment in result.segments:
        table.add_row(
            str(segment.segment_index),
            f"{segment.sss:.6f}",
            _preview(segment.ground_truth_text),
            _preview(segment.transcription_text),
        )
    console.print(table)


@app.command()
def compare(
    baseline_run_id: Annotated[str, typer.Argument(help="Baseline run ID.")],
    candidate_run_id: Annotated[str, typer.Argument(help="Candidate run ID.")],
) -> None:
    """Compare two run results and show segment deltas."""
    baseline = find_latest_result(baseline_run_id)
    candidate = find_latest_result(candidate_run_id)
    table = Table(title=f"{baseline.run_id} vs {candidate.run_id}")
    table.add_column("#", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")
    max_segments = max(len(baseline.segments), len(candidate.segments))
    for index in range(max_segments):
        base_score = baseline.segments[index].sss if index < len(baseline.segments) else 0.0
        cand_score = candidate.segments[index].sss if index < len(candidate.segments) else 0.0
        table.add_row(
            str(index + 1),
            f"{base_score:.6f}",
            f"{cand_score:.6f}",
            f"{cand_score - base_score:+.6f}",
        )
    console.print(table)
    console.print(
        f"Total SSS: {baseline.run_id}={baseline.total_sss:.6f} "
        f"{candidate.run_id}={candidate.total_sss:.6f} "
        f"delta={candidate.total_sss - baseline.total_sss:+.6f}"
    )


def _preview(text: str, width: int = 80) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= width else f"{normalized[: width - 3]}..."


if __name__ == "__main__":
    app()

