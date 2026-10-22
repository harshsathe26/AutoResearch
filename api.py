import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AutoResearch API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=1)
jobs = {}
jobs_lock = threading.Lock()
active_run_lock = threading.Lock()

STAGES = [
    "RESEARCH PLANNER",
    "PAPER RETRIEVAL",
    "ITERATIVE RETRIEVAL",
    "GRAPH RAG",
    "LITERATURE ANALYSIS AGENT",
    "CLAIM VERIFICATION AGENT",
    "REPORT GENERATION",
    "EVALUATION",
]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)


def update_job(run_id, **changes):
    with jobs_lock:
        jobs[run_id].update(changes)


def run_research(run_id, question):
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    update_job(
        run_id,
        status="running",
        stage="Starting research",
        logs=[],
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        # Only one pipeline can run at a time because the existing
        # application writes to shared output/report paths.
        with active_run_lock:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "app.py")],
                cwd=str(ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Send the question to the existing CLI's input() call.
            process.stdin.write(question + "\n")
            process.stdin.flush()
            process.stdin.close()

            logs = []
            for line in process.stdout:
                line = line.rstrip()
                logs.append(line)

                stage = next(
                    (s for s in STAGES if s in line),
                    None,
                )
                changes = {"logs": logs[-500:]}
                if stage:
                    changes["stage"] = stage.title()
                update_job(run_id, **changes)

            return_code = process.wait()

            # Snapshot outputs into a per-run directory before another
            # job can replace the shared report files.
            for filename in ("report.md", "report.docx", "evaluation.json"):
                source = OUTPUT_DIR / filename
                if source.is_file():
                    shutil.copy2(source, run_dir / filename)

        papers = []
        seen = set()
        for line in logs:
            # The CLI prints retrieved papers as:
            # 1. Paper title (year)
            match = re.match(r"^\d+\.\s+(.+?)\s+\((.*?)\)$", line)
            if match:
                title, year = match.groups()
                if title not in seen:
                    seen.add(title)
                    papers.append({"title": title, "year": year})

        has_report = (run_dir / "report.md").is_file()
        status = "completed" if return_code == 0 and has_report else "failed"

        update_job(
            run_id,
            status=status,
            stage="Completed" if status == "completed" else "Failed",
            logs=logs[-500:],
            papers=papers,
            report_files=[
                name
                for name in ("report.md", "report.docx", "evaluation.json")
                if (run_dir / name).is_file()
            ],
            error=None if status == "completed" else (
                f"Pipeline exited with code {return_code}, "
                "or no Markdown report was generated. Check the logs."
            ),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as exc:
        update_job(
            run_id,
            status="failed",
            stage="Failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AutoResearch API"}


@app.post("/api/research", status_code=202)
def start_research(request: ResearchRequest):
    question = request.question.strip()
    if len(question) < 5:
        raise HTTPException(status_code=422, detail="Enter a longer research question.")

    run_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[run_id] = {
            "id": run_id,
            "question": question,
            "status": "queued",
            "stage": "Queued",
            "logs": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "papers": [],
            "report_files": [],
            "error": None,
        }

    executor.submit(run_research, run_id, question)
    return {"id": run_id, "status": "queued"}


@app.get("/api/research")
def list_research():
    with jobs_lock:
        return sorted(
            jobs.values(),
            key=lambda job: job["created_at"],
            reverse=True,
        )


@app.get("/api/research/{run_id}")
def get_research(run_id: str):
    with jobs_lock:
        job = jobs.get(run_id)
        if not job:
            raise HTTPException(status_code=404, detail="Research run not found.")
        return dict(job)


@app.get("/api/papers")
def list_papers():
    results = []
    with jobs_lock:
        completed_jobs = list(jobs.values())

    for job in completed_jobs:
        if job["status"] == "completed":
            for paper in job.get("papers", []):
                results.append({
                    **paper,
                    "run_id": job["id"],
                    "question": job["question"],
                })
    return results


@app.get("/api/reports")
def list_reports():
    reports = []
    with jobs_lock:
        completed_jobs = list(jobs.values())

    for job in completed_jobs:
        run_dir = RUNS_DIR / job["id"]
        for filename in ("report.md", "report.docx", "evaluation.json"):
            path = run_dir / filename
            if path.is_file():
                reports.append({
                    "run_id": job["id"],
                    "question": job["question"],
                    "filename": filename,
                    "url": f"/api/reports/{job['id']}/{filename}",
                    "created_at": job["created_at"],
                })
    return reports


@app.get("/api/reports/{run_id}/{filename}")
def download_report(run_id: str, filename: str):
    if filename not in {"report.md", "report.docx", "evaluation.json"}:
        raise HTTPException(status_code=400, detail="Unsupported report file.")

    path = (RUNS_DIR / run_id / filename).resolve()
    if not path.is_relative_to(RUNS_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")

    return FileResponse(path, filename=filename)
