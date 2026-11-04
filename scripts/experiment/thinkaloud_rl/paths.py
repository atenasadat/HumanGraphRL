"""Resolve the user's relocated data without rewriting original manifests."""
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/experiment').is_dir() and (p / 'data').is_dir())
STUDY = Path.home() / 'Desktop/thougthgraph_studyData'
LOCAL = STUDY / 'local_ThoughtGraphData'
CLOUD = Path.home() / 'Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/ThoughtGraph/data'
ANNOTATIONS = STUDY / 'annotations'
TRANSCRIPTS = ROOT / 'data/think-aloud/writing_transcripts'
PREVIOUS_RESULTS = STUDY / 'thinkaloud_rl'


def relocated(path):
    path = Path(path)
    pairs = [
        (STUDY/'raw_thinkaloud_review', TRANSCRIPTS),
        (Path.home()/'Documents/ThoughtGraphData', LOCAL),
        (Path.home()/'VSProjects/Jaam/CHI2027 Analysis/annotations', ANNOTATIONS),
        (ROOT/'data/raw_thinkaloud_review', TRANSCRIPTS),
        (ROOT/'pilots/results/legacy_runs/thinkaloud_rl', PREVIOUS_RESULTS),
    ]
    for old,new in pairs:
        try: relative = path.relative_to(old)
        except ValueError: continue
        candidate = new / relative
        if candidate.exists(): return candidate
    return path


def cloud_only(path):
    return bool(getattr(Path(path).stat(), 'st_flags', 0) & 0x40000000)
