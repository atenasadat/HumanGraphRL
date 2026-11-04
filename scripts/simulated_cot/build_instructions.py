"""Assemble the task instructions each think-aloud participant worked on.

Writes, under data/ (gitignored):
  writing_instructions/P{n}.txt       proposal-writing study, P4-P17, exact prompt shown by the study app
  programming_instructions/P{n}.txt   in-the-wild programming study, P1-P14, goal reconstructed from
                                      the developer's own speech and approved by the researcher
  */manifest.json                     source, sha256 and verification status per prompt
and copies each participant's human think-aloud into data/Simulated CoT data/ as
  {domain}_P{n}_thinkAloud.txt

Writing sources live outside this repository on the study machine; their paths are recorded
in the manifest. Run once locally, then copy data/ to CARC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
OUT = DATA / 'Simulated CoT data'
HOME = Path.home()
WRITING_APP = HOME / 'VSProjects/ThoughtGraph/thoughtgraph-ui/src/app/Workspace.tsx'
WRITING_FILES = HOME / 'VSProjects/ThoughtGraph/data/task_descriptions'
SESSIONS = HOME / 'Desktop/thougthgraph_studyData/local_ThoughtGraphData'
TRANSCRIPTS = DATA / 'think-aloud'
WRITING_PIDS = [f'P{n}' for n in range(4, 18)]
PROGRAMMING_PIDS = [f'P{n}' for n in range(1, 15)]

# Evaluates the app's own prompt-assembly code so the prompt is not re-typed by hand.
RENDER_JS = r"""
const fs=require('fs');const src=fs.readFileSync(process.argv[1],'utf8');
const a=src.indexOf('const DEFAULT_TASK_DIRECTION');const b=src.indexOf('const USER_PHASE_INSTRUCTIONS');
const code=src.slice(a,b).replace(/\] as const;/,'];').replace(/\(subjectId\?: string \| null\)/,'(subjectId)')
  .replace(/\(taskDirection: string\)/,'(taskDirection)');
eval(code+';globalThis.render=(p)=>taskDescriptionTextForDirection(taskDirectionForSubjectId(p));');
const out={};for(const p of JSON.parse(process.argv[2]))out[p]=render(p);process.stdout.write(JSON.stringify(out));
"""


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def recorded_writing_prompts():
    """Every 'Task Description:' string the study app logged, mapped to the files recording it."""
    found = {}
    def walk(o, source):
        if isinstance(o, dict):
            for v in o.values(): walk(v, source)
        elif isinstance(o, list):
            for v in o: walk(v, source)
        elif isinstance(o, str) and 'Topic direction:' in o and 'Task Description:' in o:
            found.setdefault(o[o.index('Task Description:'):], set()).add(source)
    # Session manifests, saved session states and live event logs hold the prompt;
    # audio/screen folders are skipped.
    candidates = [*SESSIONS.glob('.sessions/*.json'), *SESSIONS.glob('*/manifest.json'),
                  *SESSIONS.glob('*/*_manifest.json'), *SESSIONS.glob('*/real_time_events.jsonl')]
    for path in sorted(candidates):
        if 'test' in path.name.lower():
            continue
        source = str(path.relative_to(SESSIONS))
        with open(path, errors='ignore') as handle:
            if path.suffix == '.json':
                walk(json.load(handle), source)
                continue
            # Event logs reach 13 GB; the prompt is logged when the session starts.
            for line in handle:
                if 'Topic direction:' in line:
                    walk(json.loads(line), source)
                    break
    return found


def build_writing():
    rendered = json.loads(subprocess.run(['node', '-e', RENDER_JS, str(WRITING_APP), json.dumps(WRITING_PIDS)],
                                         check=True, capture_output=True, text=True).stdout)
    recorded = recorded_writing_prompts()
    manifest = []
    for pid in WRITING_PIDS:
        text = rendered[pid]
        if (WRITING_FILES / f'{pid}_task.txt').read_text().rstrip('\n') != text:
            raise ValueError(f'{pid}: task_descriptions file differs from Workspace.tsx rendering')
        sessions = sorted(s for t, srcs in recorded.items() if t == text for s in srcs)
        conflicting = sorted(s for t, srcs in recorded.items() if t != text
                             for s in srcs if s.lower().startswith((pid.lower() + '_', pid.lower() + '/')))
        if conflicting:
            raise ValueError(f'{pid}: recorded session prompt differs from rendering: {conflicting}')
        write(DATA / 'writing_instructions' / f'{pid}.txt', text)
        manifest.append({'subject': pid, 'file': f'{pid}.txt', 'sha256': sha(text),
                         'source': str(WRITING_APP), 'extracted_copy': str(WRITING_FILES / f'{pid}_task.txt'),
                         'status': 'verified_against_recorded_session' if sessions else 'app_code_only_not_recorded',
                         'recorded_in': sessions})
        write(OUT / f'writing_{pid}_thinkAloud.txt', (TRANSCRIPTS / 'writing_transcripts' / f'{pid}.full.txt').read_text())
    write(DATA / 'writing_instructions' / 'manifest.json', json.dumps(
        {'study': 'proposal writing (ThoughtGraph)', 'prompt': 'Task description only; the on-screen phase '
         'instructions (brainstorm/draft/revise) are not included.', 'subjects': manifest}, indent=1))
    return manifest


def build_programming(approved):
    entries = {e['pid']: e for e in json.loads(Path(approved).read_text())}
    if sorted(entries, key=lambda p: int(p[1:])) != PROGRAMMING_PIDS:
        raise ValueError('Approved prompts must cover exactly ' + ', '.join(PROGRAMMING_PIDS))
    manifest = []
    for pid in PROGRAMMING_PIDS:
        e = entries[pid]
        if e.get('approved') is not True:
            raise ValueError(f'{pid}: reconstructed prompt not approved by the researcher')
        text = e['prompt'].strip()
        write(DATA / 'programming_instructions' / f'{pid}.txt', text)
        manifest.append({'subject': pid, 'file': f'{pid}.txt', 'sha256': sha(text),
                         'status': 'reconstructed_from_participant_speech_researcher_approved',
                         'evidence': e['evidence'], 'other_goals': e.get('other_goals', []),
                         'confidence': e['confidence'], 'omitted': e.get('omitted', '')})
        write(OUT / f'programming_{pid}_thinkAloud.txt',
              (TRANSCRIPTS / 'programming_transcripts' / f'{pid}.full.txt').read_text())
    write(DATA / 'programming_instructions' / 'manifest.json', json.dumps(
        {'study': 'in-the-wild programming study (developers on their own projects)', 'prompt': 'No task '
         'statement existed; each prompt restates the main goal the developer voiced, without their '
         'codebase or credentials.', 'approved_source': str(approved), 'subjects': manifest}, indent=1))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--programming-approved', type=Path,
                   help='Researcher-approved JSON of the reconstructed P1-P14 goals')
    args = p.parse_args()
    w = build_writing()
    print(f'writing: {len(w)} prompts', {s: sum(r["status"] == s for r in w) for s in {r["status"] for r in w}})
    if args.programming_approved:
        print(f'programming: {len(build_programming(args.programming_approved))} prompts')
