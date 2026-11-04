"""Task-only human think-aloud speech from the cleaned transcripts in data/think-aloud/."""
from collections import Counter
from pathlib import Path
import re
from .common import ROOT

SOURCES={'programming':ROOT/'data/think-aloud/programming_transcripts',
         'writing':ROOT/'data/think-aloud/writing_transcripts'}
# One utterance per line: "[0014 01:45.240 S2] text".
TURN=re.compile(r'^\[\d+ [\d:.]+ (?P<speaker>S\d+)\] ?(?P<text>.*)$')


def participant_turns(path):
    lines=[line for line in Path(path).read_text().splitlines() if line.strip()]
    turns=[TURN.match(line) for line in lines]
    if not all(turns):raise ValueError('Unrecognized line in cleaned transcript: '+str(path))
    # Cleaned writing files keep short researcher prompts, and diarization labels are
    # not consistent across sessions (P8 and P12 swap S1/S2); the participant is the
    # speaker with the most turns.
    participant=Counter(t['speaker'] for t in turns).most_common(1)[0][0] if turns else None
    return [t['text'].strip() for t in turns if t['speaker']==participant and t['text'].strip()]


def session_texts():
    result={}
    for domain,folder in SOURCES.items():
        paths=sorted(folder.glob('P*.full.txt'),key=lambda p:int(p.name[1:].split('.')[0]))
        result[domain]={p.name.split('.')[0]:texts for p in paths for texts in [participant_turns(p)] if texts}
        if not result[domain]:raise ValueError('No cleaned transcripts found in '+str(folder))
    return result
