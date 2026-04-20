from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(r"C:\Users\Yue\Desktop\eva_data")
MANIFEST = ROOT / "evaluation_release" / "supporting_files" / "manifest.json"
GEN_ROOT = ROOT / "generated_code"
OUT_JSON = Path(r"C:\Users\Yue\.openclaw\workspace\computed_results_metrics.json")
OUT_MD = Path(r"C:\Users\Yue\.openclaw\workspace\computed_results_metrics.md")

STOPWORDS = {
    'the','a','an','and','or','of','to','in','on','for','with','by','from','is','are','was','were','be','being','been',
    'this','that','these','those','it','its','as','at','into','than','then','their','there','such','can','could','should',
    'would','will','may','might','about','across','within','without','between','through','during','using','used','use',
    'what','which','who','whom','why','how','when','where','all','any','some','more','most','other','another','one','two',
    'three','four','five','six','seven','eight','nine','ten','only','than','also','very','much','many','each','per',
    'understanding','introduction','guide','comprehensive','quiz','title','subtitle','content','conclusion','example',
    'key','concept','concepts','used','helps','help','various','include','including','based','understand','understanding',
    'machine','learning','data','model','models','method','methods'
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\+²^]*")


def tokenize(text: str):
    toks = []
    for t in TOKEN_RE.findall(text.lower()):
        t = t.strip('-')
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        toks.append(t)
    return toks


def keyword_set(text: str):
    return set(tokenize(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def collect_slide_text(student_input: dict, storyboard: dict | None):
    parts = []
    for s in student_input.get('slides', []):
        for key in ('title', 'subtitle', 'content'):
            v = s.get(key)
            if isinstance(v, str):
                parts.append(v)
        for item in s.get('visual_assets_summary', []) or []:
            if isinstance(item, str):
                parts.append(item)
    if storyboard:
        for s in storyboard.get('slides', []):
            for key in ('title', 'subtitle', 'content'):
                v = s.get(key)
                if isinstance(v, str):
                    parts.append(v)
            for va in s.get('visual_assets', []) or []:
                if isinstance(va, dict):
                    for k, v in va.items():
                        if isinstance(v, str):
                            parts.append(v)
                        elif isinstance(v, dict):
                            parts.append(json.dumps(v, ensure_ascii=False))
    return '\n'.join(parts)


def collect_quiz_text(quiz: dict):
    parts = [quiz.get('title', '')]
    for q in quiz.get('questions', []):
        for key in ('question', 'answer', 'explanation'):
            v = q.get(key)
            if isinstance(v, str):
                parts.append(v)
        for opt in q.get('options', []) or []:
            if isinstance(opt, str):
                parts.append(opt)
    return '\n'.join(parts)


def collect_code_text(run_dir: Path):
    parts = []
    scripts = run_dir / 'scripts'
    if scripts.exists():
        for p in scripts.rglob('*.py'):
            try:
                parts.append(p.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                pass
    assets = run_dir / 'assets'
    if assets.exists():
        for p in assets.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.txt', '.csv', '.md'}:
                try:
                    parts.append(p.read_text(encoding='utf-8', errors='ignore'))
                except Exception:
                    pass
            else:
                parts.append(p.stem.replace('_', ' '))
    return '\n'.join(parts)


def collect_video_text(run_dir: Path, storyboard: dict | None):
    parts = []
    if storyboard:
        for s in storyboard.get('slides', []):
            for key in ('title', 'subtitle', 'content'):
                v = s.get(key)
                if isinstance(v, str):
                    parts.append(v)
    temp_video = run_dir / 'output' / 'temp_video'
    if temp_video.exists():
        for p in temp_video.rglob('*'):
            if p.is_file():
                parts.append(p.stem.replace('_', ' '))
    lecture = run_dir / 'output'
    if lecture.exists():
        for p in lecture.glob('*.mp4'):
            parts.append(p.stem.replace('_', ' '))
    return '\n'.join(parts)


manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))

rows = []
slide_scores = []
code_align_scores = []
quiz_coverage_scores = []
consistency_pairs = {'S-C': [], 'S-Q': [], 'S-V': [], 'C-Q': [], 'C-V': [], 'Q-V': []}
quiz_total = 0
quiz_valid = 0
questions_with_explanation = 0

for item in manifest:
    run_dir = GEN_ROOT / item['run_dir']
    student_input = json.loads(Path(item['output']).read_text(encoding='utf-8'))
    storyboard_path = run_dir / 'storyboard.json'
    storyboard = json.loads(storyboard_path.read_text(encoding='utf-8')) if storyboard_path.exists() else None
    quiz_path = run_dir / 'output' / 'quiz.json'
    quiz = json.loads(quiz_path.read_text(encoding='utf-8')) if quiz_path.exists() else {'questions': []}

    slides = student_input.get('slides', [])
    slide_count = len(slides)
    title_ok = 1 if slide_count >= 1 else 0
    summary_ok = 1 if any('conclusion' in (s.get('title','').lower()) or 'summary' in (s.get('title','').lower()) for s in slides) else 0
    content_ok = 1 if any((s.get('content') or '').strip() for s in slides[1:]) else 0
    page_bound_ok = 1 if 5 <= slide_count <= 12 else 0
    slide_score = (title_ok + summary_ok + content_ok + page_bound_ok) / 4
    slide_scores.append(slide_score)

    for q in quiz.get('questions', []):
        quiz_total += 1
        valid = isinstance(q.get('question'), str) and q.get('question').strip() and isinstance(q.get('answer'), str) and q.get('answer') in {'A','B','C','D'}
        opts = q.get('options', [])
        valid = valid and isinstance(opts, list) and len(opts) == 4 and all(isinstance(o, str) and o.strip() for o in opts)
        if valid:
            quiz_valid += 1
        if isinstance(q.get('explanation'), str) and q.get('explanation').strip():
            questions_with_explanation += 1

    slide_text = collect_slide_text(student_input, storyboard)
    code_text = collect_code_text(run_dir)
    quiz_text = collect_quiz_text(quiz if quiz.get('questions') else {'title': student_input.get('quiz_title',''), 'questions': student_input.get('questions',[])})
    video_text = collect_video_text(run_dir, storyboard)

    ks = keyword_set(slide_text)
    kc = keyword_set(code_text)
    kq = keyword_set(quiz_text)
    kv = keyword_set(video_text)
    km = ks | kc

    sc = jaccard(ks, kc)
    qcov = (len(kq & km) / len(kq)) if kq else 0.0
    sq = jaccard(ks, kq)
    sv = jaccard(ks, kv)
    cq = jaccard(kc, kq)
    cv = jaccard(kc, kv)
    qv = jaccard(kq, kv)

    code_align_scores.append(sc)
    quiz_coverage_scores.append(qcov)
    consistency_pairs['S-C'].append(sc)
    consistency_pairs['S-Q'].append(sq)
    consistency_pairs['S-V'].append(sv)
    consistency_pairs['C-Q'].append(cq)
    consistency_pairs['C-V'].append(cv)
    consistency_pairs['Q-V'].append(qv)

    rows.append({
        'topic': item['topic'],
        'run_dir': item['run_dir'],
        'slide_count': slide_count,
        'slide_compliance': slide_score,
        'code_alignment_sc': sc,
        'quiz_coverage': qcov,
        'sq': sq,
        'sv': sv,
        'cq': cq,
        'cv': cv,
        'qv': qv,
    })

summary = {
    'topic_count': len(rows),
    'code_validity_user_provided': {'successful_runs': 17, 'total_runs': 20, 'rate': 0.85},
    'slide_structural_compliance_mean': mean(slide_scores) if slide_scores else None,
    'quiz_format_validity': quiz_valid / quiz_total if quiz_total else None,
    'quiz_questions_total': quiz_total,
    'quiz_questions_with_explanation_rate': questions_with_explanation / quiz_total if quiz_total else None,
    'code_concept_alignment_mean': mean(code_align_scores) if code_align_scores else None,
    'quiz_material_coverage_mean': mean(quiz_coverage_scores) if quiz_coverage_scores else None,
    'pairwise_consistency_mean': {k: mean(v) if v else None for k, v in consistency_pairs.items()},
    'global_pairwise_consistency_mean': mean([x for vals in consistency_pairs.values() for x in vals]) if rows else None,
    'per_topic': rows,
}

OUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')

md = []
md.append('# Computed Results Metrics')
md.append(f"- Topics: {summary['topic_count']}")
md.append(f"- Code validity (user-provided): {summary['code_validity_user_provided']['successful_runs']}/{summary['code_validity_user_provided']['total_runs']} = {summary['code_validity_user_provided']['rate']:.3f}")
md.append(f"- Slide structural compliance mean: {summary['slide_structural_compliance_mean']:.3f}")
md.append(f"- Quiz format validity: {summary['quiz_format_validity']:.3f} ({quiz_valid}/{quiz_total})")
md.append(f"- Quiz explanation presence rate: {summary['quiz_questions_with_explanation_rate']:.3f}")
md.append(f"- Code concept alignment mean (S-C Jaccard): {summary['code_concept_alignment_mean']:.3f}")
md.append(f"- Quiz material coverage mean: {summary['quiz_material_coverage_mean']:.3f}")
md.append(f"- Global pairwise consistency mean: {summary['global_pairwise_consistency_mean']:.3f}")
md.append('')
md.append('## Pairwise consistency means')
for k, v in summary['pairwise_consistency_mean'].items():
    md.append(f"- {k}: {v:.3f}")
OUT_MD.write_text('\n'.join(md), encoding='utf-8')
print(json.dumps(summary, indent=2))
