# ============= African research concept recall/association probe ==========
# This is a deterministic paper-prefix cloze probe, not a free-form QA test.
# It compares the SAME model with CPT enabled and disabled. Positive raw gain
# means better prediction of the true African concept; positive association
# gain means the paper context helped beyond merely learning the concept name.
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

RECALL_SEED = 20260824
RECALL_PAPERS = 200            # deterministic papers loaded per split
RECALL_PER_SPLIT = 200         # requested probes per split after filtering
MAX_PER_CONCEPT = 3           # prevents Nigeria or one term dominating
MAX_PER_PAPER = 2
MAX_PREFIX_TOKENS = 512
MAX_TARGET_TOKENS = 12
PREFIX_CHARS = 3_000
NEUTRAL_PREFIX = 'This African research paper discusses a place or local concept:'

AFRICAN_COUNTRIES = '''Algeria|Angola|Benin|Botswana|Burkina Faso|Burundi|Cabo Verde|Cameroon|Central African Republic|Chad|Comoros|Democratic Republic of the Congo|Republic of the Congo|Djibouti|Egypt|Equatorial Guinea|Eritrea|Eswatini|Ethiopia|Gabon|Gambia|Ghana|Guinea|Guinea-Bissau|Kenya|Lesotho|Liberia|Libya|Madagascar|Malawi|Mali|Mauritania|Mauritius|Morocco|Mozambique|Namibia|Niger|Nigeria|Rwanda|Senegal|Seychelles|Sierra Leone|Somalia|South Africa|South Sudan|Sudan|Tanzania|Togo|Tunisia|Uganda|Zambia|Zimbabwe'''.split('|')
NIGERIAN_PLACES = '''Abia|Adamawa|Akwa Ibom|Anambra|Bauchi|Bayelsa|Benue|Borno|Cross River|Delta|Ebonyi|Edo|Ekiti|Enugu|Gombe|Imo|Jigawa|Kaduna|Kano|Katsina|Kebbi|Kogi|Kwara|Lagos|Nasarawa|Niger State|Ogun|Ondo|Osun|Oyo|Plateau|Rivers|Sokoto|Taraba|Yobe|Zamfara|Abuja|Federal Capital Territory|Ibadan|Ilorin|Maiduguri|Minna|Zaria|Akure|Abeokuta|Calabar|Uyo|Owerri|Port Harcourt|Benin City|Makurdi|Yola|Jos|Niger Delta|Lake Chad|Sahel|Congo Basin|Lake Victoria|Horn of Africa'''.split('|')
LOCAL_CONCEPTS = {
    'onugbu / bitter leaf': ('onugbu', 'onubu', 'bitter leaf', 'Vernonia amygdalina'),
    'zobo / roselle': ('zobo', 'zobo leaf', 'zobo leaves', 'zobo drink', 'roselle', 'Hibiscus sabdariffa'),
    'ugwu / fluted pumpkin': ('ugwu', 'ugu', 'fluted pumpkin', 'Telfairia occidentalis'),
    'ogbono / bush mango': ('ogbono', 'bush mango', 'Irvingia gabonensis'),
    'egusi': ('egusi', 'egusi melon', 'melon seed'),
    'iru / dawadawa': ('iru', 'dawadawa', 'African locust bean', 'Parkia biglobosa'),
    'garri': ('garri', 'gari'),
    'ogi / akamu': ('ogi', 'akamu'),
    'kunu': ('kunu', 'kunu-zaki', 'kunu zaki'),
    'tuwo shinkafa': ('tuwo shinkafa',),
    'amala': ('amala',),
    'eba': ('eba',),
    'fufu': ('fufu',),
    'suya': ('suya',),
    'kilishi': ('kilishi',),
    'shea': ('shea butter', 'shea nut', 'Vitellaria paradoxa'),
    'baobab': ('baobab', 'Adansonia digitata'),
    'Bambara groundnut': ('Bambara groundnut', 'Vigna subterranea'),
    'African yam bean': ('African yam bean', 'Sphenostylis stenocarpa'),
    'scent leaf': ('scent leaf', 'Ocimum gratissimum', 'efirin', 'nchuanwu'),
    'uziza': ('uziza', 'Piper guineense'),
    'utazi': ('utazi', 'Gongronema latifolium'),
    'uda': ('uda', 'Xylopia aethiopica'),
    'fonio': ('fonio',),
    'teff': ('teff',),
    'injera': ('injera',),
}

def _norm(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    return re.sub(r'\s+', ' ', value).strip()

concepts = []
for place in AFRICAN_COUNTRIES:
    aliases = (place, 'Nigerian') if place == 'Nigeria' else (place,)
    concepts.append(('place', place, aliases))
for place in NIGERIAN_PLACES:
    aliases = (place, place + ' State') if not place.endswith(('State', 'Territory', 'Delta', 'Chad', 'Sahel', 'Basin', 'Victoria', 'Africa')) else (place,)
    concepts.append(('place', place, aliases))
for canonical, aliases in LOCAL_CONCEPTS.items():
    concepts.append(('local_concept', canonical, aliases))

alias_lookup = {}
concept_aliases = {}
for category, canonical, aliases in concepts:
    concept_aliases[(category, canonical)] = tuple(aliases)
    for alias in aliases:
        alias_lookup.setdefault(_norm(alias), (category, canonical))
alias_terms = sorted(alias_lookup, key=lambda value: (-len(value), value))
alias_pattern = re.compile(
    r'(?<![\w])(?:' + '|'.join(re.escape(term).replace(r'\ ', r'\s+') for term in alias_terms) + r')(?![\w])',
    re.IGNORECASE,
)

def _contains_alias(prompt, aliases):
    normalized = _norm(prompt)
    return any(re.search(r'(?<!\w)' + re.escape(_norm(alias)) + r'(?!\w)', normalized) for alias in aliases)

def _paper_candidates(text, split_name, paper_number):
    paper_id = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    seen = set()
    rows = []
    for match in alias_pattern.finditer(text):
        record = alias_lookup.get(_norm(match.group(0)))
        if record is None or match.start() < 200 or not text[match.start() - 1].isspace():
            continue
        category, canonical = record
        surface = text[match.start():match.end()]
        if category == 'place' and not surface[:1].isupper():
            continue
        if category == 'local_concept' and len(_norm(surface)) <= 4 and surface != surface.lower():
            continue
        key = (paper_id, category, canonical)
        if key in seen:
            continue
        prompt = text[max(0, match.start() - PREFIX_CHARS):match.start()].rstrip()
        aliases = concept_aliases[(category, canonical)]
        if len(prompt) < 200 or _contains_alias(prompt, aliases):
            continue
        seen.add(key)
        rows.append({
            'split': split_name, 'paper_id': paper_id, 'paper_number': paper_number,
            'category': category, 'canonical': canonical,
            'target': surface, 'prompt': prompt,
            'source_offset': match.start(),
        })
    return rows

def _sample_candidates(texts, split_name, seed):
    pool = []
    for paper_number, text in enumerate(texts):
        pool.extend(_paper_candidates(text, split_name, paper_number))
    pool.sort(key=lambda row: (row['paper_id'], row['source_offset'], row['canonical'], row['target']))
    random.Random(seed).shuffle(pool)
    selected, per_concept, per_paper = [], Counter(), Counter()
    for desired_category in ('local_concept', 'place', None):
        category_limit = RECALL_PER_SPLIT // 2 if desired_category else RECALL_PER_SPLIT
        for row in pool:
            if row in selected or (desired_category and row['category'] != desired_category):
                continue
            if desired_category and sum(item['category'] == desired_category for item in selected) >= category_limit:
                break
            if per_concept[row['canonical']] >= MAX_PER_CONCEPT or per_paper[row['paper_id']] >= MAX_PER_PAPER:
                continue
            selected.append(row)
            per_concept[row['canonical']] += 1
            per_paper[row['paper_id']] += 1
            if len(selected) >= RECALL_PER_SPLIT:
                return selected
    return selected

# Reuse the model from the matched adapter-toggle sanity cell or the live trainer.
recall_model = None
for candidate_name in ('cpt', 'paired_model', 'cpt_model'):
    candidate = globals().get(candidate_name)
    if candidate is not None and hasattr(candidate, 'disable_adapter'):
        recall_model = candidate
        break
if recall_model is None:
    raise RuntimeError('Run the matched adapter-toggle sanity cell first, or run this cell before the notebook frees trainer.model.')
recall_tokenizer = globals().get('tokenizer') or globals().get('eval_tokenizer') or globals().get('base_tokenizer')
if recall_tokenizer is None or not getattr(recall_tokenizer, 'is_fast', False):
    raise RuntimeError('This probe requires the matching fast tokenizer with offset mappings.')
MODEL_API.for_inference(recall_model)

recall_trained_on = load_texts(TRAIN_DIR, RECALL_PAPERS)
recall_held_out = load_texts(EVAL_DIR, RECALL_PAPERS)
print('recall papers:', len(recall_trained_on), 'trained;', len(recall_held_out), 'held out')
train_hashes = {hashlib.sha256(text.encode('utf-8')).hexdigest() for text in recall_trained_on}
held_hashes = {hashlib.sha256(text.encode('utf-8')).hexdigest() for text in recall_held_out}
assert train_hashes.isdisjoint(held_hashes), 'trained and held-out probe papers overlap'
sample = _sample_candidates(recall_trained_on, 'trained_papers', RECALL_SEED)
sample += _sample_candidates(recall_held_out, 'held_out_papers', RECALL_SEED + 1)
if not sample:
    raise RuntimeError('No African concept candidates survived the strict no-copy filters.')

def _encode_cloze(prompt, target):
    prompt = prompt.rstrip()
    combined = prompt + ' ' + target
    boundary = len(prompt)
    encoded = recall_tokenizer(combined, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if any(start < boundary < end for start, end in offsets):
        return None
    target_positions = [index for index, (start, end) in enumerate(offsets) if start >= boundary and end > boundary]
    if not target_positions:
        return None
    first = target_positions[0]
    prefix_ids, target_ids = ids[:first], ids[first:]
    if not prefix_ids or not target_ids or len(target_ids) > MAX_TARGET_TOKENS:
        return None
    prefix_ids = prefix_ids[-MAX_PREFIX_TOKENS:]
    if recall_tokenizer.bos_token_id is not None:
        prefix_ids = [recall_tokenizer.bos_token_id] + prefix_ids
    return {'input_ids': prefix_ids + target_ids, 'prefix_len': len(prefix_ids), 'target_ids': target_ids}

items = []
for row in sample:
    conditional = _encode_cloze(row['prompt'], row['target'])
    neutral = _encode_cloze(NEUTRAL_PREFIX, row['target'])
    if conditional is None or neutral is None or conditional['target_ids'] != neutral['target_ids']:
        continue
    row = dict(row)
    row['conditional'], row['neutral'] = conditional, neutral
    items.append(row)
if not items:
    raise RuntimeError('All candidates failed exact joint-tokenization checks.')
sample_fingerprint = hashlib.sha256(json.dumps([
    (row['split'], row['paper_id'], row['source_offset'], row['canonical'], row['target'])
    for row in items
], ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
print('recall sample:', len(items), 'items;', Counter(row['category'] for row in items))
print('sample SHA256:', sample_fingerprint)

@torch.inference_mode()
def _score_one(model, sequence):
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor(sequence['input_ids'], dtype=torch.long, device=device)
    logits = model(input_ids=ids.unsqueeze(0)).logits[0]
    prefix_len = sequence['prefix_len']
    positions = torch.arange(prefix_len - 1, ids.numel() - 1, device=device)
    targets = ids[prefix_len:]
    selected_logits = logits[positions].float()
    nll = F.cross_entropy(selected_logits, targets, reduction='mean').item()
    first_logits = selected_logits[0]
    first_target = targets[0]
    top5 = torch.topk(first_logits, k=min(5, first_logits.numel())).indices
    return {
        'nll': nll,
        'first_top1': bool(first_logits.argmax() == first_target),
        'first_top5': bool((top5 == first_target).any()),
        'first_rank': int((first_logits > first_logits[first_target]).sum().item() + 1),
    }

def _score_items(model, rows, label):
    results = {}
    for index, row in enumerate(tqdm(rows, desc=label, unit='cloze')):
        results[(index, 'conditional')] = _score_one(model, row['conditional'])
        results[(index, 'neutral')] = _score_one(model, row['neutral'])
    return results

cpt_probe = _score_items(recall_model, items, 'CPT recall probe')
with recall_model.disable_adapter():
    base_probe = _score_items(recall_model, items, 'base recall probe')
# A short on/off/on reproducibility check without repeating the entire run.
recheck = _score_items(recall_model, items[:3], 'CPT state recheck')
for key, value in recheck.items():
    assert abs(value['nll'] - cpt_probe[key]['nll']) <= 1e-4, (key, value, cpt_probe[key])

records = []
for index, row in enumerate(items):
    bc, cc = base_probe[(index, 'conditional')], cpt_probe[(index, 'conditional')]
    bn, cn = base_probe[(index, 'neutral')], cpt_probe[(index, 'neutral')]
    raw_gain = bc['nll'] - cc['nll']
    neutral_gain = bn['nll'] - cn['nll']
    records.append({
        'split': row['split'], 'paper_id': row['paper_id'], 'category': row['category'],
        'canonical': row['canonical'], 'target': row['target'],
        'target_tokens': len(row['conditional']['target_ids']),
        'base_conditional_nll': bc['nll'], 'cpt_conditional_nll': cc['nll'],
        'raw_nll_gain': raw_gain, 'neutral_nll_gain': neutral_gain,
        'association_gain': raw_gain - neutral_gain,
        'base_first_top1': bc['first_top1'], 'cpt_first_top1': cc['first_top1'],
        'base_first_top5': bc['first_top5'], 'cpt_first_top5': cc['first_top5'],
        'base_first_rank': bc['first_rank'], 'cpt_first_rank': cc['first_rank'],
        'prompt_tail': row['prompt'][-180:].replace('\n', ' '),
    })
recall_frame = pd.DataFrame(records)

summary_rows = []
for split_name, group in recall_frame.groupby('split', sort=False):
    paper_gains = group.groupby('paper_id')['association_gain'].mean().to_numpy()
    rng = np.random.default_rng(RECALL_SEED)
    if len(paper_gains) > 1:
        boot = np.array([rng.choice(paper_gains, len(paper_gains), replace=True).mean() for _ in range(2_000)])
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    else:
        ci_low = ci_high = paper_gains.mean()
    summary_rows.append({
        'split': split_name, 'items': len(group), 'papers': group.paper_id.nunique(),
        'distinct_concepts': group.canonical.nunique(),
        'base_span_ppl': math.exp(group.base_conditional_nll.mean()),
        'cpt_span_ppl': math.exp(group.cpt_conditional_nll.mean()),
        'conditional_win_rate': (group.raw_nll_gain > 0).mean(),
        'base_first_token_recall@5': group.base_first_top5.mean(),
        'cpt_first_token_recall@5': group.cpt_first_top5.mean(),
        'mean_raw_nll_gain': group.raw_nll_gain.mean(),
        'mean_neutral_nll_gain': group.neutral_nll_gain.mean(),
        'mean_association_gain': group.association_gain.mean(),
        'association_gain_ci95_low': ci_low, 'association_gain_ci95_high': ci_high,
    })
recall_summary = pd.DataFrame(summary_rows)
display(recall_summary.round(4))
display(recall_frame[['split', 'category', 'target', 'raw_nll_gain', 'association_gain', 'base_first_rank', 'cpt_first_rank', 'prompt_tail']].head(12))

output_path = FINAL_DIR / 'african_concept_recall_probe.csv'
recall_frame.to_csv(output_path, index=False)
print('saved:', output_path)
print('Interpretation: positive raw gain = better true-target prediction; positive association gain')
print('= paper-context learning beyond a general African-name prior. Trained-paper results test')
print('retention; held-out results test transfer. This is not yet free-form factual QA recall.')
