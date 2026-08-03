Question: I'm sorry but I am not convinced about the embedding. I dont think I would use it. I'd rather try Kaggle's T4 GPU. Then run the fp or quantized version of the Qwen 3-1.7B there. No finetuning. Just run that with the abstracts, title etc and see how it will help. I can also parallelize the run depending on how much compute I can get from kaggle.

Answer: Yes, **your approach is sensible**, and you can completely skip embeddings for the first experiment.

For your nuanced criterion, a small instruction model reading the **title, abstract, keywords, topics and affiliations** may be more intuitive than embedding similarity. The essential requirement is to validate its decisions before trusting it across tens of thousands of papers.

## One change: consider Qwen3.5-2B

Rather than starting only with `Qwen/Qwen3-1.7B`, I would benchmark:

1. **`Qwen/Qwen3.5-2B`** — preferred
2. **`Qwen/Qwen3-1.7B`** — stable baseline

Qwen3.5-2B is newer and officially scores considerably higher than Qwen3-1.7B on several knowledge and reasoning benchmarks. For example, in non-thinking mode, Qwen reports MMLU-Pro scores of **55.3 versus 40.2** and SuperGPQA scores of **30.4 versus 21.0**. ([Hugging Face][1])

Both should fit comfortably on a 16 GB T4:

| Model      | Approximate FP16 weights |
| ---------- | -----------------------: |
| Qwen3-1.7B |                   3.4 GB |
| Qwen3.5-2B |                     4 GB |

Runtime memory will be higher because of the KV cache, activations and batching, but abstracts are short enough that 16 GB is ample.

## Use FP16 first, not quantization

On a T4, I would begin with:

```python
torch_dtype=torch.float16
```

You do not need 4-bit quantization for a 1.7B or 2B model. Quantization would:

* Reduce memory consumption
* Potentially allow larger batches
* Not necessarily improve speed on a T4
* Introduce another possible source of classification degradation

Use 4-bit only when:

* You move to a 4B–8B model
* FP16 runs out of memory at your desired batch size
* Your benchmark proves that 4-bit maintains almost identical classification quality

For this experiment, **FP16 is the cleaner baseline**.

## How to use Kaggle’s two T4s properly

Kaggle provides a **T4 ×2** option. Each GPU has 16 GB of VRAM. Kaggle’s current standard quota is around **30 GPU hours per week**, sometimes higher depending on available capacity, and GPU notebook sessions can run for up to 12 hours. ([Kaggle][2])

Do not split this tiny model across the two GPUs. That adds communication overhead.

Instead:

```text
T4 GPU 0:
    Full copy of Qwen
    Papers 0, 2, 4, 6, ...

T4 GPU 1:
    Full copy of Qwen
    Papers 1, 3, 5, 7, ...
```

That is **data parallel inference**: two independent workers, each processing half the papers.

Since you have Colab Pro+, Kaggle currently advertises an additional **30 Kaggle GPU hours per week** for verified Colab Pro+ users. That could bring your available allocation to around 60 hours, although you will need to claim and link the benefit. ([Kaggle][3])

## Recommended processing pipeline

```text
OpenAlex candidate metadata
        ↓
Remove duplicates, retractions and unusable records
        ↓
Qwen3.5-2B or Qwen3-1.7B on Kaggle T4 ×2
        ↓
Include / Exclude / Review
        ↓
Qwen3.7-Plus reviews uncertain cases
        ↓
Download only accepted papers
        ↓
Full-text confirmation
```

You do not need embeddings anywhere in that pipeline.

## Keep the task simple for the small model

Do not initially ask a 2B model to produce elaborate five-dimensional analyses. Give it a constrained classification task.

Use an output like:

```json
{
  "decision": "include",
  "category": "local_materials",
  "african_focus": "essential",
  "scientific_evidence": "empirical",
  "evidence": "Laterite samples were collected from three sites in southeastern Nigeria.",
  "reason": "The research experimentally studies a locally sourced Nigerian engineering material."
}
```

Allow only:

```text
decision:
- include
- exclude
- review

african_focus:
- essential
- meaningful
- incidental
- absent

scientific_evidence:
- experimental
- observational
- computational
- systematic_review
- weak_or_none
```

The `evidence` field is especially important. Require the model to extract a short supporting phrase from the abstract rather than merely asserting that the paper is relevant.

## High-recall instruction

Your screening prompt should explicitly say:

```text
Use REVIEW rather than EXCLUDE whenever the abstract does not provide
enough evidence. Do not exclude a potentially relevant paper merely
because its local significance is unclear.

Only choose EXCLUDE when the available text provides positive evidence
that the study fails the inclusion criteria.
```

This matters because false negatives are your biggest danger. You can remove weak papers later, but you cannot inspect a strong paper that was silently discarded.

## Inference settings

For Qwen3-1.7B:

```python
enable_thinking=False
do_sample=False
max_new_tokens=100
```

Qwen3 supports thinking and non-thinking modes, and Qwen documents non-thinking mode as the faster option for straightforward tasks. Its context window is 32K tokens, far more than you need for an abstract. ([Qwen][4])

Qwen3.5-2B already operates in non-thinking mode by default. Qwen recommends vLLM or SGLang for high-throughput inference, although ordinary Transformers batching is easier for your initial test. ([Hugging Face][1])

Other useful settings:

```python
batch_size = 16        # test 8, 16, 32
max_input_tokens = 1536
max_new_tokens = 100
do_sample = False
```

Sort papers by approximate input length before batching. Similar-length batches waste less computation on padding.

## How long could it take?

Suppose you have 30,000 candidate papers. Your actual speed depends on abstract length, batch size and inference framework.

| Combined two-GPU speed | Time for 30,000 papers |
| ---------------------: | ---------------------: |
|         1 paper/second |              8.3 hours |
|        2 papers/second |              4.2 hours |
|        4 papers/second |              2.1 hours |
|        6 papers/second |              1.4 hours |

A 2B model with batching should make the workload practical within a Kaggle session, but benchmark the first **1,000 papers** rather than relying on theoretical estimates.

Record:

* Papers per second
* Input tokens per second
* Peak GPU memory
* Valid JSON percentage
* Include/review/exclude distribution
* Accuracy against your manual labels

## Validation before the full run

Manually label about **300 papers**:

* 100 clear inclusions
* 100 clear exclusions
* 100 ambiguous cases

Then compare Qwen3-1.7B and Qwen3.5-2B on exactly the same set.

Your selection criterion should be:

1. Recall on relevant papers
2. Correct handling of African-affiliation-only papers
3. Valid structured output rate
4. Speed
5. Precision

I would accept somewhat lower precision to preserve recall. A sensible initial requirement is:

* Relevant-paper recall: **at least 90–95%**
* Excluded-paper precision: **at least 85%**
* Invalid JSON: **below 1–2%**

## Safeguards for a long Kaggle run

Write the results incrementally:

```text
results_gpu0.jsonl
results_gpu1.jsonl
```

Save after every 100–500 papers. Store the OpenAlex ID as the unique key, and resume by skipping IDs already present. This protects you against Kaggle’s 12-hour session boundary or notebook disconnections. ([Kaggle][5])

Your proposal is therefore sound. My preferred test would be:

> **Qwen3.5-2B FP16, non-thinking, batch size 16–32, one full model replica per T4, 300 manually labelled validation papers, and Qwen3.7-Plus only for `review` cases and disagreements.**

[1]: https://huggingface.co/Qwen/Qwen3.5-2B "Qwen/Qwen3.5-2B · Hugging Face"
[2]: https://www.kaggle.com/docs/efficient-gpu-usage?utm_source=chatgpt.com "Efficient GPU Usage Tips"
[3]: https://www.kaggle.com/product-announcements/575468?utm_source=chatgpt.com "Unlock extra GPU on Kaggle with Colab Pro"
[4]: https://qwenlm.github.io/blog/qwen3/ "Qwen3: Think Deeper, Act Faster | Qwen"
[5]: https://www.kaggle.com/docs/notebooks?utm_source=chatgpt.com "Kaggle notebooks documentation"




Question:
Ok but I have another problem, first knowing which papers to download. I need to write a sccript that would crawl through openAlex or something like that and specifically select african/nigerian papers. Also, not all african papers would be used, I have a criteria that they must match in terms of their content and focus or direction of the research. How do I start. Do I use the qwen model to evaluate the papers or what how do I know which to download vs which to keep
Answer:
Yes, use Qwen to evaluate the papers, but **do not download thousands of full PDFs before screening them**.

The right workflow is:

> **OpenAlex finds candidates → rules remove obvious mismatches → Qwen evaluates titles and abstracts → you download only accepted papers → Qwen performs a final full-text check.**

## 1. Separate “African paper” from “African-focused paper”

These are not the same:

* **Africa-produced:** at least one author belongs to an African institution.
* **Africa-published:** appears in an African journal.
* **Africa-focused:** the study investigates an African problem, material, resource, population, environment, method or innovation.
* **MUFASA-relevant:** contains scientific knowledge that helps the model reason from African evidence.

For MUFASA, **African affiliation should be a discovery signal, not an automatic inclusion rule**.

For example:

* A University of Lagos researcher studying a generic neural-network benchmark: African-produced, but probably not MUFASA-relevant.
* A German research team studying corrosion behaviour of Nigerian crude-oil pipelines: not necessarily African-produced, but strongly Africa-focused and relevant.
* A paper characterising Nigerian clay for low-cost ceramic production: strongly relevant.

Therefore, build two candidate pools and merge them:

1. Papers with Nigerian/African institutional affiliations.
2. Papers whose content is clearly about Africa, even without African-affiliated authors.

## 2. Write the inclusion rubric before crawling

Do not initially tell Qwen merely:

> “Is this paper relevant to African science?”

That is too vague. Give it measurable dimensions.

### Recommended MUFASA screening rubric

Score each paper from 0–4 on:

| Dimension           | What it measures                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| African centrality  | Is Africa central to the research rather than mentioned incidentally?                                       |
| Local specificity   | Does it involve African materials, resources, species, populations, climate, infrastructure or constraints? |
| Scientific depth    | Does it contain methods, experiments, measurements, results or scientific analysis?                         |
| Knowledge value     | Could the findings teach a model useful scientific knowledge or reasoning?                                  |
| Local applicability | Could the work inform locally appropriate solutions, designs or future experiments?                         |

### Hard exclusions

Exclude papers that are:

* African only because of the author’s affiliation.
* Editorials, announcements, letters or conference schedules.
* Pure opinion pieces without scientific evidence.
* Retracted.
* Duplicates.
* Outside your chosen STEM domains.
* Merely mentioning Nigeria or Africa in one sentence.
* Missing enough content to judge, unless placed in the manual-review queue.

A practical threshold:

* **Include:** at least 14/20
* **Review:** 10–13
* **Exclude:** below 10
* Require `African centrality >= 2`
* Require `Scientific depth >= 2`

Keep these scores individually. Do not retain only one overall score.

## 3. Use OpenAlex for broad candidate discovery

OpenAlex lets you filter works by author-institution country, topic hierarchy, publication type, abstract availability, retraction status, open-access status and downloadable content. Its work records also provide topics, keywords, authorships, OA locations and an inverted representation of the abstract. ([OpenAlex Developers][1])

Start with Nigerian-affiliated scientific articles:

```text
authorships.institutions.country_code:ng,
type:article,
has_abstract:true,
is_retracted:false
```

Do **not** initially require a downloadable PDF. Otherwise, you may discard excellent papers before classification. Add the full-text requirement only after screening.

OpenAlex cursor pagination should be used once your results exceed 10,000, and `per_page=100` reduces the number of requests. ([OpenAlex Developers][2])

### Minimal metadata crawler

```python
import json
import time
from pathlib import Path

import requests


OPENALEX_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Convert OpenAlex's inverted abstract into normal text."""
    if not inverted_index:
        return ""

    positioned_words = []

    for word, positions in inverted_index.items():
        for position in positions:
            positioned_words.append((position, word))

    positioned_words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned_words)


def crawl_openalex(
    api_key: str,
    output_file: str = "nigerian_candidates.jsonl",
    max_records: int = 20_000,
) -> None:
    filters = [
        "authorships.institutions.country_code:ng",
        "type:article",
        "has_abstract:true",
        "is_retracted:false",
    ]

    params = {
        "api_key": api_key,
        "filter": ",".join(filters),
        "per_page": 100,
        "cursor": "*",
        "select": ",".join(
            [
                "id",
                "doi",
                "title",
                "publication_year",
                "type",
                "language",
                "cited_by_count",
                "is_retracted",
                "abstract_inverted_index",
                "authorships",
                "primary_topic",
                "topics",
                "keywords",
                "best_oa_location",
                "has_content",
                "content_url",
            ]
        ),
    }

    output_path = Path(output_file)
    total = 0

    with output_path.open("w", encoding="utf-8") as file:
        while total < max_records:
            response = requests.get(
                OPENALEX_URL,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                time.sleep(10)
                continue

            response.raise_for_status()
            payload = response.json()
            works = payload.get("results", [])

            if not works:
                break

            for work in works:
                work["abstract"] = reconstruct_abstract(
                    work.pop("abstract_inverted_index", None)
                )

                file.write(json.dumps(work, ensure_ascii=False) + "\n")
                total += 1

                if total >= max_records:
                    break

            next_cursor = payload.get("meta", {}).get("next_cursor")

            if not next_cursor:
                break

            params["cursor"] = next_cursor

    print(f"Saved {total:,} candidates to {output_path}")


if __name__ == "__main__":
    crawl_openalex(
        api_key="YOUR_OPENALEX_API_KEY",
        max_records=20_000,
    )
```

OpenAlex abstracts must be reconstructed because the API returns them as an inverted index rather than plain text. ([OpenAlex Developers][3])

## 4. Create a second Africa-focused discovery pool

The first crawl finds African-authored work. It will miss internationally produced research about Africa.

For the second pool, use:

* African country and region names.
* Local materials and natural resources.
* Indigenous crops and species.
* Diseases and populations.
* Local manufacturing methods.
* Climate and environmental conditions.
* Infrastructure and engineering constraints.
* African scientific innovations.

You can also use OpenAlex semantic search with a paragraph such as:

> Scientific research centred on African materials, natural resources, crops, environments, populations, manufacturing methods, engineering constraints and locally appropriate innovations.

OpenAlex semantic search compares this description with paper titles and abstracts. However, its documentation says institution-country filters are not supported with semantic search, so retrieve semantic matches first and apply your geographic and content rules afterward. ([OpenAlex Developers][4])

Merge the two pools using, in order:

1. DOI
2. OpenAlex ID
3. Normalised title and year

## 5. Use Qwen on title and abstract, not the PDF

For each candidate, send:

* Title
* Abstract
* OpenAlex topic
* Keywords
* Author-affiliation countries
* Publication year

Require structured JSON:

```json
{
  "decision": "include",
  "african_centrality": 4,
  "local_specificity": 3,
  "scientific_depth": 4,
  "knowledge_value": 4,
  "local_applicability": 3,
  "total_score": 18,
  "paper_category": "local_materials",
  "african_locations": ["Nigeria"],
  "evidence": [
    "The study characterises locally sourced laterite.",
    "Experiments were conducted under Nigerian environmental conditions."
  ],
  "reason": "The paper contains empirical scientific knowledge about a Nigerian material and its engineering application."
}
```

### Screening prompt

```text
You are screening scientific papers for MUFASA, a scientific model that
learns from African evidence, materials, resources, methods, experiments,
constraints and innovations.

African authorship alone is not sufficient.

Include papers where the central scientific contribution concerns one or more
of the following:

1. African materials or natural resources
2. African crops, organisms, diseases or populations
3. African climate, geology or environmental conditions
4. Locally developed scientific or engineering methods
5. Infrastructure, manufacturing or technical constraints specific to Africa
6. Empirical findings useful for locally appropriate solutions
7. Scientific evidence that could support new African experiments or designs

Exclude papers where Africa is merely an author affiliation, incidental
mention or data-collection location with no locally meaningful scientific
content.

Score:
- African centrality: 0-4
- Local specificity: 0-4
- Scientific depth: 0-4
- Knowledge value: 0-4
- Local applicability: 0-4

Return only valid JSON with:
decision, the five scores, total_score, paper_category,
african_locations, evidence, and reason.

Decision rules:
- include: total >= 14, African centrality >= 2, scientific depth >= 2
- review: total 10-13 or insufficient abstract information
- exclude: total < 10 or hard exclusion applies
```

## 6. Do not use the strongest Qwen model for every paper

For perhaps **15,000–30,000 candidate abstracts**:

1. Use keywords, topics and deterministic rules to remove obvious exclusions.
2. Use a cheaper Qwen model to classify the remaining abstracts.
3. Send only the `review` cases and low-confidence cases to the strongest reasoning model.
4. Manually inspect a sample before downloading.

The strongest model should handle perhaps **10–20% of candidates**, not all of them. Classification does not normally require maximum reasoning.

Your $40 credit is better preserved for:

* Borderline scientific judgements.
* Identifying what knowledge can be extracted.
* Full-text verification.
* Generating the eventual training examples.

## 7. Manually label 200 papers first

Before screening everything, create a validation set:

* 70 definite inclusions.
* 70 definite exclusions.
* 60 difficult or borderline cases.

Run your Qwen prompt against these 200 papers.

At the candidate-screening stage, optimise for **high recall**. Missing an excellent paper is worse than allowing a few weak papers into the next stage. You can remove false positives during full-text screening.

A reasonable target would be:

* Abstract-screening recall: **at least 95%**
* Abstract-screening precision: **around 70–85%**
* Final full-text precision: **above 90%**

Do not trust the prompt until you have inspected its false negatives.

## 8. Download only after abstract screening

Use three queues:

| Queue   | Action                                                                |
| ------- | --------------------------------------------------------------------- |
| Include | Download automatically                                                |
| Review  | Download only when full text is easily available, or inspect manually |
| Exclude | Keep metadata and rejection reason, but do not download               |

For model training, OpenAlex’s GROBID/TEI XML is generally more useful than PDFs because it already contains structured text. OpenAlex currently offers cached PDF and TEI XML content and exposes availability through `has_content` and `content_url`. Content downloads cost $0.01 each; a free API key includes approximately 100 content files per day. ([OpenAlex Developers][5])

Downloading 5,000 files entirely through the OpenAlex content API would therefore cost approximately **$50**, or take around **50 days** using only the daily free allowance. You can also use `best_oa_location.pdf_url` for openly available repository copies, while respecting the host’s terms and rate limits. OpenAlex records include the best OA location and license information. ([OpenAlex Developers][3])

## 9. Perform a second evaluation after downloading

The abstract may exaggerate relevance. After obtaining the text, give Qwen only:

* Abstract
* Introduction
* Methods
* Main results
* Conclusion

Then ask:

* Is the African context scientifically essential?
* What exact local knowledge does this paper contribute?
* Does it contain sufficient evidence and methodological detail?
* Which sections should enter continued pretraining?
* Can it produce reasoning, QA, comparison or experimental-design examples?
* Is the paper primarily generic science with African data attached?

This final stage determines:

* `keep_full_text`
* `use_for_continued_pretraining`
* `use_for_sft_generation`
* `use_for_evaluation`
* `reject`

## Recommended scale

To finish with **5,000 strong papers**, I would initially collect approximately:

* **20,000–40,000 metadata candidates**
* **10,000–20,000 candidates after rule filtering**
* **7,000–10,000 after Qwen abstract screening**
* **5,000–7,000 after full-text validation**
* Final balanced corpus: approximately **5,000 papers**

Most importantly, keep the rejected records and reasons. Those exclusions become a labelled dataset you can later use to train a small, cheap relevance classifier, so Qwen does not remain permanently responsible for every future paper.

[1]: https://developers.openalex.org/api-reference/works "Works Overview - OpenAlex Developers"
[2]: https://developers.openalex.org/guides/page-through-results?utm_source=chatgpt.com "Page through Results - OpenAlex Developers"
[3]: https://developers.openalex.org/api-reference/works/get-a-single-work "Get a single work - OpenAlex Developers"
[4]: https://developers.openalex.org/guides/semantic-search?utm_source=chatgpt.com "Semantic Search - OpenAlex Developers"
[5]: https://developers.openalex.org/download/full-text-pdfs?utm_source=chatgpt.com "Full-text PDFs - OpenAlex Developers"

