
def peek(ds, n=1):
    """Print the first n examples so you can verify the real column names."""
    it = iter(ds)
    for _ in range(n):
        ex = next(it)
        print({k: (str(v)[:200]) for k, v in ex.items()})
    return ds


def _first(example, keys, default=""):
    """Return the first present, non-empty field among `keys`."""
    for k in keys:
        v = example.get(k)
        if v not in (None, "", []):
            return v
    return default


def _as_text(v):
    """Coerce list / None answers into a single string."""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v if x is not None)
    return "" if v is None else str(v)


def _messages(user, assistant):
    return {"messages": [
        {"role": "user", "content": _as_text(user).strip()},
        {"role": "assistant", "content": _as_text(assistant).strip()},
    ]}



def format_pubmedqa(ex):
    """fedml/PubMedQA_instruction — Alpaca-style instruction / input / output.
    VERIFY with peek(); typical layout is instruction/input/output."""
    instruction = _first(ex, ["instruction", "question", "QUESTION"])
    context = _first(ex, ["input", "context", "CONTEXTS", "abstract"])
    answer = _first(ex, ["output", "answer", "long_answer",
                         "LONG_ANSWER", "final_decision"])
    context = _as_text(context)
    user = f"{_as_text(instruction)}\n\n{context}".strip() if context else instruction
    return _messages(user, answer)


def format_bioasq(ex):
    """bigbio/bioasq_task_b (BigBIO QA schema): question / context / answer(list)."""
    question = _first(ex, ["question"])
    context = _as_text(_first(ex, ["context", "passages"]))
    answer = _first(ex, ["answer", "answers", "exact_answer"])
    user = f"Context: {context}\n\nQuestion: {_as_text(question)}".strip() \
        if context else question
    return _messages(user, answer)


def format_mol_instructions(ex):
    """zjunlp/Mol-Instructions — Alpaca-style instruction / input / output."""
    instruction = _first(ex, ["instruction"])
    mol_input = _as_text(_first(ex, ["input"]))
    output = _first(ex, ["output"])
    user = f"{_as_text(instruction)}\n\n{mol_input}".strip() if mol_input else instruction
    return _messages(user, output)


def format_blurb_qa(ex):
    """EMBO/BLURB — ONLY meaningful for QA subsets (e.g. pubmedqa/bioasq inside
    BLURB). Do NOT apply to NER / token-classification subsets."""
    question = _first(ex, ["question", "text", "sentence"])
    context = _as_text(_first(ex, ["context", "passage"]))
    answer = _first(ex, ["answer", "label", "labels"])
    user = f"{context}\n\n{_as_text(question)}".strip() if context else question
    return _messages(user, answer)


_LETTERS = "ABCDEFGHIJ"


def _mcq_prompt(question, options):
    body = "\n".join(f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (f"{question}\n\n{body}\n\n"
            "Answer with only the letter of the correct choice.")


def _gold_letter(ex, options):
    """Normalize a gold answer that may be an int index, a letter, or full text."""
    ans = _first(ex, ["answer", "answer_index", "label", "correct"])
    if isinstance(ans, int):
        return _LETTERS[ans]
    ans = str(ans).strip()
    if ans in _LETTERS:            # already a letter
        return ans
    if ans.isdigit():             # stringified index
        return _LETTERS[int(ans)]
    if ans in options:            # full option text
        return _LETTERS[options.index(ans)]
    return ans


def format_mmlu_mcq(ex):
    """MMLU-style: question + choices (list of 4) + integer/letter answer.
    Handles both shuyuej/... and brucewlee1/... layouts (list or A/B/C/D cols)."""
    question = _as_text(_first(ex, ["question", "Question"]))
    options = _first(ex, ["choices", "options", "Choices"], default=None)
    if options is None:           # options stored as separate A/B/C/D columns
        options = [ex[k] for k in ["A", "B", "C", "D"] if k in ex]
    options = [_as_text(o) for o in options]
    return {"messages": [{"role": "user", "content": _mcq_prompt(question, options)}],
            "answer": _gold_letter(ex, options)}


def format_mmlu_pro(ex):
    """TIGER-Lab/MMLU-Pro: question + options (list up to 10) + answer letter."""
    question = _as_text(_first(ex, ["question"]))
    options = [_as_text(o) for o in _first(ex, ["options", "choices"], default=[])]
    return {"messages": [{"role": "user", "content": _mcq_prompt(question, options)}],
            "answer": _gold_letter(ex, options)}



SFT_FORMATTERS = {
    "pubmedqa": format_pubmedqa,
    "bioasq": format_bioasq,
    "mol_instructions": format_mol_instructions,
    "blurb": format_blurb_qa,
}

BENCHMARK_FORMATTERS = {
    "mmlu_college_biology": format_mmlu_mcq,
    "mmlu_college_biology_alt": format_mmlu_mcq,
    "mmlu_pro": format_mmlu_pro,
}


def _map_stream(ds, fn):
    """Apply `fn` and drop the original columns. Works for streaming IterableDataset."""
    cols = getattr(ds, "column_names", None)
    return ds.map(fn, remove_columns=cols) if cols else ds.map(fn)


def format_sft(name, split_ds):
    return _map_stream(split_ds, SFT_FORMATTERS[name])


def format_benchmark(name, ds):
    if name == "mmlu_pro":        # MMLU-Pro spans all subjects -> keep biology only
        ds = ds.filter(lambda x: str(x.get("category", "")).lower() == "biology")
    return _map_stream(ds, BENCHMARK_FORMATTERS[name])


def format_all_sft(datasets):
    """datasets: {name: {"train": ds, "eval": ds}} -> same shape, chat-formatted."""
    return {
        name: {split: format_sft(name, ds) for split, ds in splits.items()}
        for name, splits in datasets.items()
    }


def format_all_benchmarks(benchmarks):
    return {name: format_benchmark(name, ds) for name, ds in benchmarks.items()}



if __name__ == "__main__":
    ds = load_dataset("fedml/PubMedQA_instruction", split="train", streaming=True)
    print("RAW columns / first row:")
    peek(ds)
    print("\nFORMATTED:")
    peek(format_sft("pubmedqa", ds))