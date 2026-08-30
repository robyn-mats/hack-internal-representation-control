"""Regenerate `irc/concepts.csv` from the versioned word list.

`irc/concepts.csv` is a DERIVED annotation of `irc/words_paper.py`, not an edit
to it (CLAUDE.md). Re-run this if that list ever changes. Output is committed so
the run has no NLP dependency and the form lists are auditable by eye.

Columns:
    concept, number         grammatical number, hand-assigned (17 mass / 33 plural)
    BE, DO, BELONG          agreement, resolved into the condition templates
    forms                   INFLECTIONAL forms -- the pre-registered leak measure
    forms_derived           DERIVATIONAL forms -- a declared sensitivity check

Why two tiers. Number inflection alone misses the obvious cases: a completion
saying "dusting" or "snowed" has leaked the concept. LemmInflect supplies those
from the lemma. Its occasional nonsense ("lightninged", "camerae") is harmless
precisely because such strings never occur in text, so it cannot cost precision.

WordNet derivational links are a different matter. They join by lemma STRING, so
polysemous concepts import derivations from unrelated senses -- which is why the
pruning below exists rather than being optional tidying.

Requires `pip install lemminflect` and `nltk.download('wordnet')`; neither is
needed at run time.
"""

from irc import env  # noqa: F401  -- must be the first import

import csv

import lemminflect
from nltk.corpus import wordnet as wn

from irc.paths import REPO_ROOT

# Derivational forms dropped by hand, with the reason. WordNet relates lemmas by
# string, so a polysemous concept inherits derivations from senses the
# experiment never intended.
DERIV_DROP = {
    "phones": {"phonate", "phoner", "phonetic", "phonic"},   # phone = speech sound
    "deserts": {"deserter", "desertion"},                    # desert = abandon
    "information": {"inform"},                               # "informed" is ubiquitous
    "frosts": {"frostian"},                                  # not a word
}


def singular(word: str) -> str:
    """All 33 plurals in the paper's list are regular."""
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def inflectional(word: str, number: str) -> list[str]:
    lemma = singular(word) if number == "plural" else word
    forms = {word, lemma}
    for tagged in lemminflect.getAllInflections(lemma).values():
        forms.update(tagged)
    return sorted(forms)


def derivational(word: str, number: str) -> list[str]:
    lemma = singular(word) if number == "plural" else word
    out = set()
    for syn in wn.synsets(lemma):
        for lem in syn.lemmas():
            if lem.name().lower() != lemma:
                continue
            for rel in lem.derivationally_related_forms():
                name = rel.name().lower().replace("_", " ")
                if name != lemma and " " not in name:
                    out.add(name)
    return sorted(out - DERIV_DROP.get(word, set()))


def main() -> None:
    path = REPO_ROOT / "irc" / "concepts.csv"
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        w, n = row["concept"], row["number"]
        row["forms"] = "|".join(inflectional(w, n))
        row["forms_derived"] = "|".join(derivational(w, n))

    fields = ["concept", "number", "BE", "DO", "BELONG", "forms", "forms_derived"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_deriv = sum(1 for r in rows if r["forms_derived"])
    print(f"wrote {path}  ({len(rows)} concepts, {n_deriv} with derivational forms)")
    for r in rows:
        if r["forms_derived"]:
            print(f"  {r['concept']:16s} +{r['forms_derived']}")


if __name__ == "__main__":
    main()
