#!/usr/bin/env python3
"""Build the first-100 ScientistQA lexical-debiasing intervention set.

Only the decisive (normally negative) constraint is rewritten.  Candidate
names, background clues, option order, and gold-answer metadata stay fixed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "question" / "shuffled_prepend_names_question.json"
OUTPUT = ROOT / "question" / "pretraining_debiased_first100_question.json"
REWRITE_VERSION = 2


# These descriptions intentionally omit the original proper name/acronym of
# each decisive entity.  They use compositional facts (granting body, subject,
# location, date, or duties) that are much less likely to have co-occurred
# verbatim with a candidate's name in pretraining text.
PARAPHRASED_CONSTRAINTS = [
    "However, they never received the Swedish-endowed annual distinction selected by the Stockholm medical faculty for a discovery in physiology or medicine.",
    "However, during that career this leader neither headed the federal ministry responsible for Germany's armed forces nor studied at the public university founded in 1919 in Germany's second-largest city.",
    "Despite this academic record, this person was not educated at the private science-and-technology university chartered in 1861 in Cambridge, Massachusetts.",
    "However, this astrophysicist never received the annual award established by an Anglo-American investor to honor work concerning life's spiritual dimension.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "However, despite holding a senior executive role in biotechnology, this person never held the position with ultimate responsibility for managing the company.",
    "Despite this academic record, this person was not educated at the private university founded in 1636 in Cambridge, Massachusetts.",
    "Despite this decorated career, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "However, this person was not educated at the private research university founded in 1890 on Chicago's South Side.",
    "Despite this influence, this person neither sat in the unelected upper chamber of the United Kingdom's Parliament nor led Britain's academy for the natural sciences.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite these honors, this scholar did not receive the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite this foundational role, he did not co-design the transport protocol that provides reliable ordered delivery between networked computers or the associated family of internetworking protocols.",
    "However, this person neither studied at the collegiate university founded in 1209 in eastern England nor served in the USSR's highest legislative body.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite this background, this person studied at neither the private university founded in 1740 in Philadelphia nor the Baltimore medical school opened in 1893 and named for its founding benefactor.",
    "However, this scientist never received the Israeli international prize first awarded in 1978 whose science categories comprise agriculture, chemistry, mathematics, medicine, and physics, alongside rotating awards in the arts.",
    "Notably, this scientist neither practiced the study of genes and heredity nor received Germany's major research prize administered by the country's central research-funding organization.",
    "Despite this decorated career, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite these accolades, this person was never selected for the US five-year, no-application fellowship popularly described as a 'genius grant.'",
    "Despite this background, this person studied at neither the private university founded in 1636 in Cambridge, Massachusetts, nor the private science-and-engineering institute founded in 1891 in Pasadena.",
    "However, this person's professional identity did not include the discipline concerned with numbers, quantities, structures, and rigorous proof.",
    "Despite this academic record, this person was not educated at the private science-and-technology university chartered in 1861 in Cambridge, Massachusetts.",
    "However, this person never received the international chemistry award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "However, this scientist never received the international chemistry award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "However, this scientist never received the international chemistry award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm medical faculty for a discovery in physiology or medicine.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or invention in physics.",
    "Despite these contributions, he never received the Swedish-endowed annual distinction selected by the Stockholm medical faculty for a discovery in physiology or medicine.",
    "Despite this impact, this person was not educated at the private science-and-engineering institute founded in 1891 in Pasadena, California.",
    "Despite his role in evolutionary science, he did not study at the collegiate university founded in 1209 in eastern England.",
    "However, this scientist never held the British royal-household post that serves as the monarch's senior adviser on astronomical matters.",
    "Notably, this scientist was not educated at the private science-and-technology university chartered in 1861 in Cambridge, Massachusetts.",
    "Despite these contributions, this person never received the international chemistry award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "Despite this influence, this person never received the Japanese foundation's prize, inaugurated in 1985, for fundamental advances in the natural sciences.",
    "However, this scientist never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or invention in physics.",
    "However, he was never appointed to the formal body of senior advisers to the British sovereign.",
    "Notably, this person did not attend the private science-and-technology university chartered in 1861 in Cambridge, Massachusetts.",
    "Notably, this physicist never received the international physics award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "Despite her impact, she never received the Swedish-endowed annual distinction selected by the Stockholm medical faculty for a discovery in physiology or medicine.",
    "However, this person's main specialty was not the branch of chemistry that explains chemical systems using physical principles and measurements.",
    "However, this person has not received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite these achievements, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "However, this person neither studied at the London science-and-engineering institution that received its royal charter in 1907 nor occupied the ancient crown-created chemistry chair at a British university.",
    "Despite these accolades, this scientist never received the large privately funded life-science award launched in 2013 by a group of technology entrepreneurs.",
    "Notably, this person studied at neither the private university in central New Jersey chartered in 1746 nor the private science-and-engineering institute founded in 1891 in Pasadena.",
    "However, this physicist never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or invention in physics.",
    "This person neither devised the globally linked hypertext system first proposed at CERN in 1989 nor developed its standard document-transfer protocol or page-markup language.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or invention in physics.",
    "However, this physicist neither studied at the major Moscow university founded in 1755 nor received the international physics award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "However, this person never received the Russian academy's highest distinction, a gold medal named for the eighteenth-century polymath who helped found Moscow's leading university.",
    "Despite these honors, this person neither attended the collegiate university in the English city associated with the Thames and Cherwell rivers nor received the international prize created in 2000 by a foundation bearing an Israeli philanthropist's name.",
    "Notably, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite these accolades, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "However, this person never received the international chemistry award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "Notably, this person studied at neither the private university founded in 1636 in Cambridge, Massachusetts, nor the public college founded in 1847 as New York City's first free institution of higher education.",
    "Notably, this person studied at neither the German university founded in 1527 in Hesse nor the Lower Saxony university established in 1734, and never received the German chemical society's medal named for a pioneer of sugar and purine chemistry.",
    "However, he never received the computer-architecture prize jointly presented since 1979 by the major US computing and electrical-engineering professional societies.",
    "Notably, this person never received the highest scientific award of France's national government research organization.",
    "However, this person never received the Israeli institute's prize recognizing major contributions to science, technology, human health, or peace.",
    "Despite his political role, he neither headed the French government department responsible for land warfare nor studied at the Paris engineering school founded during the French Revolution.",
    "However, this scientist studied at neither the private university in central New Jersey chartered in 1746 nor the private university founded in 1636 in Cambridge, Massachusetts.",
    "Despite this recognition, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite this legacy, this person neither taught at a university nor managed and interpreted a museum or scientific collection.",
    "Despite these honors, this person never received the Italian physical-science medal established in 1868 and named for an early researcher of electrical currents.",
    "However, this scientist did not attend the private university founded in 1636 in Cambridge, Massachusetts.",
    "Notably, this person did not receive the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Unlike some decorated peers, this person never received the British scientific academy's medal first awarded in 1731 for outstanding achievement in any field of science.",
    "Notably, this scientist did not study at the private university in central New Jersey chartered in 1746.",
    "However, this person never received the international physics award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "However, he never led Britain's academy for the natural sciences, an office elected by that society's fellows.",
    "Notably, this person neither studied at the Paris biomedical research center founded in 1887 and named for a pioneer of vaccination nor received the Swedish-endowed annual distinction selected for chemistry.",
    "Despite this career, this scientist neither studied at the private science-and-technology university chartered in 1861 in Cambridge, Massachusetts, nor received the international chemistry award established in 1978 by a foundation created by a German-born inventor and philanthropist.",
    "This person earned no degree from the Toronto university established as King's College by royal charter in 1827. They also earned no degree from the Paris science university that merged with Paris-Sorbonne in 2018 to form Sorbonne University.",
    "This person received neither the United States' highest presidential honor for scientific achievement, administered by the National Science Foundation, nor the mathematical-physics prize jointly awarded since 1959 by the American Institute of Physics and the American Physical Society.",
    "Despite this decorated career, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Notably, this scientist did not study at the Japanese national university established in 1949 in the port city west of Osaka.",
    "However, this person neither specialized in the scientific study of nonhuman primates nor attended the collegiate university founded in 1209 in eastern England.",
    "However, this scientist did not receive the Swedish-endowed annual distinction selected by the Stockholm medical faculty for a discovery in physiology or medicine.",
    "Despite this impact, this scientist attended neither the collegiate university founded in 1209 in eastern England nor the Scottish university founded in 1583.",
    "Despite these accolades, this person never received the European biomedical research prize founded in 1986 by a Geneva-based charitable foundation.",
    "However, this scientist never received the highest scientific award of France's national government research organization.",
    "Despite these honors, this person never received the Swedish-endowed annual distinction selected by the Stockholm medical faculty for a discovery in physiology or medicine.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Despite this academic journey, this person studied at neither the public university founded in 1853 in Gainesville nor the public university founded in 1848 in Wisconsin's capital.",
    "Despite these contributions, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or invention in physics.",
    "Despite these contributions to evolution, this scientist neither practiced the study of prehistoric life through fossils nor attended the London university college founded in 1826.",
    "However, this scientist never received the British scientific academy's medal first awarded in 1731 for outstanding achievement in any field of science.",
    "Despite this background, this person did not study at the private university founded in 1740 in Philadelphia.",
    "However, this person never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or invention in physics.",
    "Despite these chemistry contributions, this person never received the American chemical society's highest honor, a gold medal first awarded in 1923 and named for an English-American chemist.",
    "However, this scientist never received the American chemical society's highest honor, a gold medal first awarded in 1923 and named for an English-American chemist.",
    "Despite this academic background, this person did not study at the London science-and-engineering institution that received its royal charter in 1907.",
    "Notably, this scientist did not attend the private university founded in 1851 in Evanston, Illinois.",
    "However, this scientist never received the Swedish-endowed annual distinction selected by the Stockholm-based academy for a discovery or improvement in chemistry.",
    "Notably, this person did not study at the private university founded in 1636 in Cambridge, Massachusetts.",
    "Despite this background, this person did not attend the Lower Saxony university established in 1734 by the British king who was also Hanover's elector.",
    "Notably, this scientist was never selected for the US five-year, no-application fellowship popularly described as a 'genius grant.'",
]


def split_prompt(prompt: str) -> tuple[str, str]:
    """Return the prompt without its final constraint, plus that constraint."""
    marker = "\nQuestion:\n"
    if marker not in prompt:
        raise ValueError("Prompt lacks the expected Question marker")
    prefix, biography = prompt.split(marker, 1)
    sentences = re.split(r"(?<=[.!?])\s+", biography.strip())
    if len(sentences) < 2 or sentences[-1] != "Who is this person?":
        raise ValueError(f"Unexpected prompt ending: {prompt[-120:]}")
    original_constraint = sentences[-2]
    context = " ".join(sentences[:-2])
    return f"{prefix}{marker}{context}", original_constraint


def main() -> None:
    source_items = json.loads(SOURCE.read_text(encoding="utf-8"))[:100]
    if len(source_items) != 100 or len(PARAPHRASED_CONSTRAINTS) != 100:
        raise RuntimeError("Expected exactly 100 source items and 100 rewrites")

    output = []
    for index, (item, replacement) in enumerate(zip(source_items, PARAPHRASED_CONSTRAINTS)):
        stem, original_constraint = split_prompt(item["prompt"])
        revised = dict(item)
        revised["prompt"] = f"{stem} {replacement} Who is this person?"
        revised["original_prompt"] = item["prompt"]
        revised["original_decisive_constraint"] = original_constraint
        revised["paraphrased_decisive_constraint"] = replacement
        revised["intervention"] = "decisive_constraint_compositional_paraphrase"
        revised["rewrite_version"] = REWRITE_VERSION
        revised["source_index"] = index
        output.append(revised)

    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(output)} questions to {OUTPUT}")


if __name__ == "__main__":
    main()
