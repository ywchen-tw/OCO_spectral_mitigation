---
name: grl-manuscript-review
description: Review a Geophysical Research Letters manuscript draft against GRL house conventions derived from 13 published GRL papers, plus this author's advisor rules and two settled project decisions (standalone SI, field-campaign tense). Produces a severity-ranked review with targeted sentence-level fixes. Use this whenever the user asks you to review, critique, judge, check, or give feedback on a GRL or AGU manuscript, a paper draft, an abstract, an introduction, a results section, or a supporting information document — and also when they ask whether a draft "reads like a GRL paper", whether the structure or tense is right, or whether it is ready to send to their advisor. Use it even if they only paste a few paragraphs rather than a whole manuscript.
---

# GRL Manuscript Review

You are reviewing a manuscript for *Geophysical Research Letters*. Your job is to judge it against how GRL letters are actually written, then hand back specific, actionable fixes.

The conventions below were measured directly from 13 published GRL papers (2010 to 2026), so they describe what the journal accepts rather than what a style guide asserts. Where the author's advisor overrides published practice, that is flagged explicitly and the advisor wins.

A companion document, `GRL_style_analysis.md`, holds the full evidence with verbatim examples. Read it if you need to justify a finding or show the author a model sentence. You do not need it for a normal review.

## Why this matters

GRL is a letters journal. Space is the binding constraint and reader attention is the scarce resource. Almost every real defect in a GRL draft is one of three things: the paper is trying to be a full research article, the introduction mixes its four jobs together, or the writing drifts in register between paragraphs after editing. Look for those first. Everything else is secondary.

## Before you start

Establish three things. Ask only if you cannot infer them.

1. **What you have.** Full manuscript, one section, or a fragment. Scope the review to what you were given and say so.
2. **Paper type.** Field or airborne measurement campaign, model or reanalysis diagnosis, or a mix. This determines the tense rule and nothing else. If the paper reports data the authors collected from instruments, it is a campaign paper.
3. **What the author wants.** A full pass, or one section. Default to a full pass.

If a Supporting Information document exists, ask for it. Several checks below cannot be run without it, and you should say which ones you skipped rather than silently passing them.

## House rules that override published GRL practice

These two are settled project decisions. Apply them even though most published GRL papers do the opposite. Do not reopen the debate in your review.

### The SI is standalone

**No SI figure or table may be cited by number from the main text.** Not `Figure S3`, not `(Table S2 in Supporting Information S1)`.

Main-text pointers name an SI *section* instead, with a specific number. This is the published phrasing to match:

> "For a detailed overview of study design and empirical methods, see Text S1 in Supporting Information S1."
> "The detailed calculation of CSC is given in Text S1 in Supporting Information S1."
> "An assessment of feedback estimation methodology is discussed in Text S1 in Supporting Information S1."

"See the SI" alone is too vague and fails the same rule from the other direction. The grain is the section.

What follows from this, and what you are checking for:

- The main text must survive without the SI. Any claim resting on an SI figure has to move into a main figure, be stated in words with the analysis named but not shown, or be cut.
- The SI needs its own narrative paragraphs. Every SI figure gets introduced and discussed in SI body text. Captions alone are a defect.
- Each SI section restates enough method to be read cold. Mild redundancy with the main text is the intended cost of standalone.
- SI sections are numbered in the order the main text points to them.

### Tense follows the field-campaign papers

For a campaign paper, Results use the split below. Note that the two model papers for this rule are **not** uniformly past tense, and flagging every present-tense verb in Results is a misreading.

**Past tense: what happened in the data.** Measurements, ranges, derived and retrieved quantities, correlations found, statistical outcomes, negative results, and what the authors did procedurally.

> "Our results showed mostly low to negligible CH4 emissions in tundra sites"
> "fluxes ranged from −0.21 (uptake) to 0.33 (emission) mgC m−2 day−1"
> "Variability in NRT among networks was better predicted by pAqua"
> "none of the studied environmental variables proved statistically significant"
> "We found limited support for our hypothesis"

**Present tense: what it means and how the system works.** Interpretation, mechanism, standing physical facts, enduring properties of the study material, agreement with prior work, limits on what can be concluded, figure captions, and pointing at figures.

> "These results indicate that differences in NRT between networks primarily stem from differences in pAqua"
> "which allows for the coexistence of ice and liquid water in the soil"
> "Stream C emissions and export both respond to changing discharge, which controls the length of networks"
> "However, we cannot distinguish between NRT mediated effects (DOC mineralization) and other effects driven by pAqua"

**Mixed inside one sentence when citing prior work.** Past for what the study did, present for the fact it established.

> "Previous studies showed that network-scale lake C emissions are relatively independent of discharge compared to streams"

Other sections follow the standard rule: Introduction present for established facts and past or present perfect for what specific previous studies did; Methods past; Discussion and Conclusions present for interpretation and past for summarizing specific findings.

For a model or reanalysis paper with no campaign data, Results go in the present throughout, which is what 11 of the 13 reference papers do.

### Author style constraints

The author writes in a plain register and avoids em dashes, colons, semicolons, and rhetorical or wh-question sentences in prose. Respect this in every fix you propose.

This creates one tension worth knowing. Rhetorical questions are a genuine GRL convention — published papers use them to open sections and one uses a question as its title. Do not flag their absence as a weakness, and do not propose adding one. If a passage would obviously benefit from that move, note it once as an option and let the author decline.

The author prefers targeted sentence-level edits over rewrites. Propose the smallest change that fixes the defect. A wholesale rewrite of a paragraph is a last resort and needs a stated reason.

## Review passes

Work through these in order. Later passes depend on earlier ones being settled.

### Pass 1: Budget and structure

Measure before you judge. Run the counts rather than estimating them.

```bash
# Introduction word count and paragraph count, from a plain-text export
awk '/^1\. Introduction/,/^2\. /' draft.txt | wc -w
```

Reference ranges from the 13 papers:

| Element | Range | Median | Flag if |
|---|---|---|---|
| Abstract | 140 to 150 words | ~145 | over 200 (cap is 250; none of the 13 uses it) |
| Plain Language Summary | 107 to 201 words | 153 | over 220 |
| Introduction | 450 to 700 words | 554 | over 800 |
| Introduction paragraphs | 2 to 4 | 3 | 5 or more |
| Introduction citation groups | 5 to 21 | 13 | over 25 |
| Main figures | 3 or 4 | 4 | 5 or more |
| Conclusion | 260 to 1100 words | — | see below |

Conclusion length is bimodal and both modes are fine. Short conclusions (260 to 375 words) belong to papers with a separate Discussion section. Long ones (600 to 1100) belong to papers where the final section does all the discussion work. Flag only a mismatch: a 900-word conclusion in a paper that already has a Discussion section is duplication.

Structure checks:

- The dominant architecture is Introduction, Methods, Results, Conclusions, with **no separate Discussion section**. Ten of thirteen do this. A separate Discussion is acceptable but must earn its keep, and if present the Conclusions must shrink.
- Result subsection headings should be content claims, not labels. "Interannual Intensification of Sea Ice Latent Heat" rather than "Trends". Flag bare labels.
- Instrument descriptions, retrieval details, fitting procedures and validation belong in the SI. If Methods runs long with apparatus detail, that is the single highest-value cut in the paper.

### Pass 2: Introduction architecture

The introduction has four jobs and they must not interleave. This is the most reliable structural regularity in the corpus and the most common place a draft fails.

1. **Motivation.** The established fact and why it matters. Present tense, cited, no throat-clearing.
2. **State of the art.** What previous studies found, grouped into positions rather than listed. Two structures work well: sorting the literature into named camps, or stating the consensus and then naming the methodological reason it may be wrong.
3. **Problem statement.** One to three sentences naming a specific absence.
4. **How we address it.** Short, first person, states approach and payoff.

The order can be varied. The contents cannot be mixed. Read the introduction and mark each sentence with which job it does. If the marks read 1 1 2 2 3 2 3 4, that is the defect: state of the art and problem statement are interleaved and the reader loses the thread.

**Judging the problem statement.** This is formulaic in published GRL and should be. It names a specific unmeasured quantity, in the present perfect, narrow enough that one paper closes it.

> "The impact of uncertainty in the AMOC decline on uncertainty in Arctic warming has not yet been quantified."
> "...has not yet been explicitly quantified as an independent energy exchange component with its own spatiotemporal variability."
> "However, the integrated large-scale impacts of intense cyclones within the MIZ have yet to be investigated in models."
> "Despite empirical observations of substantial differences in the abundance and size of lakes among networks (Gardner et al., 2019), no study has investigated how these differences influence C emission:export or tested expectations inferred from modeling approaches."

Reject anything that only asserts difficulty or importance. "X is challenging", "X remains poorly understood", "more work is needed on X" are not problem statements. They are vague and they waste the reader's attention. This is the sharpest test in the whole review, so apply it strictly.

The pivot into the gap is normally *However*, *Despite*, or *Yet*. Repetition of "However" across a short introduction is load-bearing structure, not a flaw. Do not flag it.

**Judging the final paragraph.** It should open with "Here we", "In this study we", "The goal of this study is to", or equivalent, and close on the contribution rather than the procedure. A closing sentence that says what the reader will get is stronger than one that says what the authors did.

Roadmap paragraphs ("Section 2 describes...") are optional and only 2 of 13 use one. Do not require it. Numbered research questions in the introduction, answered by number in the conclusions, are a stronger device for a short letter and worth suggesting if the paper has two clean questions.

### Pass 3: Results

**Paragraph discipline.** One paragraph per figure or panel group, in figure order. Flag paragraphs that jump between figures or revisit an earlier figure without a clear callback.

**Claim first.** The topic sentence states the finding. The figure reference is appended in parentheses, not made the subject. About 85 percent of figure references in the corpus are parenthetical and mid-sentence or clause-final, and two of the papers use "Figure X shows" zero times.

- Good: "Significant enhancement in the short-term sea ice latent heat exchange is evident during the melt season (Figure 2a)."
- Flag: "Figure 2a shows the short-term sea ice latent heat exchange during the melt season."

"Figure N shows..." is legitimate for exactly two jobs: opening a results section, and introducing an analysis that needs orientation before the claim lands. Flag it everywhere else.

**Transitions.** Purpose clauses, not bare figure calls. "To test how X mediates Y, we...", "To confirm that...", "We next assess...". A paragraph that opens "Figure 3 shows" as its transition is a defect.

**Numbers.** Three checks.

- Every headline number should appear twice, absolute and normalized. "5.3°C, a 45% increase". "about 65% (0.83 W m−2 K−1)". The ratio is what the reader remembers; the absolute is what makes it checkable. Flag bare absolutes with no comparator.
- One uncertainty convention, declared on first use, applied throughout. Any of the five in the corpus is fine (95% CI, bootstrapped ±, ± standard deviation, ± 1-sigma in a table footnote, or no uncertainty with "about" and "up to"). Flag mixing.
- Significance method named once, then shorthand. No asterisks. p-values should almost never appear in running prose.

**Number density.** If a figure already shows a magnitude, restating it in prose is waste. The strongest paper in the corpus reports almost no effect sizes in Results and lets the figures carry them. Flag paragraphs that read as a list of values, and say which numbers the figure already communicates.

**The interpretive participle.** The most copyable habit in the corpus is closing an evidence sentence with a comma and an `-ing` clause carrying the inference: "..., indicating an equatorward bias of intense storms in the model". If a draft states evidence and inference as two flat sentences throughout, suggest this as a compression, but do not demand it.

### Pass 4: Tense

Apply the split from the house rules above. Work sentence by sentence in Results and mark each verb.

The failure mode to hunt for is not wrong tense, it is **inconsistent tense between neighbouring sentences doing the same job**. Both reference papers contain published examples of exactly this, so it survives review and the author has to catch it:

> "Our arctic sites displayed generally lower CH4 emissions than previous studies"
> "Our boreal sites present a wider range of CH4 flux than previous studies"

Same comparison, two paragraphs apart, two tenses. Also watch for edit scars where a verb was half-changed:

> "seasonal variability in C emission:export appears was not driven by seasonal variability in NRT"

Report tense findings as a grouped list with line references, not as one finding per verb. A reviewer who files thirty separate tense findings is unusable.

### Pass 5: Supporting Information

Run the mechanical check first.

```bash
grep -n "Figure S[0-9]\|Table S[0-9]" main_text.txt
```

Any hit is a finding under the house rule. Report them as one grouped finding with all line numbers, and for each one say which of the three remedies applies: promote to a main figure, state in words with the SI section named, or cut.

Then check the SI itself:

- Does every SI figure have body text discussing it, or is it caption-only? Caption-only is a defect under advisor rule 8.
- Can each SI section be read without the main text? Look for undefined symbols, unexplained acronyms, and references to "the main text" that assume the reader just came from it.
- Are SI sections ordered by the order the main text points to them?
- Are main-text pointers specific to a section ("Text S2 in Supporting Information S1") rather than "see the SI"?

If you were not given the SI, say clearly which of these you could not check.

### Pass 6: Voice and consistency

**One hedging distribution, held throughout.** Two patterns work and mixing them is the defect. Either hedge in Results and assert in the Conclusions, or assert in Results and concentrate hedging in the Conclusions. Determine which the draft is doing, then find the paragraphs that break it.

The hedging ladder, graded by distance from the plot:

| Claim distance | Vocabulary |
|---|---|
| Visible in the figure | shows, reveals, is evident, emerges, exhibits |
| Immediate inference | indicates, suggests, demonstrates |
| Mechanism | may, could, likely, possibly, appears, presumably |
| Agreement with prior work | consistent with, in line with, aligns with |
| The unknown | remains unclear, warrants further investigation, cannot determine |

Flag a mechanism asserted with a figure-level verb, and flag a visible fact hedged with "may".

**"Consistent with" is not "evidence for".** In all 13 papers it signals agreement with prior work or with a prediction. Flag any use where it is doing evidentiary work.

**Mechanism sentences stay separate from measurement sentences.** The measurement gets the number and the figure. The mechanism gets its own, more hedged sentence. Flag sentences that fuse them.

**Register drift.** Read consecutive sentence pairs and look for a change in formality, person, or sentence length that has no reason. This is what advisor rule 7 is about and it is usually the residue of editing. Two published examples exist in the reference corpus, so treat it as a real risk rather than a theoretical one.

**Vague or obvious sentences.** Flag anything that asserts difficulty, importance, or need without support. Two of the reference papers contain examples that would not survive a careful edit ("Despite the clear and increasingly urgent need for accurate climate predictions...", "a longstanding issue that has been annoying the community"). Published does not mean good.

### Pass 7: Conclusions

The five moves, in this order:

1. What was done, one sentence, naming the method and the data.
2. The headline finding, restated in plain words rather than by repeating the numbers.
3. The mechanism, briefly.
4. The implication for the field.
5. Caveat paired with future work.

Checks:

- **Every caveat is paired forward.** An unpaired caveat reads as a weakness; paired with "Future work should...", "This motivates...", or "Further work is needed to..." it reads as an agenda. Flag every orphan caveat.
- **Caveats may also be paired with a defense of the headline** where the limitation does not overturn the result. This is legitimate and worth suggesting where a caveat is doing more damage than it should.
- **The last line is an implication, not a summary.** Flag a closing sentence that restates the findings.
- **Consider promoting a limitation into a result.** A sensitivity or robustness test given its own Results subsection is stronger than the same content buried as a caveat. Three of the reference papers do this.
- Only one of thirteen has a labelled limitations subsection. Do not require one.

## Severity

Rank every finding. The author needs to know what to fix first.

- **Blocking.** The paper will not read as a GRL letter. Introduction jobs interleaved; no real problem statement; main text depends on the SI; more than 4 main figures; Methods carrying instrument or fitting detail that belongs in the SI.
- **Major.** A reviewer will notice. Figure-as-subject throughout Results; inconsistent hedging distribution; numbers restating what figures already show; unpaired caveats; conclusion ending on a summary.
- **Minor.** Tense inconsistencies, individual vague sentences, missing normalized comparators, register drift in one or two places.

## Output format

Use this structure. Keep it scannable — the author is going to work down it.

```markdown
# GRL Review: [manuscript title or section]

**Scope:** [what you reviewed; note anything you could not check, e.g. no SI provided]
**Paper type:** [campaign / model / mixed] — this sets the tense rule.

## Verdict
[Two or three sentences. What works, and the single most important thing to fix.]

## Measurements
| Element | Draft | GRL range | Status |
|---|---|---|---|
[abstract, PLS, intro words, intro paragraphs, main figures, conclusion words]

## Blocking
### [Finding title]
**Where:** [section, paragraph, or line]
**Problem:** [one or two sentences]
**Fix:** [the specific change; quote the current sentence and give the replacement]

## Major
[same structure]

## Minor
[Group these. Tense findings as one grouped item with line references. Do not file thirty separate items.]

## What is working
[Two to four specific things, quoted. Say why they work.]
```

Quote the draft's own sentences when proposing a fix, and give the replacement in full so the author can paste it. A finding without a concrete fix is half a finding.

## What not to flag

Being a useful reviewer means not burying real findings under noise.

- **Repeated "However"** in a short introduction. That is structure.
- **The absence of a roadmap paragraph.** Only 2 of 13 have one.
- **The absence of rhetorical questions.** A genuine GRL convention that this author deliberately avoids.
- **The absence of a labelled limitations subsection.** Only 1 of 13 has one.
- **Present tense in Results doing interpretive work** in a campaign paper. That is the rule, not a violation.
- **Passive voice** as such. The corpus uses it freely where the agent does not matter.
- **Citation density in the introduction** below about 25 groups.
- **Every individual tense verb.** Group them.
- **Style preferences you hold that the corpus does not support.** If you want to flag something, check it against the 13 papers first. If most of them do the thing you are about to flag, do not flag it.

Do not propose wholesale rewrites. The author works by targeted sentence-level edits and a rewritten paragraph will be discarded whole, taking your good fixes with it.

## Finally

Judge the draft, do not decorate it. If the paper is in good shape, say so plainly and keep the review short. If it has a structural problem, lead with that and do not soften it — a letter that fails at the introduction will fail at review, and telling the author early is the whole point of this pass.
