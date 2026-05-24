
================================================================================
                PAN26 Paper — Peer Review Report
================================================================================

Paper: "Query-Chunk Provenance Modeling for Source Retrieval in Generative Plagiarism Detection"
Reviewer: Automated Citation Check + Methodology Review

================================================================================
1. CITATION ERRORS (CRITICAL)
================================================================================

ERROR 1: Ref [3] — Wrong author list
  Current:  Potthast M, Gollub T, Hagen M, et al. ... CEUR Vol.1180, pp.975-997
  Correct:  Potthast M, Hagen M, Beyer A, Busse M, Tippmann M, Rosso P, Stein B.
            CEUR Vol.1180, pp.845-876 (2014)
  Note:     Gollub T was an author of the 5th competition overview (PAN 2013),
            NOT the 6th (PAN 2014). Pages are also incorrect.

ERROR 2: Ref [7] — Duplicate citation
  This is a duplicate of Ref [3] with the same wrong author list.
  Recommendation: DELETE Ref [7] entirely. Re-number subsequent refs.

ERROR 3: Ref [6] — Wrong venue/volume
  Current:  ACM SIGIR Forum, 2023, 57(2): 1-10
  Correct:  Fröbe M, Bevendorff J, Gienapp L, et al. "The Information Retrieval
            Experiment Platform." In: SIGIR 2023 (demo), ACM, pp.3180-3190.
            DOI: 10.1145/3539618.3591888
  Note:     There may also be a 2024 SIGIR Forum article; verify which one to cite.

ERROR 4: Ref [1] — Missing page numbers
  Current:  CEUR Workshop Proceedings, Vol. 4038, 2025 (no pages)
  Verified: pp. 3575-3585. Please ADD page numbers.

ERROR 5: Ref [2] — Author list truncated
  Current:  "et al." — but this paper has 16 authors
  Recommendation: Either list all authors or use "et al." consistently.
  Also note: this is an Extended Abstract (4 pages). The full overview paper is:
  Bevendorff et al. "Overview of PAN 2025: Voight-Kampff Generative AI Detection..."
  CLEF 2025, LNCS 16089, pp.388-411. Consider citing the full version instead.

ERROR 6: Ref [3] and Ref [4] — Potthast 2014 name confusion
  The 6th competition was at CLEF 2014, not "CLEF 2013" as some sources use.
  Verify: the "6th International Competition" ran in 2014.

================================================================================
2. STRUCTURAL & CONTENT REVIEW
================================================================================

ISSUE 1: Missing Figure 1
  The method section references Figure 1 but only has placeholder text.
  A clear architecture diagram is ESSENTIAL for the method section.
  Even a simple ASCII/flowchart diagram would suffice for v1.

ISSUE 2: Section 3.4 — Performance optimization
  This section is detailed and valuable, but its placement between the method
  description and experiments disrupts flow. Consider moving to an appendix
  or integrating into Section 3.3.2.

ISSUE 3: No comparison with published PAN 2025 participant results
  The paper states "all participants significantly underperformed" but doesn't
  quantify. At minimum, include participant R@10 values from the overview paper.

ISSUE 4: Ablation with only 200 queries
  The ablation (Table 3) uses only 200 queries. While informative, explicitly
  note this as a limitation. Consider running on 800q for robustness.

ISSUE 5: No statistical significance tests
  R@10 improvements of +0.05 on 800 samples — is this significant?
  Recommend adding McNemar's test or paired bootstrap test.

ISSUE 6: English abstract quality
  Minor: "often composite content" → "often composite" (verb form).
  "composite" as a verb is uncommon; use "synthesize" or "combine".

================================================================================
3. CONTRIBUTION ASSESSMENT
================================================================================

Strengths:
  + Novel paradigm shift from document-query to provenance-graph modeling
  + Lightweight implementation (numpy-only, TIRA-deployed)
  + Consistent results across two independent samples
  + Well-documented performance optimization (20x speedup)

Weaknesses:
  - No full-dataset evaluation (only 800/1200 samples from 42K)
  - No cross-domain validation
  - Ablation on small sample (200q)
  - No comparison with dense retrieval baselines in the same table

Overall: The contribution is solid for a shared-task paper. The provenance
modeling perspective is the key intellectual contribution. Recommend addressing
citation errors before submission.

================================================================================
4. SUMMARY OF REQUIRED FIXES
================================================================================

Critical (must fix):
  [ ] Fix Ref [3] author list (remove Gollub, add Beyer, Busse, Tippmann, Rosso, Stein)
  [ ] Fix Ref [3] pages (845-876, not 975-997)
  [ ] Delete Ref [7] (duplicate), renumber refs 8-14 → 7-13
  [ ] Add page numbers to Ref [1] (3575-3585)
  [ ] Fix Ref [6] venue (SIGIR 2023 demo, not SIGIR Forum 57(2))

Recommended:
  [ ] Add Figure 1 (method architecture diagram)
  [ ] Add participant baseline results from PAN 2025 overview
  [ ] Move Section 3.4 to appendix or integrate into 3.3.2
  [ ] Note ablation sample size limitation
  [ ] Fix English abstract: "composite" → "synthesize/combine"
================================================================================
