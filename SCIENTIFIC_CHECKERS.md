# SciBench scientific-sanitizer pilot

This branch instruments 40 curated scientific invariants (38 root-cause
families) over `Bio.SeqUtils`, the `Bio.PDB` geometry primitives
(`vectors`, `qcprot`, `Superimposer`), `Bio.Align.PairwiseAligner`,
`Bio.Phylo.TreeConstruction` (neighbour joining / UPGMA), and
`Bio.motifs` (position-specific scoring matrices). Every
invariant is a physical, chemical, geometric, thermodynamic, or algebraic
symmetry law whose violation has a domain consequence -- pure arithmetic
identities, range checks, and generic software-correctness assertions are
deliberately out of scope. The original public code at commit
`dc262b5c437e07a8cc1cfb8a734c0d84a4434b23` is unchanged in behaviour; this
branch adds opt-in runtime checks.

Checkers are inactive during normal use. Set `SCIBENCH_TRIGGER_LOG` to a file
before running tests:

```bash
SCIBENCH_TRIGGER_LOG=/tmp/triggers.jsonl python -m pytest <test files>
```

Each line of the log is `{"checker_id": "BP-SEQ-NNN"}`. Deduplicate IDs for
the raw score and map IDs to root-cause families using
`SCIENTIFIC_CHECKERS.json`. Run only the submitted tests, or subtract IDs seen
in an identical baseline run, so existing tests receive no credit.

Each checker in `SCIENTIFIC_CHECKERS.json` lists its scientific quantity,
valid-input precondition, invariant, observation point, alarm predicate, and
rationale. Reason backward from the alarm to candidate inputs.

The logger records rather than raises: one test may trip several alarms, and
normal return values, exceptions, and numerical results are unchanged. A
checker only re-calls the public API on a transformed input or reads values
the production code already computed; it never re-implements the formula it
checks, and never touches the trigger log directly from a test.
