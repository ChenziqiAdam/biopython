"""Runtime scientific-invariant trigger collection for the SciBench pilot.

This module is private and inactive unless ``SCIBENCH_TRIGGER_LOG`` names an
output file. Each JSON-lines record represents one observed invariant alarm;
the evaluator deduplicates checker IDs and maps them to root-cause families.

Design rule (see ``scientific_bug_finding/METHODOLOGY.md`` section 2): a checker
may only (a) call a public API a second time on a transformed input and/or
(b) read values already computed by production code, then compare. A checker
never re-implements the scientific formula it is checking, never raises, and
never changes a return value, exception, or numerical result.

Scope rule (METHODOLOGY.md section 1): every sanitizer in this bank guards a
**scientific** invariant -- a physical, chemical, geometric, or thermodynamic
law whose violation has a domain consequence. Pure arithmetic identities,
range/finiteness checks, lookup-table round trips, and generic software
correctness are explicitly out of scope and are not instrumented here.
"""

import itertools
import json
import math
import os
import threading

_TOL = 1e-9

# Re-entrancy guard: a checker that calls a public API a second time must not
# trigger the same checker recursively.
_active = threading.local()


def enabled():
    """Return True when the pilot evaluator requested trigger collection."""
    return bool(os.environ.get("SCIBENCH_TRIGGER_LOG"))


def trigger(checker_id):
    """Atomically append one checker ID to the configured JSON-lines log."""
    path = os.environ.get("SCIBENCH_TRIGGER_LOG")
    if not path:
        return
    payload = json.dumps({"checker_id": checker_id}, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload.encode("ascii"))
    finally:
        os.close(descriptor)


def trigger_if(condition, checker_id):
    """Record the checker ID when condition is true."""
    if condition:
        trigger(checker_id)


def _isclose(a, b, tol=_TOL):
    """Relative+absolute closeness. ``tol`` is used as BOTH rel_tol and
    abs_tol, so it is only meaningful for ``tol`` well below 1 -- a fixed
    round-off tolerance. For a magnitude-scaled tolerance (which can exceed 1)
    use ``_within`` instead: passing a large value here as ``rel_tol`` would
    make almost anything compare equal.
    """
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _within(a, b, atol):
    """True when ``a`` and ``b`` differ by at most the absolute tolerance
    ``atol``. Use this (not ``_isclose``) whenever ``atol`` is derived from the
    magnitude of the compared quantities (methodology 8.3) and may be >= 1.
    """
    return abs(float(a) - float(b)) <= atol


def _guard(checker_id):
    """Wrap a checker body so an internal error never disturbs production.

    A checker that itself errors is a curator bug, not a science alarm; it must
    not change program behaviour. We swallow it silently.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if getattr(_active, "flag", False):
                return
            _active.flag = True
            try:
                func(*args, **kwargs)
            except Exception:
                pass
            finally:
                _active.flag = False

        wrapper.__name__ = func.__name__
        return wrapper

    return decorator


# --- ProtParam.flexibility ------------------------------------------------


# --- ProtParam.protein_scale -------------------------------------------------


@_guard("protein_scale_output")
def check_protein_scale_output(sequence, param_dict, window, edge, scores):
    """BP-SEQ-004: even-window scale profile reversal equivariance (pure API).

    Position-symmetry law: the amino-acid scale profile is a windowed
    average, so reversing the sequence must reverse the profile.
    """
    if window <= 0 or window % 2 or not sequence:
        return
    if not all(r in param_dict for r in sequence):
        return
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    reverse_scores = ProteinAnalysis(sequence[::-1]).protein_scale(
        param_dict, window, edge
    )[::-1]
    differs = len(scores) != len(reverse_scores) or any(
        not _isclose(a, b) for a, b in zip(scores, reverse_scores)
    )
    trigger_if(differs, "BP-SEQ-004")


# --- ProtParam.instability_index --------------------------------------------


# --- IsoelectricPoint ------------------------------------------------------


@_guard("charge_monotonicity")
def check_charge_monotonicity(sequence):
    """BP-SEQ-011: titration physics -- a polyprotic molecule's net charge is a
    monotonically non-increasing function of pH. A rise anywhere on the curve
    means the charge model is inconsistent with acid/base equilibrium.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    analysis = ProteinAnalysis(sequence)
    grid = [i / 4 for i in range(0, 57)]
    charges = [analysis.charge_at_pH(pH) for pH in grid]
    rises = any(b > a + 1e-9 for a, b in zip(charges, charges[1:]))
    trigger_if(rises, "BP-SEQ-011")


# --- SeqUtils.molecular_weight ----------------------------------------------


@_guard("water_mass_consistency")
def check_water_mass_consistency(monoisotopic):
    """BP-SEQ-046: the average-mass condensation water (18.0153) subtracted per
    backbone bond in ``molecular_weight`` must equal the mass of an H2O
    molecule assembled from the repository's own standard atomic-weight table,
    ``IUPACData.atom_weights`` (2 * H + O). This is a consistency check between
    a hard-coded constant in one function and the authoritative element table
    the rest of Biopython uses; the curator does not know whether they agree.
    """
    if monoisotopic:
        return
    from Bio.Data import IUPACData

    h = IUPACData.atom_weights["H"]
    o = IUPACData.atom_weights["O"]
    trigger_if(not _isclose(18.0153, 2 * h + o, tol=5e-4), "BP-SEQ-046")


# --- MeltingTemp ---------------------------------------------------------

@_guard("tm_gc_monotonicity")
def check_tm_gc_monotonicity(seq, temperature, valueset, userset, Na, K, Tris,
                             Mg, dNTPs, saltcorr, mismatch):
    """BP-SEQ-019: duplex thermodynamics -- a G:C pair contributes three
    hydrogen bonds versus two for A:T, so at fixed length raising %GC cannot
    lower the melting temperature. The two comparison endpoints are computed
    with the *same* value set and salt/mismatch parameters as the production
    call, so the invariant holds for every built-in value set.

    Precondition: the empirical GC term must be increasing in %GC net of any
    salt correction. Every built-in ``valueset`` (1-8) satisfies this, but
    ``Tm_GC`` also lets the caller override the coefficients through
    ``userset = (A, B, C, D)`` where ``B`` is the per-%GC contribution; a
    caller passing ``B <= 0`` has explicitly inverted the physics, so the
    law's premise no longer holds.

    Owczarzy salt-correction methods 6 and 7 add a term
    ``(4.29*f_GC - 3.95)*1e-5*ln[Na+]`` that is itself GC-dependent and, below
    1 M Na+, *decreases* with %GC (audit 5): the checker's precondition must
    bound ``B`` against that term's span, not merely require ``B > 0`` --
    otherwise a small positive ``B`` can be outweighed by the salt term and the
    checker fires on an input where Tm genuinely is non-monotone in %GC, which
    is correct Biopython behaviour, not a defect.
    """
    from Bio.SeqUtils.MeltingTemp import Tm_GC, salt_correction

    text = str(seq).upper().replace("U", "T")
    if len(text) < 4 or set(text) - set("ACGT"):
        return
    if userset is not None:
        try:
            b = float(userset[1])
            if b <= 0.0:
                return
            if saltcorr in (6, 7):
                salt_kw = dict(Na=Na, K=K, Tris=Tris, Mg=Mg, dNTPs=dNTPs,
                                method=saltcorr)
                salt_span = abs(
                    salt_correction(seq="G" * len(text), **salt_kw)
                    - salt_correction(seq="A" * len(text), **salt_kw)
                )
                if b * 100.0 <= salt_span:
                    return
        except (TypeError, IndexError, ValueError):
            return
    kw = dict(valueset=valueset, userset=userset, Na=Na, K=K, Tris=Tris, Mg=Mg,
              dNTPs=dNTPs, saltcorr=saltcorr, mismatch=mismatch)
    tm_lower = Tm_GC("A" * len(text), **kw)
    tm_higher = Tm_GC("G" * len(text), **kw)
    # slack scales with the temperature magnitude (methodology 8.3)
    slack = 1e-9 * max(1.0, abs(temperature), abs(tm_lower), abs(tm_higher))
    trigger_if(
        tm_lower - temperature > slack or temperature - tm_higher > slack,
        "BP-SEQ-019",
    )
    trigger_if(tm_lower - tm_higher > slack, "BP-SEQ-019")


@_guard("tm_nn_revcomp")
def check_tm_nn_revcomp(original_seq, c_seq, shift, selfcomp, nn_table, saltcorr,
                        Na, K, Tris, Mg, dNTPs, dnac1, dnac2, temperature):
    """BP-SEQ-028: a DNA/DNA duplex and its reverse complement are the same
    physical molecule read from the other strand, so the nearest-neighbor
    melting temperature is invariant under reverse complementation of the
    primer (all other parameters held fixed).

    The re-call forwards every Tm_NN argument that changes the result,
    including the strand concentrations dnac1/dnac2 (methodology 8.1); a
    partial forward would compare the caller's Tm against a default-dnac Tm
    and fire spuriously.
    """
    if c_seq is not None or shift or selfcomp or nn_table is not None:
        return
    from Bio.Seq import Seq
    from Bio.SeqUtils.MeltingTemp import Tm_NN

    text = str(original_seq).upper().replace("U", "T")
    if len(text) < 2 or set(text) - set("ACGT"):
        return
    rc = str(Seq(text).reverse_complement())
    rc_temp = Tm_NN(rc, saltcorr=saltcorr, Na=Na, K=K, Tris=Tris, Mg=Mg,
                    dNTPs=dNTPs, dnac1=dnac1, dnac2=dnac2)
    tol = 1e-9 * max(1.0, abs(temperature), abs(rc_temp))
    trigger_if(not _within(rc_temp, temperature, tol), "BP-SEQ-028")


@_guard("tm_nn_salt")
def check_tm_nn_salt(original_seq, c_seq, shift, selfcomp, nn_table, saltcorr,
                     Na, K, Tris, Mg, dNTPs, dnac1, dnac2, temperature):
    """BP-SEQ-029: counterion screening -- raising [Na+] stabilises a duplex, so
    it does not lower Tm_NN (salt-correction methods 1-4).

    The re-call forwards every non-salt Tm_NN argument (dnac1/dnac2 included,
    methodology 8.1) and varies only [Na+]; a partial forward would compare a
    default-concentration Tm against the caller's and fire spuriously. As with
    BP-SEQ-028, a non-default nn_table is excluded rather than silently
    dropped: the re-call must use the caller's own NN parameter table, not
    Tm_NN's default, or it compares Tm values computed under two different
    thermodynamic models rather than probing the salt term in isolation.
    """
    if (c_seq is not None or shift or selfcomp or nn_table is not None
            or saltcorr not in (1, 2, 3, 4) or K or Tris or Mg or dNTPs):
        return
    from Bio.SeqUtils.MeltingTemp import Tm_NN

    text = str(original_seq).upper().replace("U", "T")
    if len(text) < 2 or set(text) - set("ACGT") or Na <= 0:
        return
    lower = Tm_NN(text, saltcorr=saltcorr, Na=Na / 2.0, dnac1=dnac1, dnac2=dnac2)
    higher = Tm_NN(text, saltcorr=saltcorr, Na=Na * 2.0, dnac1=dnac1,
                   dnac2=dnac2)
    slack = 1e-9 * max(1.0, abs(temperature), abs(lower), abs(higher))
    trigger_if(lower - temperature > slack or temperature - higher > slack,
               "BP-SEQ-029")
    trigger_if(lower - higher > slack, "BP-SEQ-029")


@_guard("salt_correction_monotonicity")
def check_salt_correction(Na, K, Tris, Mg, dNTPs, method, seq, corr):
    """BP-SEQ-030: for methods 1, 3, 4 the salt-correction term is
    k * log10([Na+] in M): monotonically increasing in [Na+] (more counterion
    screening -> higher Tm) and exactly 0 at the 1 M reference state.
    """
    if method not in (1, 3, 4) or K or Tris or Mg or dNTPs or Na <= 0:
        return
    from Bio.SeqUtils.MeltingTemp import salt_correction

    lower = salt_correction(Na=Na / 2.0, method=method, seq=seq)
    higher = salt_correction(Na=Na * 2.0, method=method, seq=seq)
    trigger_if(not (lower <= corr <= higher), "BP-SEQ-030")
    at_one_molar = salt_correction(Na=1000.0, method=method, seq=seq)
    trigger_if(abs(at_one_molar) > 1e-9, "BP-SEQ-030")


# --- CodonAdaptationIndex ------------------------------------------------


def _cai_contributing_weights(index, sequence):
    """Shared scan: table weights of the codons that contribute to CAI.

    Mirrors calculate()'s own exclusion logic (ATG/TGG and untabulated stop
    codons contribute nothing) so the checkers below observe exactly the
    codon set the production code itself used, without recomputing CAI.
    """
    text = str(sequence).upper()
    if not text or len(text) % 3 or set(text) - set("ACGT"):
        return None
    codons = [text[i:i + 3] for i in range(0, len(text), 3)]
    weights = []
    for codon in codons:
        if codon in ("ATG", "TGG"):
            continue
        try:
            weights.append(index[codon])
        except KeyError:
            if codon in ("TGA", "TAA", "TAG"):
                continue
            return None
    return weights or None


@_guard("cai_geometric_mean_bounds")
def check_cai_geometric_mean_bounds(index, sequence, result):
    """BP-SEQ-050: CAI is a geometric mean of the contributing codons'
    table weights. A geometric mean of a finite set of positive numbers can
    never fall outside the closed interval [min, max] of those numbers,
    regardless of the table's content or the sequence's composition. A
    reported value outside that envelope means the length-normalization or
    the log/exp aggregation has gone wrong.
    """
    if result is None or not math.isfinite(result):
        return
    weights = _cai_contributing_weights(index, sequence)
    if not weights:
        return
    lo, hi = min(weights), max(weights)
    tol = 1e-9 * len(weights) * max(1.0, hi)
    trigger_if(result < lo - tol or result > hi + tol, "BP-SEQ-050")


@_guard("cai_concatenation_additivity")
def check_cai_concatenation_additivity(index, sequence, result):
    """BP-SEQ-051: codon boundaries are exactly preserved across
    concatenation of two in-frame sequences, so the log-mean of the whole
    must equal the length-weighted average of the log-means of the parts.
    Probes the split at the sequence's own midpoint (rounded down to a
    codon boundary) against two independent calculate() calls.
    """
    if result is None or not math.isfinite(result) or result <= 0.0:
        return
    text = str(sequence).upper()
    if not text or len(text) % 3 or set(text) - set("ACGT"):
        return
    n_codons = len(text) // 3
    if n_codons < 2:
        return
    split = (n_codons // 2) * 3
    if split == 0 or split == len(text):
        return
    a, b = text[:split], text[split:]

    weights_a = _cai_contributing_weights(index, a)
    weights_b = _cai_contributing_weights(index, b)
    if not weights_a or not weights_b:
        return
    l_a, l_b = len(weights_a), len(weights_b)

    from Bio.SeqUtils import CodonAdaptationIndex  # noqa: F401

    try:
        cai_a = index.calculate(a)
        cai_b = index.calculate(b)
    except Exception:
        return
    if not (math.isfinite(cai_a) and cai_a > 0.0 and math.isfinite(cai_b) and cai_b > 0.0):
        return

    predicted_log = (l_a * math.log(cai_a) + l_b * math.log(cai_b)) / (l_a + l_b)
    tol = 1e-9 * (l_a + l_b)
    trigger_if(abs(math.log(result) - predicted_log) > tol, "BP-SEQ-051")


@_guard("cai_synonymous_substitution_monotonicity")
def check_cai_synonymous_monotonicity(index, sequence, result):
    """BP-SEQ-053: CAI is exp(arithmetic mean of log-weights); replacing
    one contributing codon by a synonymous codon of weakly higher table
    weight, holding every other position fixed, cannot decrease the mean.
    This is the entire biological point of the index (ranking sequences by
    codon preference), so a substitution that lowers the score under this
    condition means the index no longer orders sequences consistently with
    its own definition.
    """
    if result is None or not math.isfinite(result) or result <= 0.0:
        return
    text = str(sequence).upper()
    if not text or len(text) % 3 or set(text) - set("ACGT"):
        return
    codons = [text[i:i + 3] for i in range(0, len(text), 3)]

    table = index._table
    for i, codon in enumerate(codons):
        if codon in ("ATG", "TGG"):
            continue
        try:
            w = index[codon]
        except KeyError:
            continue
        aa = table.forward_table.get(codon)
        if aa is None:
            continue
        better = [
            c for c, a in table.forward_table.items()
            if a == aa and c in index and index[c] >= w and c != codon
        ]
        if not better:
            continue
        substitute = better[0]
        candidate = codons[:i] + [substitute] + codons[i + 1:]
        candidate_seq = "".join(candidate)

        from Bio.SeqUtils import CodonAdaptationIndex  # noqa: F401

        try:
            substituted_result = index.calculate(candidate_seq)
        except Exception:
            continue
        if not math.isfinite(substituted_result):
            continue
        tol = 1e-9 * max(1.0, abs(math.log(max(result, 1e-300))))
        trigger_if(substituted_result < result - tol, "BP-SEQ-053")
        return


@_guard("cai_hardcoded_codon_weight")
def check_cai_hardcoded_codon_weight(index, sequence, result):
    """BP-SEQ-054: calculate() hardcodes ATG/TGG as always weight-1.0 and
    excludes them from the table lookup and cai_length. Valid only when, for
    this instance's table, ATG and TGG are each the sole codon for their
    amino acid (the only case __init__'s own construction guarantees weight
    1.0 for them). Several genetic codes Biopython ships reassign codons so
    ATG/TGG are no longer singleton synonyms (e.g. TGA reassigned to Trp,
    making it synonymous with TGG in several mitochondrial codes). Reads
    sequence and the already-built index/table; does not recompute CAI.
    """
    if not _cai_unmodified(index):
        return
    text = str(sequence).upper()
    if not text or len(text) % 3 or set(text) - set("ACGT"):
        return
    codons = [text[i : i + 3] for i in range(0, len(text), 3)]
    for codon in codons:
        if codon not in ("ATG", "TGG"):
            continue
        weight = index.get(codon)
        if weight is None:
            trigger("BP-SEQ-054")
            continue
        trigger_if(abs(weight - 1.0) > 1e-9, "BP-SEQ-054")


def record_cai_construction(index):
    """Snapshot a CodonAdaptationIndex right after ``__init__`` finished.

    BP-SEQ-055/056/057 state invariants that ``__init__`` guarantees for the
    weights it builds. A user who then rewrites the dict (``clear()`` /
    ``update()`` with a hand-loaded published or partial table, weights
    computed without the pseudo-count, ...) has replaced those weights, so the
    guarantee no longer describes the instance.
    """
    try:
        index._scibench_init_snapshot = dict(index)
    except Exception:
        pass


def _cai_unmodified(index):
    """True when ``index`` still holds exactly what ``__init__`` built."""
    snapshot = getattr(index, "_scibench_init_snapshot", None)
    return snapshot is not None and dict(index) == snapshot


@_guard("cai_hardcoded_stop_codon_set")
def check_cai_hardcoded_stop_codon_set(index, sequence, result):
    """BP-SEQ-055: calculate() classifies TGA/TAA/TAG (a hardcoded literal
    set) as a skippable stop codon exactly when self[codon] raises
    KeyError. This should coincide with index._table.stop_codons, the
    table-derived stop set for this instance, and __init__ seeds every
    codon in that set into self, so the KeyError branch should never fire
    for a true stop codon at all under the current construction. Reads
    sequence, index, and index._table; does not recompute CAI. Currently
    unreachable for any of Biopython's shipped codon tables (audit note,
    2026-09-28) -- kept per SANITIZER.md 5.7.2, would become reachable for
    a caller-supplied custom CodonTable whose forward_table/stop_codons do
    not jointly cover all 64 codons.
    """
    if not _cai_unmodified(index):
        return
    text = str(sequence).upper()
    if not text or len(text) % 3 or set(text) - set("ACGT"):
        return
    codons = [text[i : i + 3] for i in range(0, len(text), 3)]
    table_stops = set(index._table.stop_codons)
    literal_stops = {"TGA", "TAA", "TAG"}
    for codon in codons:
        if codon in table_stops and codon not in index:
            trigger("BP-SEQ-055")
            continue
        if codon in literal_stops and codon not in index and codon not in table_stops:
            trigger("BP-SEQ-055")


@_guard("cai_sense_codon_classification_consistency")
def check_cai_sense_codon_raise(index, codon):
    """BP-SEQ-056: calculate() is about to raise TypeError because ``codon``
    is neither ATG/TGG, a key of ``index``, nor one of the three literal stop
    codons. If ``index._table.forward_table`` -- the genetic-code table this
    same instance was built from -- calls ``codon`` a genuine sense codon,
    the raise contradicts the instance's own declared genetic code: __init__
    seeds every forward_table codon into ``index``, so under a
    self-consistent instance this branch should be unreachable for any
    codon forward_table recognizes. Reads only index/index._table and the
    codon about to cause the raise; does not recompute CAI.
    """
    if not _cai_unmodified(index):
        return
    table = getattr(index, "_table", None)
    if table is None:
        return
    forward_table = getattr(table, "forward_table", None)
    if forward_table is None:
        return
    trigger_if(codon in forward_table and codon not in index, "BP-SEQ-056")


@_guard("cai_weight_range_precondition")
def check_cai_weight_domain(index, codon, weight):
    """BP-SEQ-057: __init__ builds every self[codon] as
    counts[codon] / max(counts[c] for c in synonymous group), with every
    counts[codon] >= 0.5 (the paper's pseudo-count floor) and the
    denominator itself one of those counts, so for any __init__-built
    instance every weight lies in (0, 1] by construction. calculate() relies
    on this to call log(weight) safely and to keep the reported index within
    its documented scale. Reads the weight already looked up by production
    code for this codon; does not recompute it.
    """
    if not _cai_unmodified(index):
        return
    tol = 1e-9
    trigger_if(not (0.0 < weight <= 1.0 + tol), "BP-SEQ-057")


# --- SeqUtils.GC_skew --------------------------------------------------

@_guard("gc_skew")
def check_gc_skew(seq, window, values):
    """BP-SEQ-033: Watson-Crick pairing -- every G on one strand is a C on the
    complement and vice versa, so (G - C)/(G + C) computed on the complement is
    the exact element-wise negation of the value on the original strand.
    """
    from Bio.Seq import Seq
    from Bio.SeqUtils import GC_skew

    text = str(seq).upper().replace("U", "T")
    if not text or set(text) - set("ACGT"):
        return
    complement = str(Seq(text).complement())
    other = GC_skew(complement, window)
    differs = len(values) != len(other) or any(
        not _isclose(a, -b) for a, b in zip(values, other)
    )
    trigger_if(differs, "BP-SEQ-033")


# --- ProtParam: protein molecular_weight --------------------------------

@_guard("protein_molecular_weight")
def check_protein_molecular_weight(sequence, monoisotopic, weight):
    """BP-SEQ-036: a protein's mass is the sum of its residue masses minus one
    condensation water per peptide bond. Mass is additive and order-independent,
    so the value is invariant under residue permutation.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    if len(sequence) < 2:
        return
    reversed_weight = ProteinAnalysis(
        sequence[::-1], monoisotopic=monoisotopic
    ).molecular_weight()
    tol = 1e-9 * max(1.0, abs(weight), abs(reversed_weight))
    trigger_if(not _within(weight, reversed_weight, tol), "BP-SEQ-036")


# --- SeqUtils.molecular_weight: RNA vs DNA ------------------------------

@_guard("rna_dna_mass_ordering")
def check_rna_dna_mass_ordering(original_seq, seq_type, double_stranded, circular,
                                monoisotopic, weight):
    """BP-SEQ-042: chemically, an RNA nucleotide carries a 2'-OH that the
    corresponding DNA nucleotide replaces with 2'-H, and U (RNA) is lighter
    than T (DNA) by one CH2. Summed over a strand the 2'-OH term dominates, so
    single-stranded RNA is heavier than the same-length DNA string.
    """
    if seq_type != "RNA" or double_stranded or circular:
        return
    if not original_seq or set(original_seq) - set("ACGU"):
        return
    from Bio.SeqUtils import molecular_weight

    dna_equiv = original_seq.replace("U", "T")
    dna_weight = molecular_weight(dna_equiv, "DNA", monoisotopic=monoisotopic)
    trigger_if(weight <= dna_weight, "BP-SEQ-042")


# ======================================================================
# Bio.PDB geometry
# ======================================================================

def _rigid_transform(points, seed=0):
    """Apply a fixed rotation + translation to a list of 3-tuples/arrays.

    The translation is scaled to the spread of the input points. A fixed
    ``(7, -3, 11)`` shift would, for coordinates far below unit magnitude,
    round every transformed point to the same float64 value (catastrophic
    cancellation *in this probe*, not in the API under test); scaling keeps
    the rigid motion faithful across coordinate scales.
    """
    import numpy as np

    theta = 0.9
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    pts = [np.asarray(p, dtype=float) for p in points]
    scale = max((float(np.linalg.norm(p)) for p in pts), default=0.0)
    if not math.isfinite(scale) or scale == 0.0:
        scale = 1.0
    shift = np.array([0.7, -0.3, 1.1]) * scale
    return [rot @ p + shift for p in pts]


def _magnitude_bounded(points, cap=1e6):
    """True when every point's coordinate magnitude is below ``cap``.

    ``calc_dihedral`` composes two cross products and a further cross product
    of one of those with a unit vector before taking a dot product -- degree
    enough in the input coordinates that float64 silently overflows starting
    around coordinate magnitude ~2.7e42 (empirically verified by binary
    search; ``calc_angle`` does not share this failure mode until ~1e200+,
    so only the dihedral checker needs this precondition). Past that point
    both the original call and the checker's rigid-motion probe independently
    return garbage, and the resulting "disagreement" is arithmetic breakdown,
    not a rigid-invariance or chirality violation. ``cap`` sits 36 orders of
    magnitude below the overflow onset -- far above any physically
    meaningful coordinate (PDB structures are in angstroms).
    """
    import numpy as np

    return all(float(np.max(np.abs(p))) < cap for p in points)


def _points_well_separated(points, rel=1e-6):
    """True when every pair of the given 3-D points is separated by at least
    ``rel`` times the coordinate scale.

    The rigid-invariance probe rotates and translates the points; when two of
    them are closer than ~1e-6 of the coordinate magnitude, that difference is
    below float64 resolution after the transform and the recomputed angle is
    dominated by rounding -- a limitation of the probe (and of the angle's own
    conditioning), not a violation of Euclidean invariance.
    """
    import numpy as np

    pts = [np.asarray(p, dtype=float) for p in points]
    if any(not np.all(np.isfinite(p)) for p in pts):
        return False
    scale = max((float(np.linalg.norm(p)) for p in pts), default=0.0)
    floor = rel * max(scale, 1.0)
    for a in range(len(pts)):
        for b in range(a):
            if float(np.linalg.norm(pts[a] - pts[b])) < floor:
                return False
    return True


@_guard("pdb_calc_angle")
def check_calc_angle(v1, v2, v3, angle):
    """BP-PDB-002 endpoint symmetry; BP-PDB-003 rigid invariance.

    A bond angle is a Euclidean invariant: it does not change when the three
    atoms are rigidly rotated and translated together (003), and it does not
    depend on the traversal direction of the two rays from the vertex (002).
    """
    from Bio.PDB.vectors import calc_angle, Vector

    trigger_if(not _isclose(angle, calc_angle(v3, v2, v1)), "BP-PDB-002")

    arrays = [v1.get_array(), v2.get_array(), v3.get_array()]
    # The rigid probe forms quadratic norms.  At extreme-but-finite coordinate
    # magnitudes those norms overflow before the geometric law is evaluated;
    # that is outside this float64 error model, not a loss of rigid invariance.
    if not _magnitude_bounded(arrays) or not _points_well_separated(arrays):
        return
    p1, p2, p3 = _rigid_transform(arrays)
    moved = calc_angle(Vector(p1), Vector(p2), Vector(p3))
    trigger_if(not _isclose(angle, moved, tol=1e-7), "BP-PDB-003")


@_guard("pdb_calc_dihedral")
def check_calc_dihedral(v1, v2, v3, v4, angle):
    """BP-PDB-006 rigid invariance; BP-PDB-007 chirality (mirror negates).

    A signed dihedral is invariant under a proper rigid motion (006) and
    changes sign under an improper one (a reflection, 007) -- this sign is the
    stereochemical chirality of the four-atom arrangement.
    """
    import math as _m

    from Bio.PDB.vectors import calc_dihedral, Vector

    import numpy as _np

    arrays = [v1.get_array(), v2.get_array(), v3.get_array(), v4.get_array()]
    # The dihedral is well defined only when neither the (1,2,3) nor the
    # (2,3,4) triple is collinear (each defines a plane whose normal enters the
    # angle). And near |angle| == pi the value sits on the atan2 branch cut, so
    # a proper rigid motion can legitimately return -pi instead of +pi for the
    # identical geometry -- BP-PDB-007 already excludes this; BP-PDB-006 must
    # too. Both are limitations of the probe / the angle's parameterisation,
    # not violations of Euclidean invariance or chirality.
    def _triple_ok(a, b, c):
        # The dihedral's conditioning is 1/sin(theta) where theta is the angle
        # between the two triples' normals -- and ||n|| / scale^2 IS sin(theta)
        # for that triple. A near-collinear triple (sin(theta) small) amplifies
        # the ~1 ULP coordinate error _rigid_transform introduces by 1/sin(theta),
        # so the admitted sin(theta) floor and the comparison tolerance below are
        # coupled, not independent: 1e-9 admitted a condition number of 1e9,
        # amplifying 1 ULP (~1e-16) to ~1e-7 -- exactly the tol used for the
        # comparison, so a well-computed dihedral could still fire (audit 5).
        # Requiring sin(theta) >~ 1e-3 keeps the amplified error <~1e-13, ten
        # decades under the 1e-7 comparison tolerance.
        n = _np.cross(b - a, c - b)
        scale = max(_np.linalg.norm(b - a), _np.linalg.norm(c - b), 1.0)
        return float(_np.linalg.norm(n)) > 1e-3 * scale * scale

    a0, a1, a2, a3 = (_np.asarray(x, dtype=float) for x in arrays)
    proper_dihedral = (
        _magnitude_bounded(arrays)
        and _points_well_separated(arrays)
        and _triple_ok(a0, a1, a2)
        and _triple_ok(a1, a2, a3)
        and not _isclose(abs(angle), _m.pi, tol=1e-7)
    )

    # The rigid-motion probe rounds every coordinate by ~1 ULP of its
    # magnitude (eps * max|x|); relative to the shortest bond and amplified by
    # 1/sin(theta) of the flattest triple, that is the numerical error budget
    # of the dihedral. A fixed 1e-7 ignores the coordinate-magnitude term (a
    # 6e5 A offset with sin(theta) ~ 2e-3 gives ~3e-8 of pure round-off, and
    # larger still beyond). Use the larger of 1e-7 and 1e3x that budget; when
    # the budget is itself above 1e-3 rad the comparison resolves nothing.
    tol006 = 1e-7
    resolvable = False
    if proper_dihedral:
        bonds = [_np.linalg.norm(a1 - a0), _np.linalg.norm(a2 - a1),
                 _np.linalg.norm(a3 - a2)]
        sins = [
            _np.linalg.norm(_np.cross(a1 - a0, a2 - a1)) / (bonds[0] * bonds[1]),
            _np.linalg.norm(_np.cross(a2 - a1, a3 - a2)) / (bonds[1] * bonds[2]),
        ]
        budget = (2.220446049250313e-16 * max(float(_np.max(_np.abs(a)))
                                              for a in (a0, a1, a2, a3))
                  / (min(bonds) * min(sins)))
        tol006 = max(1e-7, 1e3 * budget)
        resolvable = tol006 < 1e-3

    if resolvable:
        p = _rigid_transform(arrays)
        moved = calc_dihedral(
            Vector(p[0]), Vector(p[1]), Vector(p[2]), Vector(p[3])
        )
        trigger_if(not _isclose(angle, moved, tol=tol006), "BP-PDB-006")

    def mirror(v):
        a = v.get_array().copy()
        a[2] = -a[2]
        return Vector(a)

    mirrored = calc_dihedral(mirror(v1), mirror(v2), mirror(v3), mirror(v4))
    trigger_if(proper_dihedral
               and not _isclose(angle, -mirrored, tol=1e-7),
               "BP-PDB-007")


@_guard("pdb_rotmat")
def check_rotmat(p, q, matrix):
    """BP-PDB-011 orthogonality (det +1); BP-PDB-012 maps p onto q.

    rotmat(p, q) must be a proper rotation (an element of SO(3): orthogonal
    with determinant +1, so it preserves lengths and handedness), and it must
    actually carry the direction of p onto the direction of q.

    Precondition: p and q are non-collinear (the antiparallel case p = -k q is
    a documented rotation singularity and is excluded).
    """
    import numpy as np

    pa, qa0 = p.get_array(), q.get_array()
    # ``np.linalg.norm`` and the dot product below are quadratic in coordinate
    # magnitude.  Keep the checker inside the same bounded float64 geometry
    # domain used by the dihedral/angle probes.
    if not _magnitude_bounded([pa, qa0]):
        return
    np_ = np.linalg.norm(pa) * np.linalg.norm(qa0)
    if np_ < 1e-12 or abs(abs(float(np.dot(pa, qa0)) / np_) - 1.0) < 1e-9:
        return

    trigger_if(not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9),
               "BP-PDB-011")
    trigger_if(not _isclose(float(np.linalg.det(matrix)), 1.0, tol=1e-9),
               "BP-PDB-011")

    pr = p.left_multiply(matrix).get_array()
    trigger_if(not _isclose(float(np.linalg.norm(pr)),
                            float(np.linalg.norm(p.get_array())), tol=1e-7),
               "BP-PDB-012")
    qa = q.get_array()
    if np.linalg.norm(pr) > 1e-9 and np.linalg.norm(qa) > 1e-9:
        cos = float(np.dot(pr, qa) / (np.linalg.norm(pr) * np.linalg.norm(qa)))
        trigger_if(cos < 1.0 - 1e-7, "BP-PDB-012")


    # KNOWN LIMITATION (audit 4): QCP's Newton tolerance evalprec (1e-11) and
    # eigenvector tolerance evecprec (1e-6) are absolute constants applied to
    # quantities that scale as coordinate^2, so QCP's accuracy is scale
    # dependent -- at coordinate scale ~1e-6 it can report RMSD ~ 0 for two
    # different point sets. This checker's probes (rigid motion, argument
    # swap) are all scale preserving and cannot see it. A scale-covariance
    # probe was tried and rejected: refitting a *correct* small-coordinate
    # case at an even smaller scale hits the same QCP weakness and the probe
    # false-fires. The scale dependence remains an open upstream limitation,
    # unrelated to the argument-slot gap fixed above.


def _float_eps(*arrays):
    """Machine epsilon of the coarsest floating dtype among ``arrays``.

    Float64 (and integer / Python-list input, which numpy promotes to float64)
    gives 2.2e-16.  float32 coordinates, e.g. PDB-style single precision, make
    the library compute in float32, so its round-off is ~1.2e-7 and a float64
    tolerance would flag legitimate precision loss rather than a law violation.
    """
    import numpy as np

    eps = float(np.finfo(np.float64).eps)
    for a in arrays:
        dt = getattr(a, "dtype", None)
        if dt is not None and np.issubdtype(dt, np.floating):
            eps = max(eps, float(np.finfo(dt).eps))
    return eps


@_guard("pdb_qcp_rotation_properness")
def check_qcp_rotation_properness(rot, coords=None):
    """BP-PDB-020: QCP is defined as an eigen-decomposition method that
    returns a proper rigid-body rotation, never a reflection or a shear.
    The returned matrix must be orthogonal (rot @ rot.T == I) and
    orientation-preserving (det(rot) == +1); any departure means the
    transform is not a physically valid rigid motion.
    """
    import numpy as np

    r = np.asarray(rot, dtype=float)
    if r.shape != (3, 3):
        return
    # QCP forms products of the coordinates up to the sixth power; float64
    # overflows there at coordinate magnitude ~1e26 (rot silently becomes the
    # zero matrix). That is far outside any physical coordinate (angstroms),
    # so it is arithmetic breakdown, not a broken rigid-motion contract. Swept:
    # rot stays proper up to at least 1e24.
    if coords is not None and not _magnitude_bounded([np.asarray(coords, dtype=float)]):
        return
    orth_err = float(np.max(np.abs(r @ r.T - np.eye(3))))
    det_err = float(abs(np.linalg.det(r) - 1.0))
    # 1e-8 for float64; float32 input yields a rotation orthogonal only to a
    # few eps(float32) ~ 1e-7, so scale by the input dtype's epsilon.
    tol = max(1e-8, 64.0 * _float_eps(rot, coords))
    trigger_if(orth_err > tol or det_err > tol, "BP-PDB-020")


@_guard("pdb_qcp_rmsd_optimality")
def check_qcp_rmsd_optimality(init_rms, rms, coords_ref=None, coords=None):
    """BP-PDB-021: QCP minimizes RMSD over all rigid motions, and the
    identity motion (no rotation/translation) is always in that search
    space. Therefore the fitted RMSD can never exceed the RMSD of the
    untransformed point sets; if it does, the reported fit is not
    actually optimal.
    """
    scale = max(1.0, float(init_rms), float(rms))
    tol = 1e-6 * scale
    if coords_ref is not None and coords is not None:
        # QCP's RMSD is sqrt(2|E0 - lambda|/N), a difference of two nearly equal
        # numbers, so it is only accurate to ~sqrt(eps) times the radius of the
        # (centred) point sets, not eps: float64 identical sets at radius ~50
        # report rms ~1e-6, float32 ones ~1e-3.
        import numpy as np

        eps = _float_eps(coords_ref, coords)
        radius = max(
            float(np.sqrt(np.mean(np.sum(np.asarray(c, dtype=float) ** 2, axis=1))))
            for c in (coords_ref, coords)
        )
        # Near a multiple top root of the quaternion eigenproblem (mirror-image or
        # near-collinear sets) the eigenvalue is only accurate to ~eps**(1/3), so the
        # radius-relative floor is 1e-3 (as in BP-PDB-022).
        tol = max(tol, max(1e-3, 16.0 * math.sqrt(eps)) * max(scale, radius))
    trigger_if(float(rms) > float(init_rms) + tol, "BP-PDB-021")


@_guard("pdb_qcp_rmsd_cross_consistency")
def check_qcp_rmsd_cross_consistency(reference_coords, coords, rot, tran, rms):
    """BP-PDB-022: QCP's headline RMSD is obtained algebraically from a
    quaternion eigenvalue, never by actually moving points. It must agree
    with the RMSD computed directly from Euclidean distances between the
    reference coordinates and the transformed moving coordinates -- two
    independent representations of the same fit quality. The transform is
    applied locally here (never via the public get_transformed(), which
    caches into self.transformed_coords -- SANITIZER.md 5.5 forbids a
    checker from altering persistent state).
    """
    import numpy as np

    ref = np.asarray(reference_coords, dtype=float)
    mov = np.asarray(coords, dtype=float)
    r = np.asarray(rot, dtype=float)
    t = np.asarray(tran, dtype=float)
    if ref.shape != mov.shape or ref.shape[0] < 1 or r.shape != (3, 3):
        return
    moved = np.dot(mov, r) + t
    direct_rms = float(np.sqrt(np.mean(np.sum((ref - moved) ** 2, axis=1))))
    scale = max(1.0, float(rms), float(np.abs(ref).max()), float(np.abs(moved).max()))
    tol = 1e-6 * scale
    # QCP's rms is sqrt(2|E0-lambda|/N), a difference of nearly equal numbers, and for
    # near-collinear (rod-like) sets the top quaternion eigenvalues nearly coincide, so a
    # correct float64 run is only accurate to ~eps**0.25 of the radius (observed <= 8e-4);
    # float32 loses ~sqrt(eps). A genuine failure (identity fallback, non-converged root)
    # errs by O(1e-2..1) of the radius, so the tolerance is radius-relative and dtype-aware.
    eps = _float_eps(reference_coords, coords)
    radius = float(np.sqrt(np.mean(np.sum((ref - ref.mean(axis=0)) ** 2, axis=1))))
    floor = 64.0 * eps * max(float(np.abs(ref).max()), float(np.abs(moved).max()))
    tol = max(max(1e-3, 32.0 * math.sqrt(eps)) * radius, floor)
    trigger_if(abs(float(rms) - direct_rms) > tol, "BP-PDB-022")


@_guard("pdb_qcp_translation_invariance")
def check_qcp_translation_invariance(reference_coords, coords, rot, rms):
    """BP-PDB-023: a rigid superposition is a property of the relative
    geometry of the two point sets, not of where in space they sit. QCP's
    run() explicitly mean-centers both point sets before fitting, so
    translating both by the same constant vector must not change the
    fitted rotation, and must change the RMSD by no more than the
    numerical-error budget of the centering subtraction.
    """
    import numpy as np

    from Bio.PDB.qcprot import QCPSuperimposer

    ref = np.asarray(reference_coords, dtype=float)
    mov = np.asarray(coords, dtype=float)
    if ref.shape != mov.shape or ref.shape[0] < 3:
        return
    r = np.asarray(rot, dtype=float)
    if r.shape != (3, 3):
        return

    scale = float(max(np.abs(ref).max(), np.abs(mov).max(), 1.0))
    # Keep the translation within the data's own coordinate scale so the
    # probe stays numerically neutral (SANITIZER.md 5.8-X): an unboundedly
    # large offset would introduce centroid-subtraction cancellation
    # unrelated to the law under test.
    t = np.full(3, scale, dtype=float)

    shifted = QCPSuperimposer()
    shifted.set(ref + t, mov + t)
    shifted.run()

    rot_tol = 1e-6 * (1.0 + scale)
    rms_tol = 1e-6 * (scale + float(np.linalg.norm(t)))
    eps = _float_eps(reference_coords, coords)
    if eps > 1e-10:
        # Single precision: the original fit ran in float32 (the probe runs in
        # float64 on the same rounded values), so the two fits legitimately
        # differ by float32 round-off; the RMSD only to ~sqrt(eps).
        rot_tol = max(rot_tol, 256.0 * eps * (1.0 + scale))
        rms_tol = max(rms_tol, 16.0 * math.sqrt(eps) * (1.0 + scale))
    # QCP's rms = sqrt(2(E0-lambda)/N) is a difference of nearly equal numbers and, for
    # near-collinear sets (multiple top root), is accurate only to a radius-relative 1e-3;
    # the rotation check stays tight.
    radius = max(
        float(np.sqrt(np.mean(np.sum((a - a.mean(axis=0)) ** 2, axis=1))))
        for a in (ref, mov)
    )
    rms_tol = max(rms_tol, max(1e-3, 16.0 * math.sqrt(eps)) * radius)
    rot_err = float(np.max(np.abs(np.asarray(shifted.rot) - r)))
    trigger_if(rot_err > rot_tol, "BP-PDB-023")
    trigger_if(abs(float(shifted.rms) - float(rms)) > rms_tol, "BP-PDB-023")


# ======================================================================
# Bio.PDB.Superimposer / SVDSuperimposer
# ======================================================================

@_guard("superimposer")
def check_superimposer(fixed_coord, moving_coord, rot, tran, rms):
    """BP-SUP-001 rotation in SO(3); BP-SUP-002 RMSD rigid invariance;
    BP-SUP-003 RMSD argument symmetry; BP-SUP-004 exact solution;
    BP-SUP-007 permutation invariance.

    A least-squares superposition returns a *proper* rotation (orthogonal,
    determinant +1 -- a reflection would match mirror images, 001). The optimal
    RMSD is an intrinsic distance between the two point sets: unchanged by a
    rigid motion of the moving set before fitting (002), symmetric in its two
    arguments (003), zero when one set is a rigid image of the other (004), and
    independent of the order the point pairs are listed in (007).
    """
    import numpy as np

    from Bio.SVDSuperimposer import SVDSuperimposer

    ref = np.asarray(fixed_coord, dtype=float)
    mov = np.asarray(moving_coord, dtype=float)
    if ref.shape != mov.shape or ref.shape[0] < 3 or ref.shape[1] != 3:
        return
    r = np.asarray(rot, dtype=float)

    trigger_if(not np.allclose(r.T @ r, np.eye(3), atol=1e-6), "BP-SUP-001")
    trigger_if(abs(float(np.linalg.det(r)) - 1.0) > 1e-6, "BP-SUP-001")

    def _fit(a, b):
        s = SVDSuperimposer()
        s.set(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        s.run()
        return float(s.get_rms())

    # All four RMSD invariants below compare two least-squares RMSD values, an
    # O(coordinate-magnitude) quantity. SVD superposition round-off is ~1 ULP
    # relative (~2e-16) at every scale, so the tolerance must scale with the
    # magnitude (methodology 8.3), NOT be a fixed 1e-6 / 1e-4 absolute -- those
    # overflow above coordinate scale ~1e10 while the law still holds exactly.
    coord_scale = float(max(np.abs(ref).max(), np.abs(mov).max(), 1.0))

    def _rms_tol(*values):
        return 1e-9 * max(1.0, coord_scale, *(abs(v) for v in values))

    # The optimal rotation is determined by the second singular value of each
    # centred set: for a (nearly) linear molecule rotation about its axis is
    # (nearly) undetermined, the SVD rotation is conditioned by sv[0]/sv[1],
    # and the round-off in the probe/refit is amplified into an RMSD
    # difference that is not a violation of the invariants below. Swept
    # (near-linear chains, mirror/perturbed/rotated copies, 4000 trials): the
    # first alarms appear only for sv[1]/sv[0] < ~1e-6; 1e-4 leaves margin.
    def _rot_conditioned(x):
        sv = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
        return sv[0] > 0.0 and sv[1] > 1e-4 * sv[0]

    if not (_rot_conditioned(ref) and _rot_conditioned(mov)):
        return

    moved = np.asarray(_rigid_transform(list(mov)))
    fit_moved = _fit(ref, moved)
    trigger_if(abs(fit_moved - rms) > _rms_tol(rms, fit_moved), "BP-SUP-002")

    fit_swapped = _fit(mov, ref)
    trigger_if(abs(fit_swapped - rms) > _rms_tol(rms, fit_swapped), "BP-SUP-003")

    # BP-SUP-004 feeds _rigid_transform(ref) back as an *exact* rigid image and
    # asserts a zero fit RMSD. Unlike SUP-002/003/007 (which compare two equally
    # perturbed RMSDs), this uses the transformed set as absolute ground truth,
    # so the ~1 ULP that _rigid_transform introduces into the pairwise distances
    # matters: on a nearly rank-deficient ref that ULP is amplified by the
    # condition number into a macroscopic RMSD (audit 4, BP-SUP-004, cond ~2e16
    # -> RMSD 0.34). Only run SUP-004 when centred ref is well-conditioned, so
    # the probe is a faithful isometry.
    centred = ref - ref.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[0] > 0.0 and sv[-1] > 1e-8 * sv[0]:
        exact = np.asarray(_rigid_transform(list(ref)))
        trigger_if(_fit(ref, exact) > _rms_tol(), "BP-SUP-004")

    perm = np.random.RandomState(0).permutation(ref.shape[0])
    fit_perm = _fit(ref[perm], mov[perm])
    trigger_if(abs(fit_perm - rms) > _rms_tol(rms, fit_perm), "BP-SUP-007")


# ======================================================================
# Bio.Align.PairwiseAligner
# ======================================================================

def _seq_text(seq):
    """Return a plain string for a str/Seq/bytes sequence, else None.

    A checker that cannot express the sequence as text (e.g. a list of
    arbitrary tokens, or an already-encoded integer array) simply does not
    run -- it never guesses.
    """
    from Bio.Seq import MutableSeq, Seq

    if isinstance(seq, str):
        return seq
    if isinstance(seq, (Seq, MutableSeq)):
        return str(seq)
    if isinstance(seq, (bytes, bytearray)):
        try:
            return seq.decode("ascii")
        except Exception:
            return None
    return None


def _clone_aligner(aligner):
    """Return an independent PairwiseAligner with the same scoring scheme."""
    from Bio.Align import PairwiseAligner

    clone = PairwiseAligner()
    clone.__setstate__(aligner.__getstate__())
    return clone


def _scores_above_epsilon(aligner, sequence_length=1):
    """True when every nonzero term of the scoring scheme is safely larger than
    the aligner's ``epsilon`` (its documented traceback roundoff tolerance,
    default 1e-6).

    ``score()`` uses a pure score-only DP; ``align()`` uses a traceback whose
    ``SELECT_TRACE`` macros compare cells with ``+/- epsilon`` slack. When the
    per-column scores are themselves below ``epsilon``, the traceback
    legitimately treats distinct alignments as tied and the two paths can
    report different optima -- that is the documented behaviour of the
    ``epsilon`` parameter, not a defect. A checker comparing the two paths, or
    re-running ``score()``, must exclude such schemes.
    """
    import numpy as np

    try:
        eps = float(aligner.epsilon)
    except Exception:
        eps = 1e-6
    terms = []
    for name in (
        "match_score", "mismatch_score",
        "open_internal_insertion_score", "extend_internal_insertion_score",
        "open_internal_deletion_score", "extend_internal_deletion_score",
        "open_left_insertion_score", "extend_left_insertion_score",
        "open_right_insertion_score", "extend_right_insertion_score",
        "open_left_deletion_score", "extend_left_deletion_score",
        "open_right_deletion_score", "extend_right_deletion_score",
    ):
        try:
            terms.append(abs(float(getattr(aligner, name))))
        except Exception:
            pass
    matrix = aligner.substitution_matrix
    if matrix is not None:
        arr = np.abs(np.asarray(matrix, dtype=float))
        terms.extend(arr[np.isfinite(arr)].tolist())
    nonzero = [t for t in terms if t > 0.0]
    if not nonzero:
        return False
    # Every term must clear the traceback roundoff tolerance -- a scheme with
    # terms at or below epsilon makes score()/align() legitimately diverge
    # (documented behaviour of `epsilon`), independent of the comparison
    # tolerance used elsewhere. The dynamic-range/magnitude question (how
    # large the DP intermediates get, and what comparison tolerance survives
    # that) is handled separately by `_score_compare_tol` at each call site,
    # not folded into this precondition (audit 6 -- a fixed `max/min < 1e10`
    # here still let a fixed downstream `tol=1e-9` fail: intermediates reach
    # ~L*max_term regardless of the min/max ratio).
    if min(nonzero) <= 1e3 * eps:
        return False
    # A DP cell may accumulate O(L) score terms.  Finite individual scores do
    # not imply a representable optimum (for example 2 * 1e308).  Leave a
    # conservative factor-16 margin for affine-gap transitions and competing
    # paths; overflow is outside the checkers' float64 comparison model.
    max_term = max(nonzero)
    return max_term <= np.finfo(float).max / (
        16.0 * max(int(sequence_length), 1)
    )


def _score_compare_tol(aligner, a, b):
    """Absolute+relative tolerance for comparing two alignment scores of
    sequences ``a``/``b`` under ``aligner``'s scheme.

    The DP recurrence accumulates a sum of up to ``O(L)`` scoring terms (L the
    longer sequence length), so an intermediate cell can reach magnitude
    ``L * max(|term|)`` even when the final score is much smaller (large
    positive and negative contributions cancelling). The float64 rounding on
    that intermediate is one ULP of ITS magnitude, not of the final answer's,
    so the comparison tolerance must be derived from the intermediate scale,
    not from the answer or from a fixed constant (audit 6, BP-ALN-006: with
    match=1e8, extend=-3.3e-2, L=11, intermediates reach ~1.1e9, one ULP there
    is ~2.4e-7 -- 200x the fixed `1e-9` this replaces, at a term ratio of 3e9,
    well under the `1e10` dynamic-range gate that previously guarded this).
    """
    import numpy as np

    terms = [0.0]
    for name in (
        "match_score", "mismatch_score",
        "open_internal_insertion_score", "extend_internal_insertion_score",
        "open_internal_deletion_score", "extend_internal_deletion_score",
        "open_left_insertion_score", "extend_left_insertion_score",
        "open_right_insertion_score", "extend_right_insertion_score",
        "open_left_deletion_score", "extend_left_deletion_score",
        "open_right_deletion_score", "extend_right_deletion_score",
    ):
        try:
            v = float(getattr(aligner, name))
            if math.isfinite(v):
                terms.append(abs(v))
        except Exception:
            pass
    matrix = aligner.substitution_matrix
    if matrix is not None:
        arr = np.abs(np.asarray(matrix, dtype=float))
        finite = arr[np.isfinite(arr)]
        if finite.size:
            terms.append(float(finite.max()))
    max_term = max(terms)
    L = max(len(a), len(b), 1)
    # 1e3x safety margin over one float64 ULP of the largest plausible DP
    # intermediate; floored at 1e-9 to match this module's other exact-score
    # comparisons (methodology 8.3).
    return max(1e-9, 1e3 * 2.220446049250313e-16 * L * max_term)


def _symmetric_scoring(aligner):
    """True when the aligner's scoring scheme is symmetric in target/query."""
    import numpy as np

    if aligner.open_internal_insertion_score != aligner.open_internal_deletion_score:
        return False
    if (aligner.extend_internal_insertion_score
            != aligner.extend_internal_deletion_score):
        return False
    if aligner.open_left_insertion_score != aligner.open_left_deletion_score:
        return False
    if aligner.open_right_insertion_score != aligner.open_right_deletion_score:
        return False
    # Swapping target and query swaps insertion and deletion, so the *extend*
    # end-gap scores must match as well (a cheaper target-only end-gap
    # extension is a legitimately asymmetric scheme).
    if aligner.extend_left_insertion_score != aligner.extend_left_deletion_score:
        return False
    if aligner.extend_right_insertion_score != aligner.extend_right_deletion_score:
        return False
    matrix = aligner.substitution_matrix
    if matrix is not None:
        arr = np.asarray(matrix)
        # Exact symmetry: an approximately symmetric matrix (np.allclose) still
        # scores (a, b) and (b, a) differently by the asymmetry itself.
        return arr.ndim == 2 and np.array_equal(arr, arr.T)
    return True


@_guard("aligner_score")
def check_aligner_score(aligner, seqA, seqB, score):
    """BP-ALN-001 score symmetry; BP-ALN-006 double-reversal invariance;
    BP-ALN-009 substitution-matrix transpose invariance.

    Under a symmetric scoring scheme the optimal global alignment score is a
    symmetric function of the two sequences (001); reversing *both* sequences
    transposes the dynamic-programming matrix and must not change the global
    score (006); and for a symmetric substitution matrix, replacing it with its
    transpose is the identity (009).
    """
    try:
        if str(aligner.mode) != "global":
            return
    except Exception:
        return
    a, b = _seq_text(seqA), _seq_text(seqB)
    if a is None or b is None or not a or not b:
        return
    if not _symmetric_scoring(aligner) or not _scores_above_epsilon(
        aligner, max(len(a), len(b))
    ):
        return

    tol = _score_compare_tol(aligner, a, b)

    clone = _clone_aligner(aligner)
    trigger_if(not _within(float(clone.score(b, a)), float(score), tol),
               "BP-ALN-001")

    lr_symmetric = (
        aligner.open_left_insertion_score == aligner.open_right_insertion_score
        and aligner.extend_left_insertion_score
        == aligner.extend_right_insertion_score
        and aligner.open_left_deletion_score == aligner.open_right_deletion_score
        and aligner.extend_left_deletion_score
        == aligner.extend_right_deletion_score
    )
    if lr_symmetric:
        clone2 = _clone_aligner(aligner)
        trigger_if(
            not _within(
                float(clone2.score(a[::-1], b[::-1])), float(score), tol
            ),
            "BP-ALN-006",
        )

    matrix = aligner.substitution_matrix
    if matrix is not None:
        import numpy as np

        transposed = matrix.transpose() if hasattr(matrix, "transpose") else None
        if transposed is not None and np.array_equal(np.asarray(matrix),
                                                     np.asarray(transposed)):
            # matrix.transpose() returns a non-C-contiguous view; the
            # PairwiseAligner.substitution_matrix C-extension setter rejects
            # non-contiguous arrays. Rebuilding with the same class/alphabet
            # over a C-contiguous copy of the same values is a neutral,
            # value-preserving step (SANITIZER.md 5.8 "X"): it does not
            # change which entries are compared, only their memory layout.
            transposed = type(matrix)(
                alphabet=matrix.alphabet,
                data=np.ascontiguousarray(np.asarray(transposed)),
            )
            clone3 = _clone_aligner(aligner)
            clone3.substitution_matrix = transposed
            trigger_if(
                not _within(float(clone3.score(a, b)), float(score), tol),
                "BP-ALN-009",
            )


@_guard("aligner_align_vs_score")
def check_aligner_align_vs_score(aligner, seqA, seqB, align_score):
    """BP-ALN-003: the score returned by ``score()`` and the score of the best
    alignment returned by ``align()`` are computed by two different code paths
    (score-only DP vs. traceback) and must agree, in any alignment mode.
    """
    a, b = _seq_text(seqA), _seq_text(seqB)
    if a is None or b is None or not a or not b:
        return
    if not _scores_above_epsilon(aligner, max(len(a), len(b))):
        # scores below the aligner's traceback epsilon: score() and align()
        # legitimately diverge (documented behaviour of `epsilon`), not a bug.
        return
    clone = _clone_aligner(aligner)
    tol = _score_compare_tol(aligner, a, b)
    trigger_if(not _within(float(clone.score(a, b)), float(align_score), tol),
               "BP-ALN-003")


# ======================================================================
# Bio.Phylo.TreeConstruction
# ======================================================================

def _patristic(tree):
    """Map every unordered terminal-name pair to its path length in ``tree``.

    Reads only ``branch_length`` values production code already assigned; it
    does not recompute any evolutionary distance.
    """
    terminals = tree.get_terminals()
    dist = {}
    for i in range(len(terminals)):
        for j in range(i + 1, len(terminals)):
            a, b = terminals[i], terminals[j]
            key = tuple(sorted((a.name, b.name)))
            dist[key] = tree.distance(a, b)
    return dist


def _all_branch_lengths(tree):
    return [
        c.branch_length
        for c in tree.find_clades()
        if c.branch_length is not None
    ]


def _matrix_scale(distance_matrix):
    """Largest finite pairwise distance in the matrix, floored at 1.0.

    Patristic-distance comparisons must scale their tolerance with this
    (methodology 8.3): a fixed 1e-6 absolute tolerance is ~1e-10 relative on
    a matrix with entries ~1e4, which floating-point tree reconstruction can
    reach, so an additive large-magnitude matrix would trip the checker.
    """
    names = list(distance_matrix.names)
    vals = [
        abs(float(distance_matrix[names[i], names[j]]))
        for i in range(len(names))
        for j in range(i)
    ]
    vals = [v for v in vals if math.isfinite(v)]
    return max([1.0, *vals])


def _matrix_tol(distance_matrix):
    """Absolute tolerance for patristic / branch-length comparisons.

    ``1e-9 * scale`` for float64 entries.  If the entries are float32 (numpy
    scalars) NJ/UPGMA do their arithmetic in float32, whose round-off
    (~1.2e-7 relative) is far above 1e-9, so the tolerance becomes
    ``64 * eps(dtype) * scale``.
    """
    import numpy as np

    eps = float(np.finfo(np.float64).eps)
    names = list(distance_matrix.names)
    for i in range(len(names)):
        for j in range(i):
            v = distance_matrix[names[i], names[j]]
            if isinstance(v, np.floating):
                eps = max(eps, float(np.finfo(type(v)).eps))
    return max(1e-9, 64.0 * eps) * _matrix_scale(distance_matrix)


def _matrix_arithmetic_safe(distance_matrix):
    """Whether O(n)-term float64 distance arithmetic cannot overflow.

    NJ and UPGMA form row sums and affine combinations of pairwise distances.
    A matrix may contain finite entries while those required intermediates are
    not representable.  Such a call is outside these checkers' error model.
    """
    import sys

    names = list(distance_matrix.names)
    vals = [
        abs(float(distance_matrix[names[i], names[j]]))
        for i in range(len(names))
        for j in range(i)
    ]
    if not all(math.isfinite(v) for v in vals):
        return False
    max_value = max(vals, default=0.0)
    return max_value <= sys.float_info.max / (64.0 * max(len(names), 1))


def _input_is_additive(distance_matrix, tree=None):
    """True when the input distance matrix is additive: it is exactly
    realisable on some tree with **all-non-negative** branch lengths.

    Additivity is the precondition for NJ's order-independence and exact-
    reconstruction theorems (Saitou-Nei 1987; Studier-Keppler 1988). On a
    non-additive matrix NJ may legitimately return different trees for
    different taxon orders (tied Q-matrix minima) and may return negative
    branch lengths -- both are standard, documented NJ behaviour, not defects,
    so the checkers gated on this predicate stay silent there.

    This is decided directly on the input via the four-point condition: for
    every quartet {i,j,k,l} the two largest of d(ij)+d(kl), d(ik)+d(jl),
    d(il)+d(jk) must be equal (Buneman 1974). It does *not* ask whether NJ's
    output reproduces the matrix -- NJ can fit a non-additive matrix exactly
    using a negative pendant edge, and such a fit is itself proof of
    non-additivity, so keying off it would be circular.
    """
    names = list(distance_matrix.names)
    n = len(names)
    if not _matrix_arithmetic_safe(distance_matrix):
        return False
    if n < 4:
        return True

    def d(a, b):
        return 0.0 if a == b else float(distance_matrix[names[a], names[b]])

    # Additivity is an EXACT algebraic condition (Buneman 1974): for a true
    # tree metric the two larger quartet sums are equal, not merely close.
    # The only slack allowed is floating-point round-off in forming the sums,
    # which is a few ULP of their own magnitude -- so the tolerance is
    # ~1e-12 relative, NOT a 1e-6 fudge. A looser, magnitude-proportional
    # tolerance (an earlier version used 1e-6 * max(1, |sum|)) grows the slack
    # to ~1 on a 1e6-scale matrix and lets a plainly non-additive matrix pass
    # (methodology 8.3). The scale is the largest distance in the matrix.
    scale = max((d(a, b) for a in range(n) for b in range(a)), default=1.0)
    scale = scale if math.isfinite(scale) and scale > 0.0 else 1.0
    tol = 64.0 * 2.220446049250313e-16 * scale  # a few dozen ULP of the scale

    # A non-negative tree metric is a metric: distances non-negative and the
    # triangle inequality holds on every triple. (Buneman's four-point
    # condition alone allows negative edges; the triangle inequality is what
    # rules those out.)
    for i in range(n):
        for j in range(n):
            if i != j and d(i, j) < -tol:
                return False
    for i, j, k in itertools.permutations(range(n), 3):
        if d(i, j) + d(j, k) < d(i, k) - tol:
            return False
    for quartet in itertools.combinations(range(n), 4):
        i, j, k, l = quartet
        sums = sorted((
            d(i, j) + d(k, l),
            d(i, k) + d(j, l),
            d(i, l) + d(j, k),
        ))
        # The two largest sums must coincide.
        if abs(sums[2] - sums[1]) > tol:
            return False
    return True


@_guard("nj_leaf_order_invariance")
def check_nj_leaf_order(distance_matrix, tree):
    """BP-PHY-001: for an *additive* distance matrix, neighbour joining
    reconstructs the unique generating tree, so it is independent of the order
    the taxa are listed in: permuting the rows/columns of the input must yield
    a tree with identical patristic distances between every pair of leaves. An
    order dependence on additive input means the join order (and hence the
    inferred phylogeny) is being decided by an implementation artefact rather
    than by the data. The checker does nothing on non-additive input, where
    tied Q-matrix minima make several distinct NJ trees equally valid.
    """
    import random

    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

    n = len(distance_matrix)
    if n < 4 or not _input_is_additive(distance_matrix, tree):
        return
    names = list(distance_matrix.names)
    perm = names[:]
    random.Random(0).shuffle(perm)
    if perm == names:
        return
    permuted = DistanceMatrix(list(perm))
    for i in range(n):
        for j in range(i):
            permuted[perm[i], perm[j]] = distance_matrix[perm[i], perm[j]]

    other = DistanceTreeConstructor().nj(permuted)
    d0 = _patristic(tree)
    d1 = _patristic(other)
    if set(d0) != set(d1):
        trigger("BP-PHY-001")
        return
    tol = _matrix_tol(distance_matrix)
    differs = any(not _within(d0[k], d1[k], tol) for k in d0)
    trigger_if(differs, "BP-PHY-001")


@_guard("nj_additivity")
def check_nj_additivity(distance_matrix, tree):
    """BP-PHY-002: the correctness theorem for neighbour joining -- if the input
    distances are *additive* (exactly realisable as path lengths on a weighted
    tree), NJ reconstructs that tree, so the output patristic distances equal
    the input distances. The checker reads the produced tree's own branch
    lengths (never a re-derived distance formula) to decide whether the input
    was additive; if it was, any deviation of an output patristic distance from
    the input distance means NJ failed to recover a tree it provably should
    have. Non-additive input is out of scope (NJ makes no such guarantee, and
    negative branch lengths there are documented, expected behaviour).
    """
    n = len(distance_matrix)
    if n < 4 or not _input_is_additive(distance_matrix, tree):
        return
    # Additive input: NJ must reproduce every input distance exactly.
    patristic = _patristic(tree)
    names = list(distance_matrix.names)
    tol = _matrix_tol(distance_matrix)
    bad = False
    for i in range(n):
        for j in range(i):
            key = tuple(sorted((names[i], names[j])))
            if not _within(
                patristic[key], distance_matrix[names[i], names[j]], tol
            ):
                bad = True
    trigger_if(bad, "BP-PHY-002")


@_guard("upgma_ultrametric")
def check_upgma_ultrametric(distance_matrix, tree):
    """BP-PHY-003: UPGMA produces a *rooted, ultrametric* tree -- it assumes a
    molecular clock, so every leaf is equidistant from the root and, for any
    three leaves, the two largest of the three pairwise patristic distances are
    equal (the three-point / strong-triangle condition). A violation means the
    dendrogram cannot be interpreted as a clock-like divergence history, which
    is the whole modelling premise of UPGMA.
    """
    if len(distance_matrix) < 3 or not _matrix_arithmetic_safe(distance_matrix):
        return
    # Tolerance scales with the input distance magnitude (methodology 8.3):
    # a fixed 1e-6 is far too tight on a matrix with entries ~1e4.
    tol = _matrix_tol(distance_matrix)
    terminals = tree.get_terminals()
    root = tree.root
    depths = tree.depths()
    # all leaves equidistant from the root
    leaf_depths = [depths[t] for t in terminals if t in depths]
    if leaf_depths:
        d0 = leaf_depths[0]
        trigger_if(
            any(not _within(d, d0, tol) for d in leaf_depths),
            "BP-PHY-003",
        )
    # three-point condition
    patristic = _patristic(tree)
    names = [t.name for t in terminals]
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            for c in range(b + 1, len(names)):
                trio = sorted([
                    patristic[tuple(sorted((names[a], names[b])))],
                    patristic[tuple(sorted((names[a], names[c])))],
                    patristic[tuple(sorted((names[b], names[c])))],
                ])
                trigger_if(not _within(trio[1], trio[2], tol),
                           "BP-PHY-003")


@_guard("tree_branch_length_nonnegative")
def check_tree_branch_lengths(distance_matrix, tree, method):
    """BP-PHY-004: a branch length is an amount of evolutionary change (expected
    substitutions per site) and cannot be negative. For UPGMA on any valid
    distance matrix, and for NJ on an additive matrix, all estimated branch
    lengths are non-negative; a negative value is a spurious "negative
    evolutionary time" and distorts every downstream length-weighted analysis
    (patristic distance, rate estimation, ancestral-state reconstruction).
    """
    if not _matrix_arithmetic_safe(distance_matrix):
        return
    lengths = _all_branch_lengths(tree)
    if not lengths:
        return
    negative = min(lengths)
    # A branch length is an amount of change; "zero" here means zero to the
    # reconstruction's float64 round-off, which scales with the input distance
    # magnitude (methodology 8.3). A fixed -1e-6 fired at matrix scale ~1e12
    # where the true relative negativity was still ~1e-17 -- audit 4, BP-PHY-004.
    # PHY-001/002/003 already scale by _matrix_scale; this site was missed.
    tol = -_matrix_tol(distance_matrix)
    if method == "upgma":
        trigger_if(negative < tol, "BP-PHY-004")
        return
    # NJ: only flag when the input is additive (negative lengths are otherwise
    # an accepted outcome of NJ on non-additive data).
    if len(distance_matrix) < 4 or not _input_is_additive(distance_matrix, tree):
        return
    trigger_if(negative < tol, "BP-PHY-004")


# ======================================================================
# Bio.motifs.matrix  (PSSM / PWM)
# ======================================================================

def _pssm_columns(pssm):
    """Return the PSSM as a list of {letter: logodds} dicts, one per position."""
    return [
        {letter: pssm[letter][i] for letter in pssm.alphabet}
        for i in range(pssm.length)
    ]


def _pssm_logodds_bounded(pssm, cap=1e6):
    """True when every PSSM entry is a plausible log-odds value (|entry| < cap).

    ``PSSM.calculate`` accumulates the site score in ``np.float32`` (a
    documented space optimisation). A log-odds score of a 4-letter column is
    bounded in magnitude by ~20 in practice; even pathological pseudo-count
    choices stay well below ``cap``. With entries at ~1e30 the true site score
    can be 10+ orders of magnitude below one float32 ULP of the summed terms,
    so no float32 accumulation can be strand-symmetric or match a float64
    column sum -- that is outside the input class where the PSSM model is
    meaningful, not a defect (audit 4, BP-MTF-003). The additivity, bounds,
    and reverse-complement laws are stated over real log-odds matrices.
    """
    for col in _pssm_columns(pssm):
        for v in col.values():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return False
            if math.isfinite(fv) and abs(fv) >= cap:
                return False
    return True


@_guard("pssm_score_additivity")
def check_pssm_score_additivity(pssm, sequence, result):
    """BP-MTF-001: a position-specific scoring matrix scores a site as the *sum*
    of the per-position log-odds contributions (independence of positions is
    the defining assumption of the PSSM model). The C routine ``_pwm.calculate``
    and a direct column-wise sum are two independent implementations of that
    same sum and must agree. A mismatch means the reported motif score is not
    the log-odds of the site under the model.
    """
    text = _seq_text(sequence)
    if text is None:
        return
    text = text.upper()
    m = pssm.length
    if len(text) < m or set(text) - set("ACGT"):
        return
    if not _pssm_logodds_bounded(pssm):
        return
    cols = _pssm_columns(pssm)
    # score at offset 0 only
    expected = 0.0
    for i in range(m):
        expected += cols[i][text[i]]
    try:
        observed = float(result if not hasattr(result, "__len__") else result[0])
    except Exception:
        return
    if not math.isfinite(expected) or not math.isfinite(observed):
        return
    trigger_if(abs(observed - expected) > 1e-3 * max(1.0, abs(expected)),
               "BP-MTF-001")


@_guard("pssm_score_bounds")
def check_pssm_score_bounds(pssm, sequence, result):
    """BP-MTF-002: every site score lies between ``pssm.min`` (the score of the
    anticonsensus, the least motif-like sequence) and ``pssm.max`` (the score of
    the consensus). These bounds are what threshold-based motif search and the
    score-to-p-value mapping rely on; a score outside them means the reported
    value cannot be located on the motif's score distribution.
    """
    text = _seq_text(sequence)
    if text is None:
        return
    text = text.upper()
    if set(text) - set("ACGT"):
        return
    if not _pssm_logodds_bounded(pssm):
        return
    try:
        lo, hi = float(pssm.min), float(pssm.max)
    except Exception:
        return
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return
    # ``PSSM.calculate`` accumulates the score in float32 (C routine) while
    # ``pssm.min``/``pssm.max`` are float64 column-wise sums, so a sequence that
    # is exactly the anticonsensus/consensus lands on the bound up to float32
    # rounding. Scale the tolerance with the bound magnitude, as BP-MTF-001/003.
    tol = 1e-3 * max(1.0, abs(lo), abs(hi))
    scores = result if hasattr(result, "__len__") else [result]
    for s in scores:
        s = float(s)
        if not math.isfinite(s):
            continue
        trigger_if(s < lo - tol or s > hi + tol, "BP-MTF-002")


@_guard("pssm_reverse_complement_symmetry")
def check_pssm_revcomp(pssm, sequence, result):
    """BP-MTF-003: a transcription-factor binding site can occur on either
    strand. Scoring sequence S with the motif's PSSM must give the same value as
    scoring reverse-complement(S) with reverse_complement(PSSM) -- the two
    describe the identical physical binding event read from opposite strands.
    An asymmetry means strand choice silently changes the motif score, biasing
    every both-strands genome scan.
    """
    text = _seq_text(sequence)
    if text is None:
        return
    text = text.upper()
    m = pssm.length
    if len(text) != m or set(text) - set("ACGT"):
        return
    if not _pssm_logodds_bounded(pssm):
        return
    try:
        forward = float(result if not hasattr(result, "__len__") else result[0])
    except Exception:
        return
    if not math.isfinite(forward):
        return
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rc_text = "".join(comp[b] for b in reversed(text))
    rc_pssm = pssm.reverse_complement()
    rc_score = rc_pssm.calculate(rc_text)
    rc_val = float(rc_score if not hasattr(rc_score, "__len__") else rc_score[0])
    if not math.isfinite(rc_val):
        return
    trigger_if(abs(forward - rc_val) > 1e-3 * max(1.0, abs(forward)),
               "BP-MTF-003")
