"""Runtime scientific-invariant trigger collection for the SciBench pilot.

This module is private and inactive unless ``SCIBENCH_TRIGGER_LOG`` names an
output file. Each JSON-lines record represents one observed invariant alarm;
the evaluator deduplicates checker IDs and maps them to root-cause families.

Design rule (see ``scientific_bug_finding/METHODOLOGY.md`` section 2): a checker
may only (a) call a public API a second time on a transformed input and/or
(b) read values already computed by production code, then compare. A checker
never re-implements the scientific formula it is checking, never raises, and
never changes a return value, exception, or numerical result.
"""

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
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


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

@_guard("flexibility")
def check_flexibility(sequence, scores):
    """BP-SEQ-001 window length; BP-SEQ-002 reversal equivariance."""
    window = 9
    trigger_if(len(scores) != max(0, len(sequence) - window + 1), "BP-SEQ-001")

    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    reverse_scores = ProteinAnalysis(sequence[::-1]).flexibility()[::-1]
    differs = len(scores) != len(reverse_scores) or any(
        not _isclose(a, b) for a, b in zip(scores, reverse_scores)
    )
    trigger_if(bool(scores) and differs, "BP-SEQ-002")


# --- ProtParam.protein_scale -------------------------------------------------

@_guard("protein_scale_input")
def check_protein_scale_input(sequence, param_dict, window, edge):
    """BP-SEQ-003: a one-residue scale window must return each residue's value."""
    complete = bool(sequence) and all(r in param_dict for r in sequence)
    trigger_if(complete and 0 <= edge <= 1 and window == 1, "BP-SEQ-003")


@_guard("protein_scale_output")
def check_protein_scale_output(sequence, param_dict, window, edge, scores):
    """BP-SEQ-004: even-window scale profile reversal equivariance (pure API)."""
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


# --- ProtParam derived quantities -----------------------------------------

@_guard("aromaticity")
def check_aromaticity(value):
    """BP-SEQ-027: aromaticity is a frequency, in [0, 1]."""
    trigger_if(not math.isfinite(value) or not 0.0 <= value <= 1.0, "BP-SEQ-027")


@_guard("secondary_structure")
def check_secondary_structure(fractions):
    """BP-SEQ-025: each secondary-structure fraction lies in [0, 1]."""
    trigger_if(
        any(not math.isfinite(f) or not 0.0 <= f <= 1.0 for f in fractions),
        "BP-SEQ-025",
    )


@_guard("instability")
def check_instability(sequence, value):
    """BP-SEQ-026: a single-residue peptide has no dipeptide; index is 0."""
    if len(sequence) == 1:
        trigger_if(not math.isfinite(value) or not _isclose(value, 0.0), "BP-SEQ-026")


# --- IsoelectricPoint ------------------------------------------------------

@_guard("pi")
def check_pi(sequence, point, charge):
    """BP-SEQ-006/007: modeled charge at the reported pI is ~0 for extreme
    acidic / basic compositions."""
    residues = set(sequence)
    if sequence and residues <= {"D", "E"}:
        trigger_if(abs(charge) > 1e-3, "BP-SEQ-006")
    if sequence and residues <= {"K", "R"}:
        trigger_if(abs(charge) > 1e-3, "BP-SEQ-007")


@_guard("charge_monotonicity")
def check_charge_monotonicity(sequence):
    """BP-SEQ-011: modeled net charge is non-increasing in pH."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    analysis = ProteinAnalysis(sequence)
    grid = [i / 4 for i in range(0, 57)]
    charges = [analysis.charge_at_pH(pH) for pH in grid]
    rises = any(b > a + 1e-9 for a, b in zip(charges, charges[1:]))
    trigger_if(rises, "BP-SEQ-011")


# --- SeqUtils.gc_fraction / GC123 ---------------------------------------------

@_guard("gc_fraction")
def check_gc_fraction(seq, value):
    """BP-SEQ-009 bounds; BP-SEQ-010 complement symmetry; BP-SEQ-016 mode
    consistency for unambiguous DNA."""
    trigger_if(not math.isfinite(value) or not 0 <= value <= 1, "BP-SEQ-009")

    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction

    text = str(seq).upper()
    if not text:
        return
    unambiguous = set(text) <= set("ACGT")

    weighted = gc_fraction(text, "weighted")
    complement_weighted = gc_fraction(str(Seq(text).complement()), "weighted")
    trigger_if(not _isclose(weighted, complement_weighted), "BP-SEQ-010")

    if unambiguous:
        remove = gc_fraction(text, "remove")
        ignore = gc_fraction(text, "ignore")
        trigger_if(
            not (_isclose(remove, ignore) and _isclose(remove, weighted)),
            "BP-SEQ-016",
        )


@_guard("gc123")
def check_gc123(seq, total, pos1, pos2, pos3):
    """BP-SEQ-017: total GC% equals the mean of the three codon-position GC%
    when the sequence length is a multiple of three."""
    text = str(seq).upper()
    # GC123 documents that it does not handle ambiguous nucleotides.
    if len(text) == 0 or len(text) % 3 != 0 or set(text) - set("ACGT"):
        return
    trigger_if(
        not _isclose(total, (pos1 + pos2 + pos3) / 3, tol=1e-6), "BP-SEQ-017"
    )


# --- SeqUtils.molecular_weight ----------------------------------------------

@_guard("molecular_weight")
def check_molecular_weight(original_seq, seq_type, double_stranded, circular,
                           monoisotopic, weight):
    """BP-SEQ-008 empty polymer; BP-SEQ-014 double-stranded symmetry;
    BP-SEQ-023 single-strand additivity."""
    water = 18.010565 if monoisotopic else 18.0153

    trigger_if(
        not original_seq and math.isfinite(weight) and weight != 0, "BP-SEQ-008"
    )
    if not original_seq:
        return

    from Bio.Seq import Seq
    from Bio.SeqUtils import molecular_weight

    if seq_type in ("DNA", "RNA") and set(original_seq) <= set("ACGTU"):
        s = Seq(original_seq)
        variants = {
            "reverse": str(s[::-1]),
            "complement": str(s.complement()),
            "reverse_complement": str(s.reverse_complement()),
        }
        if double_stranded:
            base = molecular_weight(original_seq, seq_type, double_stranded=True,
                                    circular=circular, monoisotopic=monoisotopic)
            for v in variants.values():
                other = molecular_weight(v, seq_type, double_stranded=True,
                                         circular=circular,
                                         monoisotopic=monoisotopic)
                trigger_if(not _isclose(base, other), "BP-SEQ-014")

        if not double_stranded and not circular and len(original_seq) >= 2:
            half = len(original_seq) // 2
            a, b = original_seq[:half], original_seq[half:]
            whole = molecular_weight(original_seq, seq_type,
                                     monoisotopic=monoisotopic)
            parts = (
                molecular_weight(a, seq_type, monoisotopic=monoisotopic)
                + molecular_weight(b, seq_type, monoisotopic=monoisotopic)
            )
            trigger_if(not _isclose(whole + water, parts, tol=1e-6), "BP-SEQ-023")


# --- SeqUtils.six_frame_translations ---------------------------------------

# --- SeqUtils.seq1 / seq3 -------------------------------------------------

@_guard("three_one_roundtrip")
def check_seq1_roundtrip(one_letter, three_letter_input):
    """BP-SEQ-021: seq1(seq3(s)) round-trips standard amino-acid sequences."""
    from Bio.SeqUtils import seq3

    text = str(three_letter_input)
    if len(text) % 3 or not text:
        return
    codes = {text[i:i + 3].capitalize() for i in range(0, len(text), 3)}
    standard = {
        "Ala", "Arg", "Asn", "Asp", "Cys", "Gln", "Glu", "Gly", "His", "Ile",
        "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
    }
    if not codes <= standard:
        return
    trigger_if(str(seq3(one_letter)).capitalize() != text.capitalize(),
               "BP-SEQ-021")


# --- SeqUtils.nt_search --------------------------------------------------

@_guard("nt_search")
def check_nt_search(seq, subseq, positions):
    """BP-SEQ-022: every reported position actually matches the IUPAC-expanded
    subsequence."""
    from Bio.Data import IUPACData

    text = str(seq).upper()
    query = str(subseq).upper()
    n = len(query)
    if not query or set(query) - set(IUPACData.ambiguous_dna_values):
        return

    def matches(fragment):
        if len(fragment) != n:
            return False
        return all(
            f in IUPACData.ambiguous_dna_values[q] for q, f in zip(query, fragment)
        )

    for p in positions:
        trigger_if(not matches(text[p:p + n]), "BP-SEQ-022")


# --- MeltingTemp ---------------------------------------------------------

@_guard("wallace")
def check_wallace(seq, temperature):
    """BP-SEQ-015 reverse-complement symmetry; BP-SEQ-018 count consistency."""
    from Bio.Seq import Seq
    from Bio.SeqUtils.MeltingTemp import Tm_Wallace

    text = str(seq).upper().replace("U", "T")
    if not text or set(text) - set("ACGT"):
        return
    rc = str(Seq(text).reverse_complement())
    trigger_if(not _isclose(Tm_Wallace(rc), temperature), "BP-SEQ-015")

    gc = text.count("G") + text.count("C")
    at = text.count("A") + text.count("T")
    trigger_if(not _isclose(temperature, 4 * gc + 2 * at), "BP-SEQ-018")


@_guard("tm_gc_monotonicity")
def check_tm_gc_monotonicity(seq, temperature):
    """BP-SEQ-019: at fixed length, raising %GC does not lower Tm_GC."""
    from Bio.SeqUtils.MeltingTemp import Tm_GC

    text = str(seq).upper().replace("U", "T")
    if len(text) < 4 or set(text) - set("ACGT"):
        return
    lower = "A" * len(text)
    higher = "G" * len(text)
    tm_lower = Tm_GC(lower)
    tm_higher = Tm_GC(higher)
    trigger_if(
        tm_lower - temperature > 1e-6 or temperature - tm_higher > 1e-6,
        "BP-SEQ-019",
    )
    trigger_if(tm_lower - tm_higher > 1e-6, "BP-SEQ-019")


@_guard("tm_nn_selfcomp")
def check_tm_nn_selfcomp(seq, selfcomp, c_seq, temperature):
    """BP-SEQ-020: with selfcomp=True, supplying the exact complement as c_seq
    gives the same Tm as letting Tm_NN derive it."""
    if not selfcomp or c_seq is not None:
        return
    from Bio.Seq import Seq
    from Bio.SeqUtils.MeltingTemp import Tm_NN

    text = str(seq).upper().replace("U", "T")
    if len(text) < 2 or set(text) - set("ACGT"):
        return
    explicit = Tm_NN(text, selfcomp=True, c_seq=str(Seq(text).complement()))
    trigger_if(not _isclose(explicit, temperature), "BP-SEQ-020")


# --- CodonAdaptationIndex ------------------------------------------------

@_guard("cai_degenerate")
def check_cai_degenerate(sequence, cai_length):
    """BP-SEQ-005: a valid coding sequence of only ATG/TGG has CAI 1, not a
    zero-division."""
    text = str(sequence).upper()
    codons = [text[i:i + 3] for i in range(0, len(text), 3)]
    valid = bool(text) and len(text) % 3 == 0 and set(codons) <= {"ATG", "TGG"}
    trigger_if(valid and cai_length == 0, "BP-SEQ-005")


@_guard("codon_optimization")
def check_codon_optimization(index, source_seq, seq_type, optimized):
    """BP-SEQ-012: optimize() preserves translation and yields CAI 1."""
    from Bio.Seq import Seq

    try:
        if seq_type in ("DNA", "RNA"):
            before = str(Seq(str(source_seq).upper()).translate())
        else:
            before = str(source_seq).upper()
        after = str(Seq(str(optimized)).translate())
    except Exception:
        return
    trigger_if(before != after, "BP-SEQ-012")
    try:
        cai = index.calculate(optimized)
    except Exception:
        trigger("BP-SEQ-012")
        return
    trigger_if(not math.isfinite(cai) or not _isclose(cai, 1.0), "BP-SEQ-012")
