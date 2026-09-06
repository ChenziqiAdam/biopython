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

@_guard("protein_scale_window_one")
def check_protein_scale_window_one(sequence, param_dict):
    """BP-SEQ-003: a one-residue scale window must return each residue's raw
    scale value.

    Observed after the fact: the checker independently re-calls protein_scale
    with window=1 and edge=1 and compares the profile to [param_dict[r] for r
    in sequence]. A raised exception or a mismatch is the alarm; simply being
    called with window=1 is not.
    """
    if not sequence or not all(r in param_dict for r in sequence):
        return
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    expected = [param_dict[r] for r in sequence]
    try:
        profile = ProteinAnalysis(sequence).protein_scale(param_dict, 1, edge=1)
    except Exception:
        trigger("BP-SEQ-003")
        return
    differs = len(profile) != len(expected) or any(
        not _isclose(a, b) for a, b in zip(profile, expected)
    )
    trigger_if(differs, "BP-SEQ-003")


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


# --- MeltingTemp: Tm_NN, salt_correction, chem_correction ----------------

@_guard("tm_nn_revcomp")
def check_tm_nn_revcomp(original_seq, c_seq, shift, selfcomp, nn_table, saltcorr,
                        Na, K, Tris, Mg, dNTPs, temperature):
    """BP-SEQ-028: Tm_NN of a perfect DNA/DNA duplex is invariant under reverse
    complementation of the primer (all other parameters held fixed)."""
    if c_seq is not None or shift or selfcomp or nn_table is not None:
        return
    from Bio.Seq import Seq
    from Bio.SeqUtils.MeltingTemp import Tm_NN

    text = str(original_seq).upper().replace("U", "T")
    if len(text) < 2 or set(text) - set("ACGT"):
        return
    rc = str(Seq(text).reverse_complement())
    rc_temp = Tm_NN(rc, saltcorr=saltcorr, Na=Na, K=K, Tris=Tris, Mg=Mg,
                    dNTPs=dNTPs)
    trigger_if(not _isclose(rc_temp, temperature, tol=1e-6), "BP-SEQ-028")


@_guard("tm_nn_salt")
def check_tm_nn_salt(original_seq, c_seq, shift, selfcomp, saltcorr, Na, K, Tris,
                     Mg, dNTPs, temperature):
    """BP-SEQ-029: raising [Na+] does not lower Tm_NN (salt stabilises a duplex,
    salt-correction methods 1-4)."""
    if (c_seq is not None or shift or selfcomp or saltcorr not in (1, 2, 3, 4)
            or K or Tris or Mg or dNTPs):
        return
    from Bio.SeqUtils.MeltingTemp import Tm_NN

    text = str(original_seq).upper().replace("U", "T")
    if len(text) < 2 or set(text) - set("ACGT") or Na <= 0:
        return
    lower = Tm_NN(text, saltcorr=saltcorr, Na=Na / 2.0)
    higher = Tm_NN(text, saltcorr=saltcorr, Na=Na * 2.0)
    trigger_if(lower - temperature > 1e-6 or temperature - higher > 1e-6,
               "BP-SEQ-029")
    trigger_if(lower - higher > 1e-6, "BP-SEQ-029")


@_guard("salt_correction_monotonicity")
def check_salt_correction(Na, K, Tris, Mg, dNTPs, method, seq, corr):
    """BP-SEQ-030: for methods 1, 3, 4 the salt-correction term is
    k * log10([Na+] in M): monotonically increasing in [Na+] and exactly 0 at
    [Na+] = 1 M."""
    if method not in (1, 3, 4) or K or Tris or Mg or dNTPs or Na <= 0:
        return
    from Bio.SeqUtils.MeltingTemp import salt_correction

    lower = salt_correction(Na=Na / 2.0, method=method, seq=seq)
    higher = salt_correction(Na=Na * 2.0, method=method, seq=seq)
    trigger_if(not (lower <= corr <= higher), "BP-SEQ-030")
    at_one_molar = salt_correction(Na=1000.0, method=method, seq=seq)
    trigger_if(abs(at_one_molar) > 1e-9, "BP-SEQ-030")


@_guard("chem_correction_identity")
def check_chem_correction(melting_temp, DMSO, fmd, result):
    """BP-SEQ-031: with no additive the Tm is unchanged; adding DMSO strictly
    lowers it."""
    if DMSO == 0 and fmd == 0:
        trigger_if(not _isclose(result, melting_temp), "BP-SEQ-031")
    if DMSO > 0 and fmd == 0:
        trigger_if(result >= melting_temp, "BP-SEQ-031")


# --- SeqUtils.GC_skew --------------------------------------------------

@_guard("gc_skew")
def check_gc_skew(seq, window, values):
    """BP-SEQ-032 bounds; BP-SEQ-033 complement antisymmetry."""
    trigger_if(any(not math.isfinite(v) or not -1.0 <= v <= 1.0 for v in values),
               "BP-SEQ-032")

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


# --- ProtParam: gravy, protein molecular_weight, composition -------------

@_guard("gravy")
def check_gravy(sequence, scale, value):
    """BP-SEQ-034 permutation invariance; BP-SEQ-035 homopolymer identity."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    if not sequence:
        return
    reversed_value = ProteinAnalysis(sequence[::-1]).gravy(scale)
    trigger_if(not _isclose(value, reversed_value), "BP-SEQ-034")

    if len(set(sequence)) == 1:
        from Bio.SeqUtils import ProtParamData

        table = ProtParamData.gravy_scales.get(scale)
        if table is not None and sequence[0] in table:
            trigger_if(not _isclose(value, table[sequence[0]]), "BP-SEQ-035")


@_guard("protein_molecular_weight")
def check_protein_molecular_weight(sequence, weight):
    """BP-SEQ-036: a protein's average mass is invariant under residue
    permutation (it is a composition sum minus condensation water)."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    if len(sequence) < 2:
        return
    reversed_weight = ProteinAnalysis(sequence[::-1]).molecular_weight()
    trigger_if(not _isclose(weight, reversed_weight, tol=1e-6), "BP-SEQ-036")


@_guard("aa_composition")
def check_aa_composition(sequence, percentages):
    """BP-SEQ-037: for a sequence of only standard amino acids the composition
    percentages sum to 100."""
    from Bio.Data import IUPACData

    if not sequence or set(sequence) - set(IUPACData.protein_letters):
        return
    trigger_if(not _isclose(sum(percentages.values()), 100.0, tol=1e-6),
               "BP-SEQ-037")


@_guard("instability_homopolymer")
def check_instability_homopolymer(sequence, value):
    """BP-SEQ-038: for a homopolymer of length L the instability index is
    10 * (L - 1) / L * DIWV[a][a], approaching a constant as L grows."""
    if len(sequence) < 2 or len(set(sequence)) != 1:
        return
    from Bio.SeqUtils import ProtParamData

    a = sequence[0]
    try:
        expected = (10.0 / len(sequence)) * (len(sequence) - 1) * ProtParamData.DIWV[a][a]
    except KeyError:
        return
    trigger_if(not _isclose(value, expected, tol=1e-6), "BP-SEQ-038")


# --- SeqUtils.molecular_weight: RNA vs DNA ------------------------------

@_guard("rna_dna_mass_ordering")
def check_rna_dna_mass_ordering(original_seq, seq_type, double_stranded, circular,
                                monoisotopic, weight):
    """BP-SEQ-042: for the same base string, single-stranded RNA is heavier
    than single-stranded DNA (an extra 2'-OH per nucleotide)."""
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
    """Apply a fixed rotation + translation to a list of 3-tuples/arrays."""
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
    shift = np.array([7.0, -3.0, 11.0])
    return [rot @ np.asarray(p, dtype=float) + shift for p in points]


@_guard("pdb_calc_angle")
def check_calc_angle(v1, v2, v3, angle):
    """BP-PDB-001 range; BP-PDB-002 endpoint symmetry; BP-PDB-003 rigid
    invariance."""
    import math as _m

    from Bio.PDB.vectors import calc_angle, Vector

    trigger_if(not _m.isfinite(angle) or not (0.0 <= angle <= _m.pi + 1e-9),
               "BP-PDB-001")
    trigger_if(not _isclose(angle, calc_angle(v3, v2, v1)), "BP-PDB-002")

    p1, p2, p3 = _rigid_transform(
        [v1.get_array(), v2.get_array(), v3.get_array()]
    )
    moved = calc_angle(Vector(p1), Vector(p2), Vector(p3))
    trigger_if(not _isclose(angle, moved, tol=1e-7), "BP-PDB-003")


@_guard("pdb_calc_dihedral")
def check_calc_dihedral(v1, v2, v3, v4, angle):
    """BP-PDB-004 range; BP-PDB-005 full-path-reversal invariance; BP-PDB-006
    rigid invariance; BP-PDB-007 chirality (mirror negates)."""
    import math as _m

    from Bio.PDB.vectors import calc_dihedral, Vector

    trigger_if(
        not _m.isfinite(angle) or not (-_m.pi - 1e-9 <= angle <= _m.pi + 1e-9),
        "BP-PDB-004",
    )
    # Reversing the whole path (a,b,c,d) -> (d,c,b,a) preserves the rotation
    # sense about the central bond, so the signed dihedral is unchanged.
    trigger_if(not _isclose(angle, calc_dihedral(v4, v3, v2, v1), tol=1e-7)
               and not _isclose(abs(angle), _m.pi, tol=1e-7),
               "BP-PDB-005")

    p = _rigid_transform(
        [v1.get_array(), v2.get_array(), v3.get_array(), v4.get_array()]
    )
    moved = calc_dihedral(Vector(p[0]), Vector(p[1]), Vector(p[2]), Vector(p[3]))
    trigger_if(not _isclose(angle, moved, tol=1e-7), "BP-PDB-006")

    def mirror(v):
        a = v.get_array().copy()
        a[2] = -a[2]
        return Vector(a)

    mirrored = calc_dihedral(mirror(v1), mirror(v2), mirror(v3), mirror(v4))
    trigger_if(not _isclose(angle, -mirrored, tol=1e-7)
               and not _isclose(abs(angle), _m.pi, tol=1e-7),
               "BP-PDB-007")


@_guard("pdb_vector_angle")
def check_vector_angle(self_vec, other_vec, angle):
    """BP-PDB-008: Vector.angle is in [0, pi] and symmetric."""
    import math as _m

    trigger_if(not _m.isfinite(angle) or not (0.0 <= angle <= _m.pi + 1e-9),
               "BP-PDB-008")
    trigger_if(not _isclose(angle, other_vec.angle(self_vec)), "BP-PDB-008")


@_guard("pdb_cross_product")
def check_cross_product(left, right, result):
    """BP-PDB-009: the cross product is orthogonal to both operands and
    anti-commutes."""
    la, ra, xa = left.get_array(), right.get_array(), result.get_array()
    import numpy as np

    scale = (np.linalg.norm(la) * np.linalg.norm(ra)) or 1.0
    trigger_if(abs(float(np.dot(xa, la))) / scale > 1e-9, "BP-PDB-009")
    trigger_if(abs(float(np.dot(xa, ra))) / scale > 1e-9, "BP-PDB-009")
    reverse = (right ** left).get_array()
    trigger_if(not np.allclose(xa, -reverse, atol=1e-9), "BP-PDB-009")


@_guard("pdb_normalize")
def check_normalize(original_array, normalized_vec):
    """BP-PDB-010: normalizing a non-zero vector gives unit norm and preserves
    direction."""
    import numpy as np

    orig = np.asarray(original_array, dtype=float)
    if np.linalg.norm(orig) < 1e-12:
        return
    unit = normalized_vec.get_array()
    trigger_if(not _isclose(float(np.linalg.norm(unit)), 1.0, tol=1e-9),
               "BP-PDB-010")
    cross = np.cross(orig, unit)
    trigger_if(float(np.linalg.norm(cross)) / float(np.linalg.norm(orig)) > 1e-9,
               "BP-PDB-010")


@_guard("pdb_rotmat")
def check_rotmat(p, q, matrix):
    """BP-PDB-011 orthogonality (det +1); BP-PDB-012 maps p onto q."""
    import numpy as np

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


@_guard("pdb_rotaxis")
def check_rotaxis(theta, axis_vec, matrix):
    """BP-PDB-013: an axis-angle rotation matrix is orthogonal with det +1 and
    rotaxis(theta) @ rotaxis(-theta) == I."""
    import numpy as np

    from Bio.PDB.vectors import rotaxis2m

    trigger_if(not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9),
               "BP-PDB-013")
    trigger_if(not _isclose(float(np.linalg.det(matrix)), 1.0, tol=1e-9),
               "BP-PDB-013")
    inverse = rotaxis2m(-theta, axis_vec)
    trigger_if(not np.allclose(matrix @ inverse, np.eye(3), atol=1e-9),
               "BP-PDB-013")


@_guard("pdb_qcp")
def check_qcp(reference_coords, coords, rms, init_rms):
    """BP-PDB-014 nonnegative and bounded by init RMSD; BP-PDB-015 rigid
    invariance and symmetry."""
    import numpy as np

    from Bio.PDB.qcprot import QCPSuperimposer

    trigger_if(not np.isfinite(rms) or rms < -1e-9, "BP-PDB-014")
    if init_rms is not None:
        trigger_if(rms > init_rms + 1e-7, "BP-PDB-014")

    ref = np.asarray(reference_coords, dtype=float)
    mov = np.asarray(coords, dtype=float)
    if ref.shape != mov.shape or ref.shape[0] < 3:
        return

    moved = np.asarray(_rigid_transform(list(mov)))
    sup = QCPSuperimposer()
    sup.set(ref, moved)
    sup.run()
    trigger_if(not _isclose(sup.get_rms(), rms, tol=1e-6), "BP-PDB-015")

    swapped = QCPSuperimposer()
    swapped.set(mov, ref)
    swapped.run()
    trigger_if(not _isclose(swapped.get_rms(), rms, tol=1e-6), "BP-PDB-015")
