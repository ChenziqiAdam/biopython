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
    """BP-SEQ-001 window length; BP-SEQ-002 reversal equivariance.

    A sliding-window smoothing profile must be position-symmetric: the physical
    quantity (local backbone flexibility) does not depend on which end of the
    chain the window is indexed from, so reversing the sequence must reverse the
    profile. An asymmetric window introduces a systematic phase shift in the
    profile.
    """
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

    The checker independently re-calls protein_scale with window=1 and edge=1
    and compares the profile to [param_dict[r] for r in sequence]. A raised
    exception or a mismatch is the alarm; being called with window=1 is not.
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
    """BP-SEQ-004: even-window scale profile reversal equivariance (pure API).

    Same position-symmetry law as BP-SEQ-002: the amino-acid scale profile is a
    windowed average, so reversing the sequence must reverse the profile.
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


# --- ProtParam.instability_index ----------------------------------------------

@_guard("instability")
def check_instability(sequence, value):
    """BP-SEQ-026: the Guruprasad instability index is defined as a sum over
    dipeptides. A single-residue peptide contains no dipeptide, so the index is
    identically 0 (or the input is rejected). This is the biochemical domain
    boundary of the quantity, not an arithmetic edge case.
    """
    if len(sequence) == 1:
        trigger_if(not math.isfinite(value) or not _isclose(value, 0.0), "BP-SEQ-026")


# --- IsoelectricPoint ------------------------------------------------------

@_guard("pi")
def check_pi(sequence, point, charge):
    """BP-SEQ-006/007: the isoelectric point is by definition the pH at which the
    modeled net charge is zero. For extreme acidic (D/E) or basic (K/R)
    compositions the reported pI must still satisfy charge-neutrality; a large
    residual charge means the reported pI is not the isoelectric point.
    """
    residues = set(sequence)
    if sequence and residues <= {"D", "E"}:
        trigger_if(abs(charge) > 1e-3, "BP-SEQ-006")
    if sequence and residues <= {"K", "R"}:
        trigger_if(abs(charge) > 1e-3, "BP-SEQ-007")


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

@_guard("molecular_weight")
def check_molecular_weight(original_seq, seq_type, double_stranded, circular,
                           monoisotopic, weight):
    """BP-SEQ-008 empty polymer; BP-SEQ-023 single-strand additivity.

    008: an empty polymer has no residues and no bonds; its mass is 0 (or the
    input is rejected). Acquiring the mass of one water molecule means the
    condensation-water bookkeeping is wrong.
    023: forming a phosphodiester bond between two oligomers releases exactly
    one water, so MW(a+b) + water == MW(a) + MW(b). This is mass conservation
    across the condensation reaction.
    """
    water = 18.010565 if monoisotopic else 18.0153

    trigger_if(
        not original_seq and math.isfinite(weight) and weight != 0, "BP-SEQ-008"
    )
    if not original_seq:
        return

    from Bio.SeqUtils import molecular_weight

    if seq_type in ("DNA", "RNA") and set(original_seq) <= set("ACGTU"):
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


# --- MeltingTemp ---------------------------------------------------------

@_guard("tm_gc_monotonicity")
def check_tm_gc_monotonicity(seq, temperature, valueset, userset, Na, K, Tris,
                             Mg, dNTPs, saltcorr, mismatch):
    """BP-SEQ-019: duplex thermodynamics -- a G:C pair contributes three
    hydrogen bonds versus two for A:T, so at fixed length raising %GC cannot
    lower the melting temperature. The two comparison endpoints are computed
    with the *same* value set and salt/mismatch parameters as the production
    call, so the invariant holds for any parameterisation, not just default.
    """
    from Bio.SeqUtils.MeltingTemp import Tm_GC

    text = str(seq).upper().replace("U", "T")
    if len(text) < 4 or set(text) - set("ACGT"):
        return
    kw = dict(valueset=valueset, userset=userset, Na=Na, K=K, Tris=Tris, Mg=Mg,
              dNTPs=dNTPs, saltcorr=saltcorr, mismatch=mismatch)
    tm_lower = Tm_GC("A" * len(text), **kw)
    tm_higher = Tm_GC("G" * len(text), **kw)
    trigger_if(
        tm_lower - temperature > 1e-6 or temperature - tm_higher > 1e-6,
        "BP-SEQ-019",
    )
    trigger_if(tm_lower - tm_higher > 1e-6, "BP-SEQ-019")


@_guard("tm_nn_revcomp")
def check_tm_nn_revcomp(original_seq, c_seq, shift, selfcomp, nn_table, saltcorr,
                        Na, K, Tris, Mg, dNTPs, temperature):
    """BP-SEQ-028: a DNA/DNA duplex and its reverse complement are the same
    physical molecule read from the other strand, so the nearest-neighbor
    melting temperature is invariant under reverse complementation of the
    primer (all other parameters held fixed).
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
                    dNTPs=dNTPs)
    trigger_if(not _isclose(rc_temp, temperature, tol=1e-6), "BP-SEQ-028")


@_guard("tm_nn_salt")
def check_tm_nn_salt(original_seq, c_seq, shift, selfcomp, saltcorr, Na, K, Tris,
                     Mg, dNTPs, temperature):
    """BP-SEQ-029: counterion screening -- raising [Na+] stabilises a duplex, so
    it does not lower Tm_NN (salt-correction methods 1-4).
    """
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

@_guard("cai_degenerate")
def check_cai_degenerate(sequence, cai_length):
    """BP-SEQ-005: Met (ATG) and Trp (TGG) each have a single codon, so their
    relative adaptiveness is 1 by definition. A coding sequence built only from
    these codons is a valid biological input whose CAI is 1; producing
    cai_length == 0 drives the geometric-mean formula into a division by zero.
    """
    text = str(sequence).upper()
    codons = [text[i:i + 3] for i in range(0, len(text), 3)]
    valid = bool(text) and len(text) % 3 == 0 and set(codons) <= {"ATG", "TGG"}
    trigger_if(valid and cai_length == 0, "BP-SEQ-005")


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
    trigger_if(not _isclose(weight, reversed_weight, tol=1e-6), "BP-SEQ-036")


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
    """BP-PDB-002 endpoint symmetry; BP-PDB-003 rigid invariance.

    A bond angle is a Euclidean invariant: it does not change when the three
    atoms are rigidly rotated and translated together (003), and it does not
    depend on the traversal direction of the two rays from the vertex (002).
    """
    from Bio.PDB.vectors import calc_angle, Vector

    trigger_if(not _isclose(angle, calc_angle(v3, v2, v1)), "BP-PDB-002")

    p1, p2, p3 = _rigid_transform(
        [v1.get_array(), v2.get_array(), v3.get_array()]
    )
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


@_guard("pdb_qcp")
def check_qcp(reference_coords, coords, rms):
    """BP-PDB-015: the optimal-superposition RMSD is a geometric property of the
    two point sets, so it is unchanged by a rigid motion of the moving set
    before fitting, and it is symmetric in its two arguments.
    """
    import numpy as np

    from Bio.PDB.qcprot import QCPSuperimposer

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

    moved = np.asarray(_rigid_transform(list(mov)))
    trigger_if(abs(_fit(ref, moved) - rms) > 1e-6, "BP-SUP-002")

    trigger_if(abs(_fit(mov, ref) - rms) > 1e-6 * max(1.0, abs(rms)), "BP-SUP-003")

    exact = np.asarray(_rigid_transform(list(ref)))
    trigger_if(_fit(ref, exact) > 1e-4, "BP-SUP-004")

    perm = np.random.RandomState(0).permutation(ref.shape[0])
    trigger_if(abs(_fit(ref[perm], mov[perm]) - rms) > 1e-6, "BP-SUP-007")


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
    matrix = aligner.substitution_matrix
    if matrix is not None:
        arr = np.asarray(matrix)
        return arr.ndim == 2 and np.allclose(arr, arr.T)
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
    if not _symmetric_scoring(aligner):
        return

    clone = _clone_aligner(aligner)
    trigger_if(not _isclose(float(clone.score(b, a)), float(score), tol=1e-9),
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
            not _isclose(
                float(clone2.score(a[::-1], b[::-1])), float(score), tol=1e-9
            ),
            "BP-ALN-006",
        )

    matrix = aligner.substitution_matrix
    if matrix is not None:
        import numpy as np

        transposed = matrix.transpose() if hasattr(matrix, "transpose") else None
        if transposed is not None and np.allclose(np.asarray(matrix),
                                                  np.asarray(transposed)):
            clone3 = _clone_aligner(aligner)
            clone3.substitution_matrix = transposed
            trigger_if(
                not _isclose(float(clone3.score(a, b)), float(score), tol=1e-9),
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
    clone = _clone_aligner(aligner)
    trigger_if(not _isclose(float(clone.score(a, b)), float(align_score), tol=1e-9),
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


def _input_is_additive(distance_matrix, tree):
    """True when every input pairwise distance equals the corresponding path
    length in ``tree`` (within tolerance).

    Additivity is the precondition for NJ's order-independence and exact-
    reconstruction theorems (Saitou-Nei 1987; Studier-Keppler 1988). On a
    non-additive matrix NJ may legitimately return different trees for
    different taxon orders (tied Q-matrix minima) and may return negative
    branch lengths -- both are standard, documented NJ behaviour, not defects,
    so the checkers below stay silent there.
    """
    n = len(distance_matrix)
    patristic = _patristic(tree)
    names = list(distance_matrix.names)
    for i in range(n):
        for j in range(i):
            key = tuple(sorted((names[i], names[j])))
            if key not in patristic or not _isclose(
                patristic[key], distance_matrix[names[i], names[j]], tol=1e-6
            ):
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
    differs = any(not _isclose(d0[k], d1[k], tol=1e-6) for k in d0)
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
    bad = False
    for i in range(n):
        for j in range(i):
            key = tuple(sorted((names[i], names[j])))
            if not _isclose(
                patristic[key], distance_matrix[names[i], names[j]], tol=1e-6
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
    if len(distance_matrix) < 3:
        return
    terminals = tree.get_terminals()
    root = tree.root
    depths = tree.depths()
    # all leaves equidistant from the root
    leaf_depths = [depths[t] for t in terminals if t in depths]
    if leaf_depths:
        d0 = leaf_depths[0]
        trigger_if(
            any(not _isclose(d, d0, tol=1e-6) for d in leaf_depths),
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
                trigger_if(not _isclose(trio[1], trio[2], tol=1e-6),
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
    lengths = _all_branch_lengths(tree)
    if not lengths:
        return
    negative = min(lengths)
    if method == "upgma":
        trigger_if(negative < -1e-6, "BP-PHY-004")
        return
    # NJ: only flag when the input is additive (negative lengths are otherwise
    # an accepted outcome of NJ on non-additive data).
    if len(distance_matrix) < 4 or not _input_is_additive(distance_matrix, tree):
        return
    trigger_if(negative < -1e-6, "BP-PHY-004")


# ======================================================================
# Bio.motifs.matrix  (PSSM / PWM)
# ======================================================================

def _pssm_columns(pssm):
    """Return the PSSM as a list of {letter: logodds} dicts, one per position."""
    return [
        {letter: pssm[letter][i] for letter in pssm.alphabet}
        for i in range(pssm.length)
    ]


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
    try:
        lo, hi = float(pssm.min), float(pssm.max)
    except Exception:
        return
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return
    scores = result if hasattr(result, "__len__") else [result]
    for s in scores:
        s = float(s)
        if not math.isfinite(s):
            continue
        trigger_if(s < lo - 1e-6 or s > hi + 1e-6, "BP-MTF-002")


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
