"""Runtime traditional-SWE-invariant trigger collection for the SciBench
reference-group bank (SANITIZER.md section 12).

This module is private and inactive unless ``SCIBENCH_TRADITIONAL_LOG`` names
an output file. It is a sibling of ``Bio._scientific_checkers`` and follows
the identical instrumentation discipline (non-disruptive, log-only, never
changes a return value/exception/precision when disabled) -- the only
difference is what each invariant is allowed to appeal to.

Scope rule (SANITIZER.md 12.2): every check here is a **generic
software-correctness property** -- finiteness, an unguarded division, a
shape/length identity, or an index/bounds/exception-safety property a
competent engineer would check without any domain knowledge. No check here
cites a physical, chemical, geometric, or thermodynamic law; that is the
separate scientific bank in ``Bio._scientific_checkers``.

This bank was designed independently of the scientific bank: each candidate
was found by reading production source directly (SANITIZER.md 12.3), not by
looking for a traditional counterpart of an existing scientific sanitizer id.
"""

import math
import os
import threading
import json

_active = threading.local()


def enabled():
    """Return True when the pilot evaluator requested traditional trigger collection."""
    return bool(os.environ.get("SCIBENCH_TRADITIONAL_LOG"))


def trigger(checker_id):
    """Atomically append one checker ID to the configured JSON-lines log."""
    path = os.environ.get("SCIBENCH_TRADITIONAL_LOG")
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


def _all_finite(values):
    return all(math.isfinite(v) for v in values)


def _guard(checker_id):
    """Wrap a checker body so an internal error never disturbs production.

    A checker that itself errors is a curator bug, not a real alarm; it must
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


# --- ProtParam.instability_index -------------------------------------------

@_guard("instability_index_length_division")
def check_instability_index_length(length):
    """BP-SWE-001: ``instability = (10.0/self.length) * score`` divides by the
    sequence length with no precondition that it is nonzero. A zero-length
    ``ProteinAnalysis`` is a valid call through the public constructor;
    observed immediately before the division so the check still runs even
    though the division itself would raise ``ZeroDivisionError``.
    """
    trigger_if(length == 0, "BP-SWE-001")


# --- ProtParam.protein_scale -------------------------------------------------

@_guard("protein_scale_output_shape")
def check_protein_scale_shape(length, window, scores):
    """BP-SWE-003: the output profile length must equal exactly
    ``max(0, length - window + 1)`` for any window >= 1 -- a length
    bookkeeping identity of the sliding-window loop, independent of what the
    per-position values mean.
    """
    if window < 1:
        return
    expected = max(0, length - window + 1)
    trigger_if(len(scores) != expected, "BP-SWE-003")


# --- IsoelectricPoint.pi (bisection search) ---------------------------------

@_guard("pi_bisection_bracket_bounds")
def check_pi_bracket_bounds(min_, max_, result):
    """BP-SWE-004: a bisection search must return a value inside its own
    initial bracket ``[min_, max_]`` -- a bounds-validity property of the
    search algorithm itself, true for any monotonic-ish objective, not a
    claim about what the correct isoelectric point is.
    """
    trigger_if(not (min_ - 1e-9 <= result <= max_ + 1e-9), "BP-SWE-004")


# --- IsoelectricPoint.charge_at_pH -------------------------------------------

@_guard("charge_at_ph_overflow")
def check_charge_finite(pH, charge):
    """BP-SWE-005: ``10 ** (pH - pK)`` can overflow to ``inf`` for extreme pH
    values passed directly to the public ``charge_at_pH`` API (not gated by
    ``pi()``'s bounded bisection). The resulting partial charge would then
    silently collapse to 0 rather than raise or stay a well-defined float.
    Checks only that the returned charge is finite -- no claim about the
    correct titration curve.
    """
    trigger_if(not math.isfinite(charge), "BP-SWE-005")


# --- SeqUtils.GC_skew --------------------------------------------------------

@_guard("gc_skew_output_finite")
def check_gc_skew_finite(values):
    """BP-SWE-007: every windowed skew value must be a finite float. The
    function's own zero-GC-window case is handled by an explicit
    ``ZeroDivisionError`` catch that substitutes 0.0; this check only
    confirms no window ever leaks a NaN/Inf, independent of what skew means.
    """
    trigger_if(not _all_finite(values), "BP-SWE-007")


# --- SeqUtils.CodonAdaptationIndex.calculate ---------------------------------

@_guard("cai_zero_length_division")
def check_cai_length_division(cai_length):
    """BP-SWE-008: ``exp(cai_value / cai_length)`` divides by the count of
    codons that contributed a log-odds term. A coding sequence built
    entirely from ATG/TGG/stop codons is a documented-valid input (the
    docstring only requires "a valid DNA sequence for the provided sequence")
    yet drives ``cai_length`` to 0, which would raise an uncaught
    ``ZeroDivisionError``. Observed immediately before the division.
    """
    trigger_if(cai_length == 0, "BP-SWE-008")


# --- SeqUtils.MeltingTemp.Tm_GC ----------------------------------------------

@_guard("tm_gc_empty_sequence_division")
def check_tm_gc_length_division(seq_len):
    """BP-SWE-010/011: ``Tm_GC`` divides by ``len(seq)`` twice (the ``C/N``
    term and, when ``mismatch=True``, the percent-mismatch term) with no
    precondition that ``seq`` is non-empty; ``_check``/``strict`` do not
    reject the empty string. Observed immediately before the first division.
    """
    trigger_if(seq_len == 0, "BP-SWE-010")


# --- SeqUtils.MeltingTemp.salt_correction ------------------------------------

@_guard("salt_correction_unit_length_division")
def check_salt_correction_unit_length(method, seq_len):
    """BP-SWE-012: method 7's final expression divides by
    ``2.0 * (len(seq) - 1)``. A one-nucleotide ``seq`` is not rejected by any
    precondition in the function and drives this denominator to zero.
    Observed immediately before the division.
    """
    trigger_if(method == 7 and seq_len == 1, "BP-SWE-012")


# --- SeqUtils.MeltingTemp.Tm_NN -----------------------------------------------

@_guard("tm_nn_log_domain")
def check_tm_nn_log_domain(dnac1, dnac2, selfcomp):
    """BP-SWE-013: ``k = (dnac1 - dnac2/2) * 1e-9`` (or ``k = dnac1 * 1e-9``
    when ``selfcomp``) feeds directly into ``math.log(k)``. Neither argument
    is validated against the other, so any ``dnac1 <= dnac2/2`` (ordinary
    user-supplied concentrations, not an extreme or malformed input) makes
    ``k <= 0`` and ``math.log`` raise. Observed immediately before the call.
    """
    if selfcomp:
        trigger_if(dnac1 <= 0, "BP-SWE-013")
    else:
        trigger_if(dnac1 <= dnac2 / 2.0, "BP-SWE-013")


@_guard("tm_nn_denominator_finite")
def check_tm_nn_result_finite(melting_temp):
    """BP-SWE-014: ``melting_temp = (1000*delta_h) / (delta_s + R*log(k))``
    and, for saltcorr in (6, 7), a further ``1 / (1/(melting_temp+273.15) +
    corr)`` -- both denominators are sums of independently varying terms with
    no stated nonzero guarantee. Checks only that the final result is finite.
    """
    trigger_if(not math.isfinite(melting_temp), "BP-SWE-014")


# --- PDB.vectors.Vector.angle ------------------------------------------------

@_guard("vector_angle_zero_length_division")
def check_vector_angle_zero_length(n1, n2):
    """BP-SWE-016: ``c = (self*other) / (n1*n2)`` divides by the product of
    the two vector norms with no guard. ``Vector(0, 0, 0)`` is a valid
    construction that reaches this public method through ``calc_angle``.
    Observed immediately before the division.
    """
    trigger_if(n1 == 0 or n2 == 0, "BP-SWE-016")


@_guard("angle_dihedral_return_contract")
def check_angle_return_range(kind, angle):
    """BP-SWE-017: ``calc_dihedral``'s own docstring states its return value
    lies in ``]-pi, pi]``; ``calc_angle`` returns a value in ``[0, pi]``.
    Checks the function's own stated numeric contract, not a physical claim
    about dihedral geometry.
    """
    if not math.isfinite(angle):
        trigger("BP-SWE-017")
        return
    if kind == "dihedral":
        trigger_if(not (-math.pi < angle <= math.pi + 1e-9), "BP-SWE-017")
    elif kind == "angle":
        trigger_if(not (-1e-9 <= angle <= math.pi + 1e-9), "BP-SWE-017")


# --- PDB.vectors.rotmat -------------------------------------------------------

@_guard("rotmat_output_shape")
def check_rotmat_shape(rot):
    """BP-SWE-018: ``rotmat`` must return a ``(3, 3)`` array -- a shape
    invariant of the function's declared return type, independent of
    whether the matrix is a genuine rotation (that is BP-PDB-011's
    orthogonality claim in the scientific bank).
    """
    shape = getattr(rot, "shape", None)
    trigger_if(shape != (3, 3), "BP-SWE-018")


@_guard("refmat_normalize_overflow")
def check_refmat_normalize_finite(components):
    """BP-SWE-019: ``Vector.normalize`` divides by ``self.norm()`` whenever
    it is truthy, with no lower bound; a vector with a tiny nonzero norm can
    produce ``inf`` components without raising. Checks the normalized
    components stay finite.
    """
    trigger_if(not _all_finite(components), "BP-SWE-019")


# --- PDB.Superimposer ---------------------------------------------------------

@_guard("superimposer_output_shape")
def check_superimposer_shape(rot, tran):
    """BP-SWE-020: ``Superimposer.apply`` unpacks ``self.rotran`` as
    ``rot, tran``; the class's own contract requires ``rot`` to be ``(3, 3)``
    and ``tran`` to be ``(3,)`` for ``Atom.transform`` to accept them.
    """
    rot_shape = getattr(rot, "shape", None)
    tran_shape = getattr(tran, "shape", None)
    trigger_if(rot_shape != (3, 3) or tran_shape != (3,), "BP-SWE-020")


@_guard("superimposer_rms_finite")
def check_superimposer_rms_finite(rms):
    """BP-SWE-021: ``self.rms`` must be a finite float -- a generic
    numeric-validity guard, no claim about RMSD correctness."""
    trigger_if(not math.isfinite(rms), "BP-SWE-021")


# --- PDB.qcprot.QCPSuperimposer -----------------------------------------------

@_guard("qcp_empty_input_nan")
def check_qcp_natoms(natoms, rms):
    """BP-SWE-022: ``set()`` accepts a 0-atom coordinate array with no
    precondition check; ``run()``'s ``np.mean(coords, axis=0)`` on an empty
    array produces NaN, which then silently propagates into ``self.rms``.
    """
    trigger_if(natoms == 0 and not math.isfinite(rms), "BP-SWE-022")


@_guard("qcp_rms_finite")
def check_qcp_rms_finite(natoms, rms):
    """BP-SWE-023: for a non-empty input, ``self.rms`` must still be a
    finite float -- same generic property as BP-SWE-021, independent code
    path (the QCP algorithm rather than classical SVD)."""
    trigger_if(natoms > 0 and not math.isfinite(rms), "BP-SWE-023")


# --- Align.PairwiseAligner -----------------------------------------------------

@_guard("aligner_score_finite")
def check_aligner_score_finite(result):
    """BP-SWE-024: the returned alignment score must be finite for any pair
    of sequences accepted by the input-conversion code -- pure numeric
    validity of the C scoring routine's output, no alignment-semantics
    content."""
    trigger_if(not math.isfinite(result), "BP-SWE-024")


# --- Phylo.TreeConstruction ----------------------------------------------------

@_guard("tree_leaf_count_identity")
def check_tree_leaf_count(names, tree, method):
    """BP-SWE-027/028: the constructed tree must have exactly
    ``len(names)`` terminal (leaf) clades -- one output leaf per input
    taxon, a discrete-structure identity independent of branch-length or
    ultrametricity content (the scientific bank's territory).
    """
    leaves = list(tree.get_terminals())
    trigger_if(len(leaves) != len(names), "BP-SWE-027")


@_guard("tree_leaf_set_identity")
def check_tree_leaf_names(names, tree):
    """BP-SWE-029: the output tree's leaf names must be exactly the input
    taxon names, with no duplicates and no omissions -- a set-equality
    bookkeeping property of the clade-merging loop.
    """
    leaves = list(tree.get_terminals())
    leaf_names = [leaf.name for leaf in leaves]
    trigger_if(
        len(leaf_names) != len(set(leaf_names)) or set(leaf_names) != set(names),
        "BP-SWE-029",
    )


@_guard("upgma_intermediate_matrix_finite")
def check_upgma_matrix_finite(distance_matrix):
    """BP-SWE-030: the working distance matrix must never contain NaN/Inf
    after an averaging update step -- a generic numeric-validity check on
    intermediate state, independent of tree quality.
    """
    for row in distance_matrix.matrix:
        if not _all_finite(row):
            trigger("BP-SWE-030")
            return


# --- motifs.matrix.PositionSpecificScoringMatrix.calculate ---------------------

@_guard("pssm_short_sequence_bounds")
def check_pssm_short_sequence(n, m):
    """BP-SWE-031: ``scores = np.empty(n - m + 1, np.float32)`` -- when the
    input sequence is shorter than the motif (``n < m``), the requested size
    is non-positive. Observed immediately before the allocation: a
    documented "sequence" input class includes sequences shorter than the
    motif, since ``calculate`` states no minimum-length precondition.
    """
    trigger_if(n < m, "BP-SWE-031")


@_guard("pssm_output_shape")
def check_pssm_output_shape(n, m, result):
    """BP-SWE-032: the output length must equal exactly ``n - m + 1`` --
    an exact shape identity between input length and output length, no
    scoring-value content.
    """
    if n < m:
        return
    expected = n - m + 1
    try:
        actual = len(result)
    except TypeError:
        actual = 1
    trigger_if(actual != expected, "BP-SWE-032")


@_guard("pssm_output_finite")
def check_pssm_output_finite(n, m, sequence_is_acgt, result):
    """BP-SWE-033: for a sequence containing only A/C/G/T (the module's own
    comment documents that ambiguous bases are intentionally scored as NaN,
    so they are excluded from this precondition), every returned score must
    be finite. ``np.empty`` does not zero-initialize its buffer, so an
    unwritten element (e.g. from a future early-exit bug in the C routine)
    would otherwise leak uninitialized memory rather than a documented NaN;
    this guards memory-initialization discipline, not scoring correctness.
    """
    if n < m or not sequence_is_acgt:
        return
    try:
        values = list(result)
    except TypeError:
        values = [result]
    trigger_if(not _all_finite(values), "BP-SWE-033")
