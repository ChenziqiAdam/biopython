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


def _magnitudes_ok(*values, limit=1e100):
    """True when every given number/array is finite and at most ``limit`` in
    magnitude (None is ignored). Beyond this, products and sums of squares
    overflow float64 -- a floating-point limit, not a missing guard."""
    import numpy as np

    for value in values:
        if value is None:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.size and not (np.all(np.isfinite(arr)) and np.max(np.abs(arr)) <= limit):
            return False
    return True


def _aligner_params_ok(aligner, limit=1e100):
    """True when every numeric scoring parameter of a PairwiseAligner /
    CodonAligner (and its substitution matrix) is finite and at most
    ``limit`` in magnitude."""
    for name in dir(aligner):
        if name.startswith("_") or "score" not in name:
            continue
        try:
            value = getattr(aligner, name)
        except Exception:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not _magnitudes_ok(value, limit=limit):
            return False
    matrix = getattr(aligner, "substitution_matrix", None)
    if matrix is not None and not _magnitudes_ok(matrix, limit=limit):
        return False
    return True


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
    correct titration curve. A pH outside [-250, 250] (physical pH is about
    0-14) overflows ``10 ** (pH - pK)`` in float64 by construction, a
    floating-point limit rather than a missing guard, and is excluded.
    """
    if not (math.isfinite(pH) and abs(pH) <= 250.0):
        return
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
def check_tm_nn_result_finite(melting_temp, delta_h=0.0, delta_s=0.0):
    """BP-SWE-014: ``melting_temp = (1000*delta_h) / (delta_s + R*log(k))``
    and, for saltcorr in (6, 7), a further ``1 / (1/(melting_temp+273.15) +
    corr)`` -- both denominators are sums of independently varying terms with
    no stated nonzero guarantee. Checks only that the final result is finite.
    A non-finite or astronomically large (|value| > 1e100) enthalpy or
    entropy is a floating-point limit and is excluded.
    """
    if not _magnitudes_ok(delta_h, delta_s):
        return
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
def check_superimposer_rms_finite(rms, fixed=None, moving=None):
    """BP-SWE-021: ``self.rms`` must be a finite float -- a generic
    numeric-validity guard, no claim about RMSD correctness. Coordinates
    larger than 1e100 in magnitude overflow the sum of squares: a
    floating-point limit, excluded."""
    if not _magnitudes_ok(fixed, moving):
        return
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
def check_qcp_rms_finite(natoms, rms, coords=None, coords_ref=None):
    """BP-SWE-023: for a non-empty input, ``self.rms`` must still be a
    finite float -- same generic property as BP-SWE-021, independent code
    path (the QCP algorithm rather than classical SVD). The characteristic
    polynomial is degree 8 in the coordinates, so coordinates above 1e30 in
    magnitude overflow float64: a floating-point limit, excluded."""
    if not _magnitudes_ok(coords, coords_ref, limit=1e30):
        return
    trigger_if(natoms > 0 and not math.isfinite(rms), "BP-SWE-023")


# --- Align.PairwiseAligner -----------------------------------------------------

@_guard("aligner_score_finite")
def check_aligner_score_finite(aligner, result):
    """BP-SWE-024: the returned alignment score must be finite for any pair
    of sequences accepted by the input-conversion code -- pure numeric
    validity of the C scoring routine's output, no alignment-semantics
    content. Scoring parameters that are infinite (a documented way to
    forbid gaps) or above 1e100 in magnitude make an infinite score
    legitimate or overflow float64, and are excluded."""
    if not _aligner_params_ok(aligner):
        return
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
def check_upgma_matrix_finite(distance_matrix, original=None):
    """BP-SWE-030: the working distance matrix must never contain NaN/Inf
    after an averaging update step -- a generic numeric-validity check on
    intermediate state, independent of tree quality.
    """
    rows = distance_matrix.matrix
    # Input distances that are non-finite or above 1e100 overflow when
    # averaged: a floating-point limit.
    source = original if original is not None else distance_matrix
    if not _magnitudes_ok(*source.matrix):
        return
    for row in rows:
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
    ``n == m - 1`` requests an empty array, which is valid (the result is an
    empty score array); only ``n < m - 1`` asks for a negative size.
    """
    trigger_if(n < m - 1, "BP-SWE-031")


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
    ``sequence_is_acgt`` also requires every log-odds entry of the PSSM to be
    finite: with zero pseudocounts a never-seen letter has log-odds -inf, so
    a -inf score is the correct value, not a defect.
    """
    if n < m or not sequence_is_acgt:
        return
    try:
        values = list(result)
    except TypeError:
        values = [result]
    trigger_if(not _all_finite(values), "BP-SWE-033")


# --- Align.Alignment.shape / indices ------------------------------------------

@_guard("alignment_shape_identity")
def check_alignment_shape(n, m, shape):
    """BP-SWE-034: ``Alignment.shape`` must equal exactly
    ``(len(alignment), alignment.length)`` -- a definitional identity between
    two independently computed properties (``__len__`` reads
    ``len(self.coordinates)``; ``length`` walks the coordinate steps), no
    alignment-semantics content.
    """
    trigger_if(tuple(shape) != (n, m), "BP-SWE-034")


@_guard("alignment_indices_shape_identity")
def check_alignment_indices_shape(indices_shape, alignment_shape):
    """BP-SWE-035: ``Alignment.indices`` is documented to return an array
    with "the same number of rows and columns as the alignment, as given by
    self.shape" -- a shape identity stated in the property's own docstring.
    """
    trigger_if(tuple(indices_shape) != tuple(alignment_shape), "BP-SWE-035")


# --- SeqUtils.GC123 -----------------------------------------------------------

@_guard("gc123_zero_length_division")
def check_gc123_length_division(nall):
    """BP-SWE-037: ``gcall = 100.0 * gcall / nall`` divides by the total
    A/T/G/C count across all three codon positions with no guard, unlike the
    per-position ``gc[i]`` computation three lines above which *is* wrapped
    in a ``try/except``. An empty (or all-ambiguous) sequence is a
    documented-valid input -- ``GC123``'s own docstring only warns it "does
    NOT deal with ambiguous nucleotides" correctly, not that it rejects
    them -- and drives this specific denominator to zero while the
    per-position one is already caught. Observed immediately before the
    division.
    """
    trigger_if(nall == 0, "BP-SWE-037")


# --- Align.CodonAligner.score -------------------------------------------------

@_guard("codon_aligner_score_finite")
def check_codon_aligner_score_finite(aligner, result):
    """BP-SWE-036: the returned codon-alignment score must be finite for any
    pair of sequences accepted by the input-conversion code -- the same
    generic numeric-validity property as BP-SWE-024, on the independent
    ``CodonAligner`` C extension rather than ``PairwiseAligner``. Infinite or
    >1e100 scoring parameters are excluded as for BP-SWE-024.
    """
    if not _aligner_params_ok(aligner):
        return
    trigger_if(not math.isfinite(result), "BP-SWE-036")


# --- motifs.matrix.PositionSpecificScoringMatrix.max/min ----------------------

@_guard("pssm_max_min_order")
def check_pssm_max_min_order(max_value, min_value):
    """BP-SWE-038: ``max()`` sums the per-position maximum log-odds value and
    ``min()`` sums the per-position minimum; since max-per-position is always
    >= min-per-position, the summed ``max()`` must be >= the summed
    ``min()`` -- an order identity between two independently implemented
    methods, no claim about what a "good" motif score is.
    """
    trigger_if(max_value < min_value, "BP-SWE-038")


# --- Phylo.TreeConstruction.DistanceCalculator._pairwise ----------------------

@_guard("distance_calculator_pairwise_range")
def check_pairwise_distance_range(distance):
    """BP-SWE-039: ``_pairwise``'s own docstring states it "Returns a value
    between 0 (identical sequences) and 1 (completely different, or seq1 is
    an empty string)" -- checking the method's own stated numeric contract,
    not a claim about what the correct evolutionary distance is.
    """
    trigger_if(not (-1e-9 <= distance <= 1 + 1e-9), "BP-SWE-039")


# --- ProtParam.gravy -----------------------------------------------------------

@_guard("gravy_length_division")
def check_gravy_length(length):
    """BP-SWE-040: ``value = total_gravy / self.length`` divides by the
    sequence length with no precondition that it is nonzero -- the same
    unguarded-division shape as BP-SWE-001's ``instability_index``, on the
    sibling ``gravy`` method. A zero-length ``ProteinAnalysis`` is a valid
    call through the public constructor.
    """
    trigger_if(length == 0, "BP-SWE-040")


# --- motifs.matrix.PositionSpecificScoringMatrix.dist_pearson/_at -------------

@_guard("pssm_dist_pearson_finite")
def check_dist_pearson_at_denominator(denominator):
    """BP-SWE-041 (pre-division half): ``dist_pearson_at``'s
    ``numerator / denominator`` has no guard against a zero-variance column
    (``denominator == sqrt((sxx - sx*sx) * (syy - sy*sy))``), which occurs
    whenever a PSSM column's per-letter log-odds values are constant across
    the compared positions. When every compared value is finite, this drives
    ``denominator`` to *exactly* 0.0 and the division raises
    ``ZeroDivisionError`` -- observed here, before the division, since a
    checker placed after it would never be reached on this path.
    """
    trigger_if(denominator == 0.0, "BP-SWE-041")


@_guard("pssm_dist_pearson_finite")
def check_dist_pearson_at_finite(value):
    """BP-SWE-041 (post-division half): the other route to the same
    invariant violation -- a column containing -inf log-odds from an
    unobserved letter -- produces a nonzero-but-still-degenerate
    ``denominator`` via inf-arithmetic, so the division itself succeeds but
    silently returns NaN rather than raising. Observed on the result,
    complementing the pre-division check above.
    """
    trigger_if(not math.isfinite(value), "BP-SWE-041")


# --- SeqUtils.MeltingTemp.salt_correction (method 7) ---------------------------

@_guard("salt_correction_log_domain")
def check_salt_correction_log_domain(mg):
    """BP-SWE-042: method 7's final expression calls ``math.log(mg)`` twice
    (and squares it) after the ``if Mon > 0`` branch, but ``mg`` (== ``Mg *
    1e-3``, or the dNTP-corrected free-Mg concentration) is only guarded
    against being zero via the *unrelated* ``mon`` check for methods 1-6;
    method 7 has no such guard on ``mg`` itself. ``Na=K=Tris=Mg=0`` -- all
    default values of the public API -- makes ``mon == 0`` too, but ``mon``
    is never range-checked for method 7 (only for methods 1-6), so
    ``math.log(mg)`` with ``mg == 0`` raises ``ValueError: math domain
    error`` before any of this bank's method-7 length check can even help.
    """
    trigger_if(mg <= 0, "BP-SWE-042")


# --- Align.Alignment.inverse_indices --------------------------------------------

@_guard("alignment_inverse_indices_length_identity")
def check_inverse_indices_lengths(sequence_lengths, array_lengths):
    """BP-SWE-043: ``inverse_indices``'s own docstring states the returned
    list has "the number of arrays ... equal to the number of aligned
    sequences, and the length of each array ... equal to the length of the
    corresponding sequence" -- a definitional length identity against
    ``self.sequences``, independent of ``indices``/``shape`` (BP-SWE-035),
    since ``inverse_indices`` is computed from ``self.coordinates`` and
    ``self.sequences`` directly rather than by inverting ``indices``.
    """
    trigger_if(list(sequence_lengths) != list(array_lengths), "BP-SWE-043")


# --- Align.Alignment.counts (AlignmentCounts) -----------------------------------

@_guard("alignment_counts_identity_mismatch_partition")
def check_counts_identity_mismatch_partition(no_wildcard, identities, mismatches, aligned):
    """BP-SWE-044: ``AlignmentCounts`` (a C extension) is documented to
    report ``identities`` as "the number of identical letters" and
    ``mismatches`` as "the number of mismatched letters", both counted over
    the ``aligned`` letter pairs. When no wildcard character is configured
    (neither passed directly nor set on an aligner argument), every aligned
    pair is by construction either identical or a mismatch, so
    ``identities + mismatches`` must equal ``aligned`` exactly. A configured
    wildcard character is documented to be "ignored in the calculation of
    the number of matches, mismatches" while still counting toward
    ``aligned``, so the identity legitimately does not hold in that case --
    confirmed empirically via ``test_pairwise_aligner.py``'s wildcard tests,
    which is why this precondition excludes it rather than treating it as a
    real defect.
    """
    # ``no_wildcard`` is also False when a sequence is undefined (Seq(None,
    # length)): ``counts()`` deliberately counts such blocks as aligned
    # without identities or mismatches.
    if not no_wildcard:
        return
    trigger_if(identities + mismatches != aligned, "BP-SWE-044")
