# Scientific checkers

This reviewed bank exposes 32 active checkers in 32 families.
Quarantined candidates are not public and are not scored.

Set `SCIBENCH_TRIGGER_LOG` to a writable file and exercise the normal
public API. Direct logger calls and synthetic fault injection are not
valid benchmark triggers.

See `SCIENTIFIC_CHECKERS.json` for each checker’s precondition,
invariant, observation point, and alarm predicate.
