class ReconstructionError(RuntimeError):
    """Base class for a failed reconstruction with a declared reason."""


class InputContractError(ReconstructionError):
    """Input data or model violates the declared experiment contract."""


class ProfileRankDeficient(ReconstructionError):
    """K8 noise profiles do not span the required hidden-node powers."""


class InconsistentObservation(ReconstructionError):
    """Observations cannot be reconciled with the exact theorem class."""


class PrecisionFailure(ReconstructionError):
    """Finite precision or noise prevents a stable exact-class inverse."""


class UnidentifiableNode(ReconstructionError):
    """A node is disconnected from the observed root or receives no signal."""


class TheoremBoundaryFailure(ReconstructionError):
    """The supplied network falls outside the positive reciprocal class."""

