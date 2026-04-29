from abc import ABC, abstractmethod

from algebra.basis_subset import MatrixEntryMask
from algebra import MatrixStandardBasis


class LocalSymmetry(ABC):
    """
    Corresponds to a MatrixEntryMask for the Choi matrix of a d -> d channel having this symmetry.
    """

    @abstractmethod
    def choi_state_mask(self, din: int) -> MatrixEntryMask:
        """
        Returns a MatrixEntryMask for a d -> d channel having this symmetry.
        The mask operates on the Choi matrix as a d^2 x d^2 matrix.
        """
        pass

    def choi_state_bipartite_mask(self, din: int) -> MatrixEntryMask:
        """
        Returns a MatrixEntryMask for a d -> d channel having this symmetry.
        The mask operates on the Choi matrix as a d x d x d x d tensor, with index convention
        (i_in, i_out, j_in, j_out).
        """
        state_mask_obj = self.choi_state_mask(din)
        pairs = state_mask_obj.index_pairs
        channel_pairs = [(i // din, i % din, j // din, j % din) for (i, j) in pairs]
        
        return MatrixEntryMask((din, din, din, din), channel_pairs)

    @classmethod
    def from_string(cls, symmetry_string: str):
        # This is here for backwards compatibility, in the end everything should just use the class and not strings.
        if symmetry_string.lower() == "cldiu":
            return CLDUISymmetry()
        elif symmetry_string.lower() == "ldoi":
            return LDOISymmetry()
        elif symmetry_string.lower() == "ldui":
            return LDUISymmetry()
        else:
            raise ValueError(f"Unknown symmetry string: {symmetry_string}")

class NoSymmetry(LocalSymmetry):
    def choi_state_mask(self, din: int) -> MatrixEntryMask:
        return MatrixEntryMask((din ** 2, din ** 2), set(MatrixStandardBasis(din ** 2).iterate_labels()))

class CLDUISymmetry(LocalSymmetry):
    def choi_state_mask(self, din: int) -> MatrixEntryMask:
        allowed_pairs = set()
        if din == 2:
            allowed_pairs.update({
                (0, 0), (1, 1), (2, 2), (3, 3),
                (0, 3), (3, 0)
            })
        elif din == 3:
            allowed_pairs.update({
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                (0, 4), (4, 0),
                (0, 8), (8, 0),
                (4, 8), (8, 4)
            })
        else:
            raise NotImplementedError(f"CLDUISymmetry state_mask is only implemented for d=2 or 3, got d={din}.")
            
        return MatrixEntryMask((din ** 2, din ** 2), allowed_pairs, name=f"CLDUI({din})")


class LDOISymmetry(LocalSymmetry):
    def choi_state_mask(self, din: int) -> MatrixEntryMask:
        allowed_pairs = set()
        if din == 2:
            allowed_pairs.update({
                (0, 0), (1, 1), (2, 2), (3, 3),
                (1, 2), (2, 1),
                (0, 3), (3, 0)
            })
        elif din == 3:
            allowed_pairs.update({
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                (1, 3), (3, 1),
                (2, 6), (6, 2),
                (5, 7), (7, 5),
                (0, 4), (4, 0),
                (0, 8), (8, 0),
                (4, 8), (8, 4)
            })
        else:
            raise NotImplementedError(f"LDOISymmetry state_mask is only implemented for d=2 or 3, got d={din}.")
            
        return MatrixEntryMask((din ** 2, din ** 2), allowed_pairs, name=f"LDOI({din})")


class LDUISymmetry(LocalSymmetry):
    def choi_state_mask(self, din: int) -> MatrixEntryMask:
        allowed_pairs = set()
        if din == 2:
            allowed_pairs.update({
                (0, 0), (1, 1), (2, 2), (3, 3),
                (1, 2), (2, 1)
            })
        elif din == 3:
            allowed_pairs.update({
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                (1, 3), (3, 1),  
                (2, 6), (6, 2),  
                (5, 7), (7, 5)   
            })
        else:
            raise NotImplementedError(f"LDUISymmetry state_mask is only implemented for d=2 or 3, got d={din}.")
            
        return MatrixEntryMask((din ** 2, din ** 2), allowed_pairs, name=f"LDUI({din})")
