from __future__ import annotations

from numbers import Integral

import numpy as np

from .types import GroupArray


class ParameterValidator:
    """
    Runtime checks that range comparisons cannot express.

    Non-integers pass integer range checks, so they would otherwise only fail
    once samples are partitioned.
    """

    @staticmethod
    def require_integers(**values: object) -> None:
        """
        Check that values are integers, or None for optional parameters.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is neither None nor an integer; booleans are rejected.
        """
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {value!r}")

    @staticmethod
    def require_group_identifiers(groups: GroupArray) -> None:
        """
        Check that group identifiers are discrete.

        Parameters
        ----------
        groups : GroupArray
            The group identifier of each sample.

        Raises
        ------
        TypeError
            If groups does not contain integer or string identifiers.
        """
        is_discrete: bool = any(
            np.issubdtype(groups.dtype, group_type)
            for group_type in (np.integer, np.str_)
        )
        if not is_discrete:
            raise TypeError(
                f"groups must contain integer or string identifiers, got dtype {groups.dtype}"
            )
