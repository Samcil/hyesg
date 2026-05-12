"""2D parametric surface for volatility surfaces."""

from __future__ import annotations

from collections.abc import Callable

from hyesg.math.curves.protocol import ParametricCurve


class ParametricSurface:
    """A surface S(x, y) built from a family of curves indexed by y.

    Used for volatility surfaces where x=strike, y=maturity.
    Interpolates linearly in y between the defined curves.

    Args:
        curve_factory: Callable(y) -> ParametricCurve that creates a
            curve for each y value.
        y_values: The y coordinates where curves are defined.
    """

    def __init__(
        self,
        curve_factory: Callable[[float], ParametricCurve],
        y_values: tuple[float, ...],
    ) -> None:
        self._y_values = tuple(sorted(y_values))
        self._curves = {y: curve_factory(y) for y in self._y_values}

    def evaluate(self, x: float, y: float) -> float:
        """Evaluate surface at (x, y) with linear interpolation in y.

        Args:
            x: First coordinate (e.g. strike).
            y: Second coordinate (e.g. maturity).

        Returns:
            Interpolated surface value at (x, y).
        """
        if y <= self._y_values[0]:
            return self._curves[self._y_values[0]].evaluate(x)
        if y >= self._y_values[-1]:
            return self._curves[self._y_values[-1]].evaluate(x)
        for i in range(len(self._y_values) - 1):
            if self._y_values[i] <= y <= self._y_values[i + 1]:
                y0, y1 = self._y_values[i], self._y_values[i + 1]
                w = (y - y0) / (y1 - y0)
                v0 = self._curves[y0].evaluate(x)
                v1 = self._curves[y1].evaluate(x)
                return (1.0 - w) * v0 + w * v1
        return self._curves[self._y_values[-1]].evaluate(x)

    def slice_at_y(self, y: float) -> ParametricCurve:
        """Extract curve at an exact defined y value.

        Args:
            y: The y coordinate to extract.

        Returns:
            The curve at the given y value.

        Raises:
            KeyError: If y is not an exact defined y value.
        """
        if y in self._curves:
            return self._curves[y]
        raise KeyError(
            f"No curve at y={y}. Available: {list(self._y_values)}"
        )
