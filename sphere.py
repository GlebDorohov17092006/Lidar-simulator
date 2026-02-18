import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Sphere:
    radius: float
    center: np.ndarray[np.float64]
    color: np.ndarray[np.uint8]

    def intersect(
        self,
        guiding_vector: np.ndarray[np.float64],
        start_vector: np.ndarray[np.float64]
    ) -> Optional[float]:

        """coefficients of the quadratic equation"""
        a = np.sum(guiding_vector**2)
        b = 2 * np.sum(guiding_vector * (start_vector - self.center))
        c = np.sum((start_vector - self.center)**2) - self.radius**2

        if b**2 - 4 * a * c < 0:
            return None

        t1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        t2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        if abs(t1) < abs(t2):
            return t1
        else:
            return t2
