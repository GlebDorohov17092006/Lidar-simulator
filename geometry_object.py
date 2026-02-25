import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    normal: np.ndarray
    color: np.ndarray


class GeometryObject:
    def __init__(self, triangles: list[Triangle]) -> None:
        self.triangles = triangles

    def intersect(
        self,
        guiding_vector: np.ndarray[np.float64],
        start_vector: np.ndarray[np.float64]
    ) -> Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        
        t_min = float('inf')
        epsilon = 1e-6
        hit_tri = None
        
        for tri in self.triangles:
            if np.sum(guiding_vector * tri.normal) > 0:
                continue
                
            edge1 = tri.v1 - tri.v0
            edge2 = tri.v2 - tri.v0
            
            pvec = np.cross(guiding_vector, edge2)
            det = np.sum(edge1 * pvec)
            
            if abs(det) < epsilon:
                continue
            
            inv_det = 1.0 / det
            tvec = start_vector - tri.v0
            
            u = np.sum(tvec * pvec) * inv_det
            if u < 0.0 or u > 1.0:
                continue
            
            qvec = np.cross(tvec, edge1)
            v = np.sum(guiding_vector * qvec) * inv_det
            if v < 0.0 or u + v > 1.0:
                continue
            
            t = np.sum(edge2 * qvec) * inv_det
    
            if abs(t) < abs(t_min):
                t_min = t
                hit_tri = tri

        if hit_tri is not None:
            hit_point = start_vector + guiding_vector * t_min
            return (t_min, hit_point, hit_tri.normal, hit_tri.color)
        return None