from camera import Camera
from plane import Plane
from sphere import Sphere
from light import Light
from dataclasses import dataclass
import numpy as np


@dataclass
class RayTracingSphere:
    sphere: Sphere
    camera: Camera
    plane: Plane
    light: Light

    def __normalize(self, v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    def trace_ray(self, height: int, width: int) -> np.ndarray:
        image = np.full((height, width, 3), [240, 240, 240], dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                u = (x + 0.5) / width - 0.5
                v = 0.5 - (y + 0.5) / height
                
                ray_direction = self.__normalize(
                    self.camera.forward +
                    u * self.camera.viewport_width * self.camera.right +
                    v * self.camera.viewport_height * self.camera.real_up
                )
                
                t_sphere = self.sphere.intersect(ray_direction, self.camera.position)
                t_plane = self.plane.intersect(ray_direction, self.camera.position)
                
                t_min = float('inf')
                hit_object = None
                
                if t_sphere is not None and abs(t_sphere) < abs(t_min):
                    t_min = t_sphere
                    hit_object = self.sphere
                
                if t_plane is not None and abs(t_plane) < abs(t_min):
                    t_min = t_plane
                    hit_object = self.plane
                
                if hit_object is not None:
                    image[y, x] = hit_object.color
        
        return image