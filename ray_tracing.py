from camera import Camera
from plane import Plane
from sphere import Sphere
from light import Light
from geometry_object import GeometryObject
from dataclasses import dataclass
import numpy as np


@dataclass
class RayTracingGeometryObject:
    geometry_object: GeometryObject
    camera: Camera
    plane: Plane
    light: Light

    def __normalize(self, v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)
    
    def trace_ray(self, height: int, width: int) -> np.ndarray:
        image = np.full((height, width, 3), [100, 100, 100], dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                u = (x + 0.5) / width - 0.5
                v = 0.5 - (y + 0.5) / height
                
                ray_direction = self.__normalize(
                    self.camera.forward +
                    u * self.camera.viewport_width * self.camera.right +
                    v * self.camera.viewport_height * self.camera.real_up
                )
                
                result = self.geometry_object.intersect(ray_direction, self.camera.position)
                
                if result is None:
                    continue
                
                t, hit_point, normal, color_tri = result
                
                light_dir = self.__normalize(self.light.position - hit_point)
                view_dir = self.__normalize(self.camera.position - hit_point)
                
                diffuse = max(0.1, np.sum(normal * light_dir))
                
                reflect_dir = 2 * np.sum(normal * light_dir) * normal - light_dir
                reflect_dir = self.__normalize(reflect_dir)
                specular = max(0, np.sum(view_dir * reflect_dir)) ** 30
                
                brightness = diffuse + specular*5
                
                color = color_tri * brightness
                color = np.clip(color, 0, 255).astype(np.uint8)
                image[y, x] = color
        
        return image


@dataclass
class RayTracingSphere:
    sphere: Sphere
    camera: Camera
    plane: Plane
    light: Light

    def __normalize(self, v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    def trace_ray(self, height: int, width: int) -> np.ndarray:
        image = np.full((height, width, 3), [100, 100, 100], dtype=np.uint8)
        
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
                
                if t_sphere is None:
                    continue
                
                hit_point = self.camera.position + t_sphere * ray_direction
                normal = self.__normalize(hit_point - self.sphere.center)
                
                light_dir = self.__normalize(self.light.position - hit_point)
                view_dir = self.__normalize(self.camera.position - hit_point)
                
                diffuse = max(0.1, np.sum(normal * light_dir))
                
                reflect_dir = 2 * np.sum(normal * light_dir) * normal - light_dir
                reflect_dir = self.__normalize(reflect_dir)
                specular = max(0, np.sum(view_dir * reflect_dir)) ** 30
                
                brightness = diffuse + specular
                
                color = self.sphere.color * brightness
                color = np.clip(color, 0, 255).astype(np.uint8)
                image[y, x] = color
        
        return image