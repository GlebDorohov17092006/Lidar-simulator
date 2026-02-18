import numpy as np
from camera import Camera
from sphere import Sphere
from plane import Plane
from light import Light
from ray_tracing import RayTracingSphere
from PIL import Image

def test_simple():
    camera = Camera(
        position=np.array([0, 1, 5]),
        look_at=np.array([0, 1, -2]),
        vector_up=np.array([0, 1, 0]),
        fov_vertical=45,
        fov_horizontal=60
    )
    
    sphere = Sphere(
        radius=1.2,
        center=np.array([0.5, 1.2, -2]),
        color=np.array([255, 80, 80])
    )
    
    plane = Plane(
        point=np.array([0, -0.5, 0]),
        normal=np.array([0, 1, 0]),
        color=np.array([200, 200, 200])
    )
    
    light = Light(
        position=np.array([3, 5, 2]),
        color=np.array([255, 255, 220]),
        intensity=2.0
    )
    
    rt = RayTracingSphere(
        sphere=sphere,
        camera=camera,
        plane=plane,
        light=light
    )
    
    image = rt.trace_ray(600, 800)
    
    img = Image.fromarray(image)
    img.save('sphere_volume.png')
    print("Сохранено как sphere_volume.png")

if __name__ == "__main__":
    test_simple()