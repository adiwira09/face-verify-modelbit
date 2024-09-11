from ultralytics import SAM
from PIL import Image
import requests

model = SAM("sam2_b.pt")


def sam2_for_object_detection(url: str, x_coord: int, y_coord: int):
  img = Image.open(requests.get(url, stream=True, timeout=15).raw)
  results = model(img, points=[x_coord, y_coord], labels=[1])
  return results[0].masks.xy[0][0]
