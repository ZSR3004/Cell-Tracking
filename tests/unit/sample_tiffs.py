import os

TIFF_PATHS = [os.path.join("datasets/", f) for f in os.listdir("datasets/") if (not f.endswith('.md'))]

print(TIFF_PATHS)
