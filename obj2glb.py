import bpy
import sys
import os

argv = sys.argv
argv = argv[argv.index("--") + 1:]

in_obj = os.path.abspath(argv[0])
out_glb = os.path.abspath(argv[1])

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

# Enable old OBJ importer addon for Blender 3.x
try:
    bpy.ops.preferences.addon_enable(module="io_scene_obj")
except Exception as e:
    print("Could not enable io_scene_obj addon:", e)

# Import OBJ using Blender 3.x API
bpy.ops.import_scene.obj(filepath=in_obj)

# Select all imported objects
bpy.ops.object.select_all(action="SELECT")

# Apply transforms
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Export GLB
bpy.ops.export_scene.gltf(
    filepath=out_glb,
    export_format="GLB",
    export_texcoords=True,
    export_materials="EXPORT",
)

print("Exported:", out_glb)