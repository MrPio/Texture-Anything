bl_info = {
    "name": "Texture Anything",
    "author": "Valerio Morelli, Mattia Sbattella, Jacopo Coloccioni",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > TextureAnything",
    "description": "Generates a texture from caption and UV map",
    "category": "Object",
}

import subprocess
import sys
import hashlib
import io
from pathlib import Path
import tempfile
from PIL import Image, ImageOps

import bpy
import os
import threading
import queue
import random

# ensure_pillow_installed()
import time
from PIL import Image, ImageDraw
import bmesh

import requests
from PIL import Image
import io
import base64
from pathlib import Path

def predict(caption, image,seed,steps):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "data": [
           caption,
            {"name": "image.png", "data": base64_image},
            steps,
            seed,
            True
        ]
    }

    response = requests.post("http://localhost:7860/predict", json=payload)

    if response.ok:
        output_base64 = response.json()["data"][0]
        return Image.open(io.BytesIO(base64.b64decode(output_base64)))
    else:
        print("Error:", response.text)


execution_queue = queue.Queue()  # Queue for scheduling functions to run on Blender's main thread

def process_queue():
    # Process all functions in the execution queue
    while not execution_queue.empty():
        fn = execution_queue.get()
        try:
            print("[TextureAnything] Executing function from queue...")
            fn()
        except Exception as e:
            print(f"[TextureAnything] Error in callback: {e}")
    # Return delay in seconds to rerun this timer function
    return 0.5


def unset_loading(scene):
    scene.texture_anything_loading = False
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


def apply_texture(tex_path, context):
    obj = context.active_object

    # Check if the texture file exists
    if not os.path.exists(tex_path):
        print(f"[TextureAnything] File not found: {tex_path}")
        return

    # Flip image vertically, because the drawn UV are vertically flipped
    fd, path = tempfile.mkstemp(suffix=".png")
    Image.open(tex_path).transpose(Image.FLIP_TOP_BOTTOM).save(path)
    os.close(fd)

    # Load the generated image into Blender
    img = bpy.data.images.load(path)

    # Create a new material with nodes enabled
    mat = bpy.data.materials.new(name="AI_Generated_Texture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get the Principled BSDF shader node
    bsdf = nodes.get("Principled BSDF")

    # Create a new Image Texture node and assign the loaded image
    texNode = nodes.new("ShaderNodeTexImage")
    texNode.image = img

    # Connect the Image Texture color output to the Base Color input of the shader
    links.new(bsdf.inputs["Base Color"], texNode.outputs["Color"])

    # Assign the material to the active object, replacing existing or appending
    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Clear loading state flag
    context.scene.texture_anything_loading = False
    print("[TextureAnything] Texture successfully applied.")


def draw_uv(size=512) -> Image.Image | None:
    bm = bmesh.new()
    bm.from_mesh(bpy.context.active_object.data)
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        raise Exception("No UV layers found on the mesh")

    # === Create white transparent image ===
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    def is_outside_uv(vector, threshold=0.15):
        for coord in [vector.x, vector.y]:
            if not -threshold < coord < 1 + threshold:
                return True
        return False

    # === Draw UV edges ===
    for face in bm.faces:
        uv_coords = [loop[uv_layer].uv for loop in face.loops]
        if any(map(is_outside_uv, uv_coords)):
            print("The UV map has negative values")
            return None
        if len(uv_coords) < 2:
            continue
        # Scale and convert UVs to pixel coordinates (flip V axis)
        points = [(int(uv.x * size), int(uv.y * size)) for uv in uv_coords]
        # Close the loop
        points.append(points[0])
        draw.line(points, fill=(0, 0, 0, 255), width=1)

    return img


def register():
    bpy.utils.register_class(OT_generate)
    bpy.utils.register_class(PT_panel)
    # Register properties on the Scene for user input and loading state
    bpy.types.Scene.texture_anything_caption = bpy.props.StringProperty(
        name="Description",
        description="Describe the texture you want",
        default="A yellow plate with flowers motives",
    )
    bpy.types.Scene.texture_anything_seed = bpy.props.IntProperty(
        min=-1,
        name="Seed",
        default=-1,
    )
    bpy.types.Scene.texture_anything_steps = bpy.props.IntProperty(
        min=4,
        max=60,
        name="Generation Steps",
        default=20,
    )
    bpy.types.Scene.texture_anything_loading = bpy.props.BoolProperty(name="Loading", default=False)
    # Register a timer to process the main thread execution queue periodically
    bpy.app.timers.register(process_queue)


def unregister():
    bpy.utils.unregister_class(OT_generate)
    bpy.utils.unregister_class(PT_panel)
    del bpy.types.Scene.texture_anything_caption
    del bpy.types.Scene.texture_anything_seed
    del bpy.types.Scene.texture_anything_steps
    del bpy.types.Scene.texture_anything_loading
    bpy.app.timers.unregister(process_queue)


class OT_generate(bpy.types.Operator):
    bl_idname = "texture_anything.generate"
    bl_label = "Generate Texture"
    bl_description = "Generate a texture from UV map and user description"

    def execute(self, context):
        scene = context.scene
        obj = context.active_object

        # Ensure an active mesh object is selected
        # if not bpy.data.filepath:
        #     self.report({"ERROR"}, "Save the file .blend and retry")
        #     return {"CANCELLED"}
        # Ensure an active mesh object is selected
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object")
            return {"CANCELLED"}

        # User description of desired texture (currently unused in mock)
        caption = scene.texture_anything_caption
        seed= scene.texture_anything_seed
        steps= scene.texture_anything_steps

        # Export the UV layout of the active object
        # uv_path = bpy.path.abspath("C:\\Users\\tiasb\\Desktop\\uv_layout.png")
        timestamp = int(time.time())

        # Base path (relative to the .blend file)
        base_path = bpy.path.abspath("//")

        # Ensure UV and texture directories exist
        uv_dir = os.path.join(base_path, "uv_maps")
        tex_dir = os.path.join(base_path, "texture")

        try:
            os.makedirs(uv_dir, exist_ok=True)
            os.makedirs(tex_dir, exist_ok=True)
        except Exception as e:
            self.report({"ERROR"}, f"Errore nella creazione delle directory: {e}")
            return {"CANCELLED"}

        uv_path = os.path.join(uv_dir, f"uv_{timestamp}.png")
        out_path = os.path.join(tex_dir, f"generated_texture_{timestamp}.png")

        print(f"[TextureAnything] UV path: {uv_path}")
        print(f"[TextureAnything] Output texture path: {out_path}")

        # Export the UV layout
        # bpy.ops.uv.export_layout(filepath=uv_path, size=(1024, 1024))
        condition_image=draw_uv()
        # condition_image.save(uv_path)

        def worker():
            try:
                # Set loading state to show progress in UI
                print("[TextureAnything] Generating texture...")
                # img = Image.new(
                #     "RGB",
                #     (1024, 1024),
                #     color=random.choice(["red", "black", "blue", "green", "yellow", "orange", "purple", "pink"]),
                # )  # Create a black image

                img = predict(caption, condition_image,seed,steps)

                img.save(out_path)  # Save the generated image
                print("[TextureAnything] texture saved at:", out_path)

                # Schedule texture application on the main Blender thread
                execution_queue.put(lambda: apply_texture(out_path, context))
            finally:
                # Always unset loading flag on main thread, even if error occurs
                execution_queue.put(lambda: apply_texture(out_path, context))

        # Start the mock generation in a separate thread to avoid freezing Blender UI
        scene.texture_anything_loading = True
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

        return {"FINISHED"}


class PT_panel(bpy.types.Panel):
    bl_label = "AI Texture Generator"
    bl_idname = "PT_panel"
    bl_space_type = "VIEW_3D"  # The panel is in the 3D Viewport
    bl_region_type = "UI"  # Located in the right sidebar
    bl_category = "Texture Anything"  # Tab category name

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        caption = getattr(scene, "texture_anything_caption", "")
        loading = getattr(scene, "texture_anything_loading")

        layout.label(text="Texture Anything Settings")

        col = layout.column(align=True)
        col.prop(scene, "texture_anything_caption", text="Texture Description")
        col.prop(scene, "texture_anything_seed", text="Seed")
        col.prop(scene, "texture_anything_steps", text="Steps")

        if loading:
            layout.label(text="⏳ Generating...")
        else:
            layout.operator("texture_anything.generate")


if __name__ == "__main__":
    register()
