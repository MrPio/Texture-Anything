bl_info = {
    "name": "UV Texture Generator with AI (Threaded)",
    "author": "Sbatt",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > AI Texture",
    "description": "Generates a texture via AI from caption and UV map with external threads",
    "category": "Object",
}

# Uncomment these lines to install required Python packages in Blender's bundled Python:
# "C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe" -m ensurepip
# "C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe" -m pip install pillow

import bpy
import os
import threading
import subprocess
import queue
import random
import time
from PIL import Image  # Pillow library for image creation and manipulation

execution_queue = queue.Queue()  # Queue for scheduling functions to run on Blender's main thread

def run_in_main_thread(func):
    # Put a function into the queue to be executed in Blender's main thread
    execution_queue.put(func)

def process_queue():
    # Process all functions in the execution queue
    while not execution_queue.empty():
        fn = execution_queue.get()
        try:
            print("[AI_Texture] Executing function from queue...")
            fn()
        except Exception as e:
            print(f"[AI_Texture] Error in callback: {e}")
    # Return delay in seconds to rerun this timer function
    return 0.5

def unset_loading(scene):
    scene.ai_texture_loading = False
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

class AITEXTURE_OT_generate(bpy.types.Operator):
    bl_idname = "ai_texture.generate"
    bl_label = "Generate Texture"
    bl_description = "Generate a black texture from UV map and user description"

    def execute(self, context):
        scene = context.scene
        obj = context.active_object

        # Ensure an active mesh object is selected
        if not bpy.data.filepath:
            self.report({'ERROR'}, "Save the file .blend and retry")
            return {'CANCELLED'}
        # Ensure an active mesh object is selected
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object")
            return {'CANCELLED'}

    
        # User description of desired texture (currently unused in mock)
        caption = scene.ai_texture_caption  

        # Export the UV layout of the active object
        #uv_path = bpy.path.abspath("C:\\Users\\tiasb\\Desktop\\uv_layout.png")
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
            self.report({'ERROR'}, f"Errore nella creazione delle directory: {e}")
            return {'CANCELLED'}

        uv_path = os.path.join(uv_dir, f"uv_{timestamp}.png")
        out_path = os.path.join(tex_dir, f"generated_texture_{timestamp}.png")

        print(f"[AI_Texture] UV path: {uv_path}")
        print(f"[AI_Texture] Output texture path: {out_path}")

        # Export the UV layout
        bpy.ops.uv.export_layout(filepath=uv_path, size=(1024, 1024))
        

        def worker():
            try:
                # --- START: MOCK TEXTURE GENERATION ---     <------CHANGE THIS BLOCK
                # Set loading state to show progress in UI
                print("[AI_Texture] Generating fake black texture...")
                time.sleep(3)  # ⏳ Fake simulation delay
                img = Image.new('RGB', (1024, 1024), color=random.choice( ['red', 'black', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']))  # Create a black image
                img.save(out_path)  # Save the generated image
                print("[AI_Texture] Black texture saved at:", out_path)
                # --- END: MOCK TEXTURE GENERATION ---

                # Schedule texture application on the main Blender thread
                run_in_main_thread(lambda: apply_texture(out_path, context))
            except Exception as e:
                print(f"[AI_Texture] Mock generation error: {e}")
            finally:
                # Always unset loading flag on main thread, even if error occurs
                #run_in_main_thread(lambda: setattr(scene, "ai_texture_loading", False))
                run_in_main_thread(lambda: unset_loading(scene))

        # Start the mock generation in a separate thread to avoid freezing Blender UI
        scene.ai_texture_loading = True
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

        return {'FINISHED'}

def apply_texture(tex_path, context):
    obj = context.active_object

    # Check if the texture file exists
    if not os.path.exists(tex_path):
        print(f"[AI_Texture] File not found: {tex_path}")
        return

    # Load the generated image into Blender
    img = bpy.data.images.load(tex_path)

    # Create a new material with nodes enabled
    mat = bpy.data.materials.new(name="AI_Generated_Texture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get the Principled BSDF shader node
    bsdf = nodes.get("Principled BSDF")

    # Create a new Image Texture node and assign the loaded image
    texNode = nodes.new('ShaderNodeTexImage')
    texNode.image = img

    # Connect the Image Texture color output to the Base Color input of the shader
    links.new(bsdf.inputs['Base Color'], texNode.outputs['Color'])

    # Assign the material to the active object, replacing existing or appending
    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Clear loading state flag
    context.scene.ai_texture_loading = False
    print("[AI_Texture] Texture successfully applied.")

class AITEXTURE_PT_panel(bpy.types.Panel):
    bl_label = "AI Texture Generator"
    bl_idname = "AI_TEXTURE_PT_panel"
    bl_space_type = 'VIEW_3D'  # The panel is in the 3D Viewport
    bl_region_type = 'UI'      # Located in the right sidebar
    bl_category = 'AI Texture' # Tab category name

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        caption = getattr(scene, "ai_texture_caption", "")
        loading = getattr(scene, "ai_texture_loading")

        layout.label(text="Texture Description:")

        row = layout.row()
        row.prop(scene, "ai_texture_caption", text="")
        if loading:
            layout.label(text="⏳ Generating...")
        else:
            layout.operator("ai_texture.generate")

def register():
    bpy.utils.register_class(AITEXTURE_OT_generate)
    bpy.utils.register_class(AITEXTURE_PT_panel)
    # Register properties on the Scene for user input and loading state
    bpy.types.Scene.ai_texture_caption = bpy.props.StringProperty(
        name="Description",
        description="Describe the texture you want",
        default="futuristic metallic texture",
    )
    bpy.types.Scene.ai_texture_loading = bpy.props.BoolProperty(
        name="Loading",
        default=False
    )
    # Register a timer to process the main thread execution queue periodically
    bpy.app.timers.register(process_queue)

def unregister():
    bpy.utils.unregister_class(AITEXTURE_OT_generate)
    bpy.utils.unregister_class(AITEXTURE_PT_panel)
    del bpy.types.Scene.ai_texture_caption
    del bpy.types.Scene.ai_texture_loading
    bpy.app.timers.unregister(process_queue)

if __name__ == "__main__":
    register()
