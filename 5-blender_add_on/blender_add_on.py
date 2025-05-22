bl_info = {
    "name": "UV Texture Generator with AI (Threaded)",
    "author": "Sbatt",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > AI Texture",
    "description": "Genera una texture via AI da caption e UV map con thread esterni",
    "category": "Object",
}

import bpy
import os
import threading
import subprocess
import queue
from PIL import Image 

execution_queue = queue.Queue()

def run_in_main_thread(func):
    execution_queue.put(func)

def process_queue():
    while not execution_queue.empty():
        fn = execution_queue.get()
        try:
            fn()
        except Exception as e:
            print(f"[AI_Texture] Errore in callback: {e}")
    return 0.5

class AITEXTURE_OT_generate(bpy.types.Operator):
    bl_idname = "ai_texture.generate"
    bl_label = "Genera Texture"
    bl_description = "Genera texture nera da UV map e descrizione utente (mock)"

    def execute(self, context):
        scene = context.scene
        obj = context.active_object

        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Seleziona un oggetto mesh")
            return {'CANCELLED'}

        scene.ai_texture_loading = True

        # Esporta UV map
        uv_path = bpy.path.abspath("C:\\Users\\tiasb\\Desktop\\uv_layout.png")
        bpy.ops.uv.export_layout(filepath=uv_path, size=(1024, 1024))

        caption = scene.ai_texture_caption  # In futuro sarà usata

        # Percorso per la texture mock
        out_path = bpy.path.abspath("//generated_texture.png")

        def worker():
            try:
                # --- INIZIO: MOCK GENERAZIONE TEXTURE ---
                print("[AI_Texture] Generazione texture nera fittizia...")
                img = Image.new('RGB', (1024, 1024), color='black')
                img.save(out_path)
                print("[AI_Texture] Texture nera salvata in:", out_path)
                # --- FINE: MOCK GENERAZIONE TEXTURE ---

                run_in_main_thread(lambda: apply_texture(out_path, context))
            except Exception as e:
                print(f"[AI_Texture] Errore mock generazione: {e}")
            finally:
                run_in_main_thread(lambda: setattr(scene, "ai_texture_loading", False))

        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

        return {'FINISHED'}

def apply_texture(tex_path, context):
    obj = context.active_object
    if not os.path.exists(tex_path):
        print(f"[AI_Texture] File non trovato: {tex_path}")
        return

    img = bpy.data.images.load(tex_path)

    mat = bpy.data.materials.new(name="AI_Generated_Texture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    texNode = nodes.new('ShaderNodeTexImage')
    texNode.image = img
    links.new(bsdf.inputs['Base Color'], texNode.outputs['Color'])

    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    context.scene.ai_texture_loading = False
    print("[AI_Texture] Texture nera applicata con successo.")


class AITEXTURE_PT_panel(bpy.types.Panel):
    bl_label = "AI Texture Generator"
    bl_idname = "AI_TEXTURE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AI Texture'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "ai_texture_caption")

        if scene.ai_texture_loading:
            layout.label(text="⏳ Generazione in corso…")
        else:
            layout.operator("ai_texture.generate")

def register():
    bpy.utils.register_class(AITEXTURE_OT_generate)
    bpy.utils.register_class(AITEXTURE_PT_panel)
    bpy.types.Scene.ai_texture_caption = bpy.props.StringProperty(
        name="Descrizione",
        description="Descrivi la texture che desideri",
        default="texture futuristica metallica"
    )
    bpy.types.Scene.ai_texture_loading = bpy.props.BoolProperty(
        name="In Caricamento",
        default=False
    )
    # Registra il timer per processare la queue
    bpy.app.timers.register(process_queue)

def unregister():
    bpy.utils.unregister_class(AITEXTURE_OT_generate)
    bpy.utils.unregister_class(AITEXTURE_PT_panel)
    del bpy.types.Scene.ai_texture_caption
    del bpy.types.Scene.ai_texture_loading
    # Rimuove il timer
    bpy.app.timers.unregister(process_queue)

if __name__ == "__main__":
    register()
