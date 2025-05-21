bl_info = {
    "name": "UV Texture Generator with AI",
    "author": "Sbatt",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > AI Texture",
    "description": "Genera una texture via AI da caption e UV map",
    "category": "Object",
}

import bpy
import os

class AITEXTURE_OT_generate(bpy.types.Operator):
    bl_idname = "ai_texture.generate"  # ID per chiamare l’operatore (interno)
    bl_label = "Genera Texture"        # Nome visibile del bottone
    bl_description = "Genera texture da UV map e descrizione utente"

    # Funzione chiamata dal pulsante  
    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Seleziona un oggetto mesh")
            return {'CANCELLED'}

        # Esporta la UV map come immagine
        uv_path = bpy.path.abspath("//uv_layout.png")
        bpy.ops.uv.export_layout(filepath=uv_path, size=(1024, 1024))

        # Prende la caption dell'utente
        caption = context.scene.ai_texture_caption

        # --------- PUNTO SOSPESO: chiamata al modello IA ----------
        # Si invia `uv_path` + `caption` al modello AI
        # restituisce un'immagine compatibile con la UV map
        #
        # Esempio:
        # generated_texture_path = call_ai_model(uv_path, caption)
        #
        # Per ora simuliamo il path di ritorno:
        generated_texture_path = bpy.path.abspath("//generated_texture.png")
        self.report({'INFO'}, f"Attendi: salva un'immagine finta in {generated_texture_path}")
        # ----------------------------------------------------------

        # Se il file generato esiste, lo carichiamo
        if not os.path.exists(generated_texture_path):
            self.report({'ERROR'}, f"Texture generata non trovata: {generated_texture_path}")
            return {'CANCELLED'}

        # Crea un nuovo materiale e applica la texture
        mat = bpy.data.materials.new(name="AI_Generated_Texture")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        #Collega l’output del nodo immagine al Base Color del Principled BSDF.
        bsdf = nodes.get("Principled BSDF")
        texImage = nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(generated_texture_path)
        links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

        # Applica il materiale all'oggetto
        if len(obj.data.materials):
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        self.report({'INFO'}, "Texture applicata con successo.")
        return {'FINISHED'}

#Definisce un pannello nella Sidebar di Blender (tasto N), categoria AI Texture.
class AITEXTURE_PT_panel(bpy.types.Panel):
    bl_label = "AI Texture Generator"
    bl_idname = "AITEXTURE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AI Texture'

    #un campo per inserire la caption (StringProperty)
    #un pulsante per avviare la generazione della texture.
    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "ai_texture_caption")
        layout.operator("ai_texture.generate")

# Installa l’add-on
def register():
    #Registra classi operatore e pannello.
    bpy.utils.register_class(AITEXTURE_OT_generate)
    bpy.utils.register_class(AITEXTURE_PT_panel)
    #Aggiunge una proprietà personalizzata alla scena: un campo testo per l'input utente.
    bpy.types.Scene.ai_texture_caption = bpy.props.StringProperty(
        name="Descrizione",
        description="Descrivi la texture che desideri",
        default="texture futuristica metallica"
    )

# Rimuove l'add-on
def unregister():
    bpy.utils.unregister_class(AITEXTURE_OT_generate)
    bpy.utils.unregister_class(AITEXTURE_PT_panel)
    del bpy.types.Scene.ai_texture_caption

if __name__ == "__main__":
    register()
