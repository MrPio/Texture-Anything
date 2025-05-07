import PIL.Image
import bpy
import PIL
import numpy as np
from ..utils import plot_images


def __reset_scene():
    """Delete all the objects from the current scene

    Args:
        clear_collections (bool, optional): If clear the collections too. Defaults to True.
    """

    # 1. Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # 2. Remove all collections (except the master “Scene Collection”, which will be empty)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)

    # 3. Purge all datablocks of unused/orphan data types
    # You may want to call this once at the end, to be sure nothing is left hanging.
    def purge_datablocks(datablocks):
        for db in list(datablocks):
            datablocks.remove(db)

    purge_datablocks(bpy.data.meshes)
    purge_datablocks(bpy.data.curves)
    purge_datablocks(bpy.data.materials)
    purge_datablocks(bpy.data.textures)
    purge_datablocks(bpy.data.images)
    purge_datablocks(bpy.data.armatures)
    purge_datablocks(bpy.data.actions)
    purge_datablocks(bpy.data.node_groups)
    purge_datablocks(bpy.data.cameras)
    purge_datablocks(bpy.data.lights)

    # 4. (Optional) Force orphan datablock purge
    # This will scan for any data-blocks without users and remove them.
    # Note: in some versions of Blender you might need to toggle these flags.
    # bpy.ops.outliner.orphans_purge(
    #     do_local_ids=True, do_linked_ids=True, do_recursive=True)


def load_glb(path: str, reset_scene: bool = True) -> list:
    """Load a GLB 3D model

    Args:
        path (str): the path of the GLB file

    Returns:
        list: a list of Blender's objects in the scene.
    """
    if reset_scene:
        __reset_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    return list(bpy.context.scene.objects)


def get_scene_stats() -> dict:
    """Get the properties of the current scene.
    """
    mesh_objects = [
        obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    return {'mesh_count': len(mesh_objects)}


def get_mesh_stats(mesh) -> dict:
    """Get the properties of a given mesh in the current scene.
    """
    assert mesh.type == 'MESH'

    texture_count = 0
    for slot in mesh.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    for output in node.outputs:
                        for link in output.links:
                            if link.to_socket.name == 'Base Color':
                                texture_count += 1

    return {'uv_count': len(mesh.data.uv_layers), 'texture_count': texture_count}


def get_textures(plot: bool = False) -> list[PIL.Image.Image]:
    """Unpack all the texture images in the scene as PIL
    """

    embedded_images = [
        img for img in bpy.data.images if img.packed_file is not None]
    images_pil = []

    if embedded_images:
        for i, img in enumerate(embedded_images):
            pixels = (np.array(img.pixels)*255).astype(np.uint8)
            pixels = pixels.reshape((*img.size, 4))
            image_pil = PIL.Image.fromarray(pixels, 'RGBA')
            images_pil.append(image_pil)
    if plot and len(images_pil) > 0:
        plot_images(images_pil, col=min(4, len(images_pil)))
    return images_pil
