import bpy
import PIL


__scene_resetted = 0


def __reset_scene():
    """Delete all the objects from the current scene

    Args:
        clear_collections (bool, optional): If clear the collections too. Defaults to True.
    """
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except:
        pass

    # 1. Delete all objects
    bpy.ops.object.select_all(action="SELECT")
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

    global __scene_resetted
    __scene_resetted += 1
    if __scene_resetted % 100 == 0:
        bpy.ops.wm.read_factory_settings(use_empty=True)


def load_model(path: str, reset_scene: bool = True) -> list:
    """Load a GLB or OBJ 3D model

    Args:
        path (str): the path of the GLB/OBJ file

    Returns:
        list: a list of Blender's objects in the scene.
    """
    if reset_scene:
        __reset_scene()
    (bpy.ops.wm.obj_import if path.endswith(".obj") else bpy.ops.import_scene.gltf)(filepath=path)
    return list(bpy.context.scene.objects)
