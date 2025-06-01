from collections import defaultdict
import bpy

from src.blender.object3d.object3d import Object3D
from src.logger import cprint
from src.utils import *


class Processor:
    def __init__(self, object3d: Object3D):
        self.object3d = object3d
        self._elements: set[tuple[bpy.types.Object, bpy.types.Material, bpy.types.UVLoopLayers, bpy.types.Image]] = (
            None
        )

    @property
    def elements(self):
        assert self._elements is not None, "You need to analyze the scene first."
        return self._elements

    def uvs(self, pil=False) -> list[bpy.types.UVLoopLayers | Image.Image]:
        uvs = {x[2]: x[0].data for x in self.elements}
        return [self.object3d.draw_uv(mesh, uv.name, verbose=False) if pil else uv for uv, mesh in uvs.items()]

    def masks(self) -> list[np.ndarray]:
        uvs = {x[2]: x[0].data for x in self.elements}
        uv_fills = [self.object3d.draw_uv(mesh, uv.name, fill=True, verbose=False) for uv, mesh in uvs.items()]
        return [np.all(np.array(uv_fill) == [0, 0, 0, 255], axis=2) if uv_fill else None for uv_fill in uv_fills]

    def diffuses(self, pil=False) -> list[bpy.types.UVLoopLayers | Image.Image]:
        diffuses = {x[3] for x in self.elements}
        return [bpy2pil(dif) if pil else dif for dif in diffuses]

    def analyze_scene(self, verbose=True):
        elements = []
        for obj in bpy.data.objects:
            mesh = obj.data

            # Gather this mesh's UV layer names
            if len(mesh.uv_layers) == 0:
                continue

            # Gather all materials on this mesh
            mat_slots = list(obj.material_slots)
            if len(mat_slots) == 0:
                # No materials → record one line with "<no_mesh_material>"
                continue

            # For each material slot, find diffuse images
            for slot in mat_slots:
                mat = slot.material
                if mat is None or not mat.use_nodes:
                    # Material with no node tree → no texture
                    continue

                # Inspect the node tree
                found_any_image_for_this_material = False
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        # Check if this node is actually feeding Base Color of a principled BSDF
                        for out_socket in node.outputs:
                            # We expect node.outputs['Color'] → some link → principled_bsdf.inputs['Base Color']
                            for link in out_socket.links:
                                to_node = link.to_node

                                # Walk the chain if necessary: If the link goes into a Mix→...→Principled,
                                # we should ideally find Principled BSDF. For simplicity, we check if *any*
                                # path leads to a Principled BSDF Base Color.
                                def leads_to_base_color(node_in):
                                    """
                                    Returns True if node_in is a Principled BSDF that is connected to 'Base Color'.
                                    If node_in is a Mix/RGB node that eventually feeds into principled, we recurse.
                                    """
                                    if node_in.type == "BSDF_PRINCIPLED":
                                        # Check if this link is into ‘Base Color’ specifically
                                        for idx_input, socket in enumerate(node_in.inputs):
                                            if socket.name == "Base Color":
                                                # If any of its links come from our chain, OK
                                                for s_link in socket.links:
                                                    if s_link.from_node == node:
                                                        return True
                                                # (Note: in complicated graphs, we might have to traverse further,
                                                # but in most GLTF imports it’s direct.)
                                        return False
                                    else:
                                        # If it’s a MixRGB or other node with outputs that go to Principled, recurse
                                        for out2 in node_in.outputs:
                                            for l2 in out2.links:
                                                if leads_to_base_color(l2.to_node):
                                                    return True
                                        return False

                                if leads_to_base_color(to_node):
                                    # At this point, node.image is the diffuse image
                                    img = node.image
                                    if img is not None:
                                        # Determine the UV map used by this TexImage node
                                        uv_input = node.inputs.get("Vector")
                                        if uv_input is not None and len(uv_input.links) > 0:
                                            from_node = uv_input.links[0].from_node
                                            if from_node.type == "UVMAP":
                                                uv_layer = from_node.uv_map
                                            else:
                                                # If no UVMap node is inserted, Blender defaults to first UV
                                                uv_layer = mesh.uv_layers[0]
                                        else:
                                            # No link → use default “UVMap” if present
                                            uv_layer = mesh.uv_layers[0]

                                        elements.append((obj, mat, uv_layer, img))

        if verbose:
            cprint(
                "Found",
                f"red:{len({x[1] for x in elements})} materials,",
                f"blue:{len({x[2] for x in elements})} UV-maps,",
                f"green:{len({x[3] for x in elements})} diffuse",
                "in",
                f"yellow:{len(bpy.data.objects)} objects",
            )

        self._elements = set(elements)

    def group_meshes(self, verbose=True):
        # Find groups of meshes with sharing the same material
        groups = defaultdict(list)
        for obj, mat, _, _ in self.elements:
            groups[mat].append(obj)

        # Join the group of meshes
        for mat, objs in groups.items():
            if len(objs) > 1:
                self.object3d.join(*objs)

        if verbose:
            cprint(
                "Joined",
                f"blue:({', '.join(str(len(objs)) for _, objs in groups.items() if len(objs)>1)}) objects",
                "from",
                f"red:({', '.join(mat.name for mat, objs in groups.items() if len(objs)>1)}) materials",
            )

        self._elements = None
