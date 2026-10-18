"""OMSI Mesh Importer Module

Reads OMSI Mesh (.o3d) files and converts them to Blender Objects. It is
the inverse of `exporter.py`, reusing the format description in
`meshio.py`.

Classes:
`ImportO3D`: Blender import Operator

Functions:
`load()`: Decode binary data into a `meshio.Mesh`
`menu_func_import()`: File > Import menu entry
"""

# <pep8-80 compliant>

import struct
from pathlib import Path, PureWindowsPath

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    StringProperty,
)
from bpy.types import Context, Operator, OperatorFileListElement
from bpy_extras.io_utils import ImportHelper

from . import meshio


class O3DImportError(Exception):
    """Raised when an .o3d file cannot be decoded."""


# ---------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------

def load(data: bytes) -> meshio.Mesh:
    """Decode the contents of an .o3d file into a `meshio.Mesh`."""
    mfs = meshio.MeshFormatSpec

    if len(data) < 3 or data[:2] != mfs._idBytes:
        raise O3DImportError("Not an OMSI Mesh file (bad header)")

    version = data[2]
    pos = 3
    try:
        fs = mfs(version)
    except ValueError:
        raise O3DImportError(f"Unknown mesh version {version}")

    try:
        # extended addressing and equality bit
        if fs.supportsLongIndexes or fs.supportsEqualityBit:
            extra = data[pos]
            pos += 1
            fs.longIndexes = bool(extra & 1)
            fs.equalityBit = bool(extra & 2)

        # encryption
        if fs.supportsEncryption:
            (key,) = mfs._encryptionKeyFormat.unpack_from(data, pos)
            pos += mfs._encryptionKeyFormat.size
            fs.encryptionKey = key
            if fs.isEncrypted:
                raise O3DImportError("Encrypted meshes are not supported")
    except IndexError:
        raise O3DImportError("File is truncated")

    def read(fmt: struct.Struct) -> tuple:
        nonlocal pos
        values = fmt.unpack_from(data, pos)
        pos += fmt.size
        return values

    def read_string() -> str:
        nonlocal pos
        (n,) = read(mfs._stringLengthFormat)
        raw = data[pos:pos + n]
        if len(raw) != n:
            raise struct.error("string out of range")
        pos += n
        return raw.decode(mfs._stringFormat, errors='replace')

    mesh = meshio.Mesh()

    try:
        while pos < len(data):
            block = data[pos:pos + 1]
            pos += 1

            if block == mfs._idByteVertexList:
                (count,) = read(fs._vertexCount)
                for _ in range(count):
                    f = read(fs._vertex)
                    mesh.vertices.append(meshio.Vertex(
                        meshio.Vector3(*f[0:3]),
                        meshio.Vector3(*f[3:6]),
                        meshio.Vector2(*f[6:8]),
                    ))

            elif block == mfs._idByteTriangleList:
                (count,) = read(fs._triangleCount)
                for _ in range(count):
                    f = read(fs._triangle)
                    mesh.triangles.append(
                        meshio.Triangle((f[0], f[1], f[2]), f[3])
                    )

            elif block == mfs._idByteMaterialList:
                (count,) = read(fs._materialCount)
                for _ in range(count):
                    f = read(fs._material)
                    mesh.materials.append(meshio.Material(
                        meshio.ColorRGBA(*f[0:4]),
                        meshio.ColorRGB(*f[4:7]),
                        meshio.ColorRGB(*f[7:10]),
                        f[10],
                        read_string(),
                    ))

            elif block == mfs._idByteMatrix:
                f = read(fs._matrix)
                mesh.matrix = tuple(tuple(f[i:i + 4]) for i in range(0, 16, 4))

            elif block == mfs._idByteBoneList:
                (count,) = read(fs._boneCount)
                mesh.bones = []
                for _ in range(count):
                    name = read_string()
                    (wcount,) = read(fs._skinWeightCount)
                    weights = tuple(
                        meshio.SkinWeight(*read(fs._skinWeight))
                        for _ in range(wcount)
                    )
                    mesh.bones.append(meshio.Bone(name, weights))

            else:
                raise O3DImportError(
                    f"Unknown block id {block.hex()} at offset {pos - 1}"
                )
    except struct.error:
        raise O3DImportError("File is truncated or corrupt")

    return mesh


# ---------------------------------------------------------------------
# Blender conversion
# ---------------------------------------------------------------------

def _swap(v) -> tuple[float, float, float]:
    """Swap Y and Z (the exporter's reflection across the y=z plane)."""
    return (v[0], v[2], v[1])


def _find_texture_dirs(o3d_path: Path) -> list[Path]:
    """Find folders named 'texture' next to or above the .o3d file."""
    found: list[Path] = []
    for parent in (o3d_path.parent, *o3d_path.parents):
        try:
            for child in parent.iterdir():
                if child.is_dir() and child.name.lower() == "texture":
                    found.append(child)
        except OSError:
            continue
    return found


def _get_image(texture: str, o3d_path: Path, tex_dirs: list[Path]):
    """Load the image for an OMSI texture string, or a placeholder."""
    rel = Path(*PureWindowsPath(texture).parts)

    candidates = [d / rel for d in tex_dirs] + [o3d_path.parent / rel]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return bpy.data.images.load(str(candidate),
                                            check_existing=True)
            except RuntimeError:
                break

    # file not found: create a placeholder that still remembers where
    # the texture is expected to be, so re-exporting keeps the path
    guess = (tex_dirs[0] if tex_dirs
             else o3d_path.parent.parent / "Texture") / rel
    img = bpy.data.images.get(rel.name)
    if img is None or Path(bpy.path.abspath(img.filepath)) != guess:
        img = bpy.data.images.new(rel.name, 8, 8)
        img.source = 'FILE'
        img.filepath = str(guess)
    return img


def _make_material(name: str, m: meshio.Material, o3d_path: Path,
                   tex_dirs: list[Path], load_textures: bool):
    """Build a Principled BSDF material the exporter can read back."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes
                if n.bl_idname == 'ShaderNodeBsdfPrincipled')

    d, s, e = m.diffuse, m.specular, m.emissive
    roughness = min(1.0, max(0.0, (1000.0 - m.power) / 1000.0))

    bsdf.inputs['Base Color'].default_value = (d.r, d.g, d.b, 1.0)
    bsdf.inputs['Alpha'].default_value = d.a
    bsdf.inputs['Roughness'].default_value = roughness
    # exporter: specular = IOR level * tint
    bsdf.inputs['Specular IOR Level'].default_value = 1.0
    bsdf.inputs['Specular Tint'].default_value = (s.r, s.g, s.b, 1.0)
    # exporter: emissive = strength * colour
    bsdf.inputs['Emission Color'].default_value = (e.r, e.g, e.b, 1.0)
    bsdf.inputs['Emission Strength'].default_value = 1.0

    mat.diffuse_color = (d.r, d.g, d.b, d.a)

    if m.texture and load_textures:
        img = _get_image(m.texture, o3d_path, tex_dirs)
        node = nt.nodes.new('ShaderNodeTexImage')
        node.image = img
        node.location = (bsdf.location.x - 350, bsdf.location.y)
        nt.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
        if img.alpha_mode != 'NONE':
            # display only: the exporter ignores this link
            nt.links.new(node.outputs['Alpha'], bsdf.inputs['Alpha'])
    return mat


def build_object(collection: bpy.types.Collection, mesh: meshio.Mesh,
                 path: Path, merge_vertices: bool, weights: bool,
                 load_textures: bool) -> bpy.types.Object:
    """Convert a `meshio.Mesh` into a Blender Object in `collection`."""
    verts = mesh.vertices
    name = path.stem

    # --- vertex positions (optionally welded) ------------------------
    positions: list[tuple[float, float, float]] = []
    vmap: list[int] = []
    if merge_vertices:
        seen: dict[tuple, int] = {}
        for v in verts:
            # only weld vertices with the same position AND normal:
            # welding across different normals would force Blender to
            # represent the hard edge with sharp edge marks
            key = (tuple(v.co), tuple(round(x, 5) for x in v.no))
            idx = seen.get(key)
            if idx is None:
                idx = seen[key] = len(positions)
                positions.append(_swap(v.co))
            vmap.append(idx)
    else:
        positions = [_swap(v.co) for v in verts]
        vmap = list(range(len(verts)))

    # --- materials -> slots (identical materials share a slot) -------
    tex_dirs = _find_texture_dirs(path) if load_textures else []
    slot_by_key: dict[meshio.Material, int] = {}
    bl_materials = []
    slot_of: list[int] = []
    for i, m in enumerate(mesh.materials):
        slot = slot_by_key.get(m)
        if slot is None:
            slot = slot_by_key[m] = len(bl_materials)
            label = (Path(PureWindowsPath(m.texture).name).stem
                     if m.texture else f"{name}_mat")
            bl_materials.append(
                _make_material(label, m, path, tex_dirs, load_textures)
            )
        slot_of.append(slot)

    # --- faces, per-loop UVs and normals -----------------------------
    faces: list[tuple[int, int, int]] = []
    uvs: list[float] = []
    normals: list[tuple[float, float, float]] = []
    mat_indices: list[int] = []
    nverts = len(verts)

    for t in mesh.triangles:
        a, b, c = t.vertices
        if max(a, b, c) >= nverts:
            raise O3DImportError("Triangle references a missing vertex")
        # winding is reversed by the Y/Z swap (a reflection)
        order = (a, c, b)
        bv = tuple(vmap[i] for i in order)
        if len(set(bv)) < 3:
            continue  # degenerate after welding
        faces.append(bv)
        for i in order:
            v = verts[i]
            uvs.extend((v.uv[0], 1.0 - v.uv[1]))
            n = _swap(v.no)
            length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
            normals.append(tuple(x / length for x in n) if length > 1e-9
                           else (0.0, 0.0, 1.0))
        if slot_of:
            mat_indices.append(slot_of[min(t.material, len(slot_of) - 1)])

    # --- create the Blender mesh --------------------------------------
    me = bpy.data.meshes.new(name)
    me.from_pydata(positions, [], faces)

    for mat in bl_materials:
        me.materials.append(mat)
    if mat_indices:
        me.polygons.foreach_set("material_index", mat_indices)

    uv_layer = me.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", uvs)

    me.polygons.foreach_set("use_smooth", [True] * len(faces))
    if hasattr(me, "use_auto_smooth"):  # Blender < 4.1
        me.use_auto_smooth = True
    me.normals_split_custom_set(normals)
    me.update()

    obj = bpy.data.objects.new(name, me)
    collection.objects.link(obj)

    # --- skin weights -> vertex groups --------------------------------
    if weights and mesh.bones:
        for bone in mesh.bones:
            vg = obj.vertex_groups.new(name=bone.name)
            for w in bone.weights:
                if w.vertex < len(vmap):
                    vg.add([vmap[w.vertex]], w.weight, 'REPLACE')

    return obj


# ---------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------

class ImportO3D(Operator, ImportHelper):
    """Import OMSI Mesh file(s) (.o3d)"""
    bl_idname = "import_scene.o3d"
    bl_label = "Import O3D"
    bl_options = {'UNDO', 'PRESET'}

    filename_ext = ".o3d"
    filter_glob: StringProperty(default="*.o3d", options={'HIDDEN'})
    files: CollectionProperty(
        type=OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: StringProperty(subtype='DIR_PATH')

    merge_vertices: BoolProperty(
        name="Merge Vertices",
        description=("Weld vertices that share position and normal (the "
                     "format duplicates them per UV)"),
        default=True,
    )
    textures: BoolProperty(
        name="Load Textures",
        description="Look for textures in the nearest 'Texture' folder",
        default=True,
    )
    weights: BoolProperty(
        name="Skin Weights",
        description="Import bones as vertex groups",
        default=True,
    )

    def execute(self, context: Context) -> set[str]:
        folder = Path(self.directory)
        paths = [folder / f.name for f in self.files]
        if not paths:
            paths = [Path(self.filepath)]

        # several files: put them in a new Collection named after the
        # folder. It is linked directly under the Scene Collection so
        # it is a top-level collection, which is what the 1-0 number
        # keys in the viewport toggle between
        created = False
        if len(paths) > 1:
            col_name = folder.name or "O3D"
            # reuse the collection if a previous import already made it
            target = bpy.data.collections.get(col_name)
            if target is None:
                target = bpy.data.collections.new(col_name)
                created = True
            # make sure it is part of this scene (new, or previously
            # unlinked); an existing one stays wherever it already is
            if target not in context.scene.collection.children_recursive:
                context.scene.collection.children.link(target)
        else:
            target = context.collection

        imported = []
        for path in paths:
            try:
                mesh = load(path.read_bytes())
                obj = build_object(target, mesh, path,
                                   self.merge_vertices, self.weights,
                                   self.textures)
            except (O3DImportError, OSError) as e:
                self.report({'ERROR'}, f"{path.name}: {e}")
                continue
            imported.append(obj)

        if not imported:
            # nothing was imported, don't leave an empty collection behind
            if created:
                bpy.data.collections.remove(target)
            return {'CANCELLED'}

        for o in context.selected_objects:
            o.select_set(False)
        for o in imported:
            o.select_set(True)
        context.view_layer.objects.active = imported[-1]
        return {'FINISHED'}


def menu_func_import(self, _: Context) -> None:
    self.layout.operator(ImportO3D.bl_idname, text="Import OMSI Mesh (.o3d)")