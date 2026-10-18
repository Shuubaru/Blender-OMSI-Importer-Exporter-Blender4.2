"""OMSI Mesh Sidebar Module

Registers the "OMSI" tab in the 3D Viewport Sidebar (N panel).

Panels:
`O3D_PT_sidebar`: Import / Export buttons
`O3D_PT_materials`: Viewport Display colour (diffuse) of materials
`O3D_PT_collections`: Viewport visibility of top-level collections

Operators are referenced by their idname so this module does not need to
import the operator classes (which would be a circular import).
"""

# <pep8-80 compliant>

from bpy.types import Context, Material, Panel
from bpy.utils import register_class, unregister_class


def _materials(context: Context) -> list[Material]:
    """Unique materials of the selected objects (or the active object)."""
    objects = list(context.selected_objects)
    if not objects and context.active_object:
        objects = [context.active_object]

    found: list[Material] = []
    seen: set[str] = set()
    for obj in objects:
        for slot in obj.material_slots:
            mat = slot.material
            if mat and mat.name_full not in seen:
                seen.add(mat.name_full)
                found.append(mat)
    return found


class O3D_PT_sidebar(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMSI"
    bl_label = "OMSI Mesh"

    def draw(self, context: Context) -> None:
        col = self.layout.column(align=True)
        col.operator("import_scene.o3d", text="Import", icon='IMPORT')
        col.operator("export_scene.o3d", text="Export", icon='EXPORT')


class O3D_PT_materials(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMSI"
    bl_label = "Diffuse Color"

    def draw(self, context: Context) -> None:
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False

        materials = _materials(context)
        if not materials:
            layout.label(text="No materials in selection")
            return

        # Material > Viewport Display > Color, the value the exporter
        # reads when "Textured Diffuse" is set to Viewport
        col = layout.column(align=True)
        for mat in materials:
            col.prop(mat, "diffuse_color", text=mat.name)


class O3D_PT_collections(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OMSI"
    bl_label = "Collections"

    def draw(self, context: Context) -> None:
        layout = self.layout

        # immediate children of the Scene Collection, in Outliner order
        # (the same ones, in the same order, as the 1-0 number keys)
        children = context.view_layer.layer_collection.children
        if not children:
            layout.label(text="No collections")
            return

        col = layout.column(align=True)
        for i, lc in enumerate(children):
            # the number keys only cover the first ten collections
            label = f"{(i + 1) % 10}  {lc.name}" if i < 10 else lc.name
            col.prop(
                lc, "hide_viewport",
                text=label,
                icon='HIDE_ON' if lc.hide_viewport else 'HIDE_OFF',
                invert_checkbox=True,
                toggle=True,
            )


classes = (
    O3D_PT_sidebar,
    O3D_PT_materials,
    O3D_PT_collections,
)


def register() -> None:
    for cls in classes:
        register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        unregister_class(cls)
