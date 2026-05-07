"""
Blender API compatibility shims for ComfyUI-UniRig.

Lives separately so future API drifts have a clear home. Imported lazily
from each call site to avoid module-level `import bpy` (UniRig defers
that until inside functions to avoid clashing with torch_cluster on
Linux).

Currently covers ONE break:

  Blender 4.4 introduced "slotted actions" — `bpy.types.Action.fcurves`
  moved from a flat collection to a slot/layer/strip/channelbag tree.
  Actions created via `bpy.data.actions.new()` in 4.4+ have no top-level
  `.fcurves` attribute (AttributeError on access).
  Action.copy() of a legacy action retains `.fcurves`, but
  bpy.data.actions.new() does not.

  https://developer.blender.org/docs/release_notes/4.4/animation/

Tested against Blender 4.2 LTS (legacy) and 5.x (slotted).
"""

from __future__ import annotations


def action_fcurves(action):
    """Return an Action's fcurves collection regardless of Blender API era.

    Blender 4.2 and earlier: ``action.fcurves`` always works.
    Blender 4.4+: legacy actions still expose ``.fcurves`` as a shim
    when the action has a single slot+layer+strip; brand-new actions
    from ``bpy.data.actions.new()`` have no top-level ``.fcurves``.

    For new actions we lazily seed
    ``action.layers[0].strips[0].channelbags[0]`` and return its
    ``.fcurves``. Reads on legacy actions take the fast path.

    Both legacy collection and channelbag.fcurves expose the same
    API surface (``.new(data_path=..., index=...)``, iteration,
    ``len()``), so call sites don't need to change beyond the
    accessor.
    """
    # Legacy fast path — works on 4.2, and on 4.4 actions where the
    # shim is in effect.
    fc = getattr(action, "fcurves", None)
    if fc is not None:
        return fc

    # Slotted action (Blender 4.4+, action freshly created by
    # bpy.data.actions.new() with no slot yet).
    # Seed slot → layer → strip → channelbag → return its fcurves.
    if not action.slots:
        action.slots.new(id_type='OBJECT', name="ActionSlot")
    slot = action.slots[0]

    if not action.layers:
        action.layers.new(name="Layer")
    layer = action.layers[0]

    if not layer.strips:
        # 'KEYFRAME' is the strip type for keyframed F-curves;
        # the API requires a type kwarg in 4.4+.
        layer.strips.new(type='KEYFRAME')
    strip = layer.strips[0]

    if not strip.channelbags:
        strip.channelbags.new(slot=slot)
    return strip.channelbags[0].fcurves
