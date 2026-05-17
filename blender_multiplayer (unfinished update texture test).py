# ╔══════════════════════════════════════════════════════════════════════════╗
# ║         BLENDER MULTIPLAYER SYNC  v5.1  –  blender_multiplayer.py      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# FIX IN v5.1  (viewport not showing texture even though preview was correct)
# ──────────────────────────────────────────────────────────────────────────
#   The texture was connected to the node tree correctly but the Blender
#   depsgraph / GPU shader cache was not being invalidated, so the viewport
#   kept rendering the old solid colour.
#
#   Fixes applied:
#   • mat.node_tree.update_tag()  after every node-tree change
#   • obj.update_tag()            for every object that owns the material
#   • img.source = 'FILE' + img.reload()  guarantees pixel data is decoded
#   • bpy.context.view_layer.update()  called in the sync timer after any
#     texture was installed that tick
#   • mat_color is no longer written when a TEX_IMAGE node is already wired
#     to the Base Color socket (prevents the solid-colour from overriding
#     the texture after a subsequent delta tick)
#
# ARCHITECTURE (unchanged from v5.0)
#   Manifest mp_textures.json stores (obj_name, mat_name, img_name) per hash.
#   _assign_texture() walks straight to that object → material → node → image.
#   No filepath searching, no guessing.
#
# INSTALL  Edit > Preferences > Add-ons > Install… → pick this file → ✓
# USE      N-panel → "Multiplayer" tab.  Set Texture Folder BEFORE connecting.
# NETWORK  LAN / same Wi-Fi works out of the box (TCP 19283).
#          Internet: host port-forwards 19283, or use ZeroTier / Tailscale.

bl_info = {
    "name":        "Blender Multiplayer Sync",
    "author":      "Claude & LAGkit",
    "version":     (5, 1, 0),
    "blender":     (3, 0, 0),
    "location":    "View3D > N-Panel > Multiplayer",
    "description": "Real-time collaboration – objects, meshes, textures (direct assign), keyframes",
    "category":    "3D View",
}

import os, base64, json, struct, math, socket, threading
import mathutils, bpy, bpy_extras.view3d_utils, blf, gpu, bmesh
from gpu_extras.batch import batch_for_shader
from bpy.props import StringProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PORT   = 19283
SYNC_INTERVAL  = 0.05
LABEL_SIZE     = 14
CURSOR_R       = 10
CAM_R          = 7
PREC           = 5
TEX_LIMIT_MB   = 32
MANIFEST_FILE  = "mp_textures.json"

COLOURS = [
    (1.00, 0.42, 0.12, 1.0), (0.18, 0.85, 0.42, 1.0),
    (0.20, 0.55, 1.00, 1.0), (0.95, 0.15, 0.60, 1.0),
    (1.00, 0.90, 0.10, 1.0), (0.55, 0.20, 1.00, 1.0),
    (0.10, 0.88, 0.90, 1.0),
]


# ═════════════════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═════════════════════════════════════════════════════════════════════════════

class _MP:
    server_sock  = None
    my_sock      = None
    client_conns = {}
    conn_lock    = threading.Lock()

    peer_data    = {}
    peer_lock    = threading.Lock()

    in_queue     = []
    q_lock       = threading.Lock()

    obj_baseline  = {}
    baseline_lock = threading.Lock()

    sent_tex_hashes = set()
    tex_lock        = threading.Lock()

    # { tex_hash → [(obj_name, mat_name, img_name)] }
    pending      = {}
    pending_lock = threading.Lock()

    # Set to True by _install_texture; cleared after view_layer.update() in timer
    needs_depsgraph_update = False

    is_host      = False
    connected    = False
    status       = "Disconnected"
    colour_idx   = 0
    draw_handle  = None
    timer_active = False

    cached_region    = None
    cached_region_3d = None

MP = _MP()


# ═════════════════════════════════════════════════════════════════════════════
#  MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def _mpath(folder: str) -> str:
    return os.path.join(bpy.path.abspath(folder), MANIFEST_FILE)


def _load_manifest(folder: str) -> dict:
    try:
        with open(_mpath(folder), "r", encoding="utf-8") as f:
            return json.load(f).get("entries", {})
    except Exception:
        return {}


def _save_manifest(folder: str, entries: dict):
    try:
        os.makedirs(bpy.path.abspath(folder), exist_ok=True)
        with open(_mpath(folder), "w", encoding="utf-8") as f:
            json.dump({"version": 2, "entries": entries}, f, indent=2)
    except Exception as e:
        print(f"[MP] manifest write: {e}")


def _manifest_add(folder: str, tex_hash: str,
                  img_name: str, mat_name: str, obj_name: str, filename: str):
    entries = _load_manifest(folder)
    entries[tex_hash] = {"img_name": img_name, "mat_name": mat_name,
                         "obj_name": obj_name,  "filename": filename}
    _save_manifest(folder, entries)


# ═════════════════════════════════════════════════════════════════════════════
#  VIEWPORT / DEPSGRAPH  force-refresh helpers
# ═════════════════════════════════════════════════════════════════════════════

def _tag_material_update(mat):
    """
    Tell Blender's depsgraph that this material's node tree has changed.
    Without this the GPU shader cache is stale and the viewport shows the
    old colour even though the node IS wired correctly.
    """
    try:
        mat.node_tree.update_tag()
    except Exception:
        pass


def _tag_objects_using(mat):
    """Tag every object that has this material so the viewport re-shades them."""
    for obj in bpy.data.objects:
        if any(ms.material == mat for ms in obj.material_slots):
            try:
                obj.update_tag()
            except Exception:
                pass


def _force_viewport_redraw():
    """Redraw all open 3-D viewports and node editors."""
    try:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type in ('VIEW_3D', 'NODE_EDITOR', 'PROPERTIES'):
                    area.tag_redraw()
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  DIRECT TEXTURE ASSIGNMENT  ← core of the rework
# ═════════════════════════════════════════════════════════════════════════════

def _assign_texture(obj_name: str, mat_name: str, img: "bpy.types.Image") -> bool:
    """
    Directly wire `img` into the TEX_IMAGE node of `mat_name` and connect it
    to the first available colour socket.  Then invalidate the depsgraph so
    the viewport actually redraws with the new texture.

    Returns True if the material was found and updated.
    """
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        return False

    if not mat.use_nodes:
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # ── find / create TEX_IMAGE node ──────────────────────────────────────
    tex_node = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)
    if tex_node is None:
        tex_node          = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, 300)

    # Assign image + force pixel data to be decoded from disk
    tex_node.image        = img
    img.source            = 'FILE'          # ensure file-based, not generated
    try:
        img.reload()                        # decode pixels now
    except Exception:
        pass

    # ── find shader node to wire into ─────────────────────────────────────
    shader = (
        next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None) or
        next((n for n in nodes if n.type == 'EMISSION'),         None) or
        next((n for n in nodes if n.bl_idname.startswith('ShaderNodeBsdf')), None)
    )

    if shader:
        for sname in ('Base Color', 'Color', 'Emission Color'):
            if sname in shader.inputs:
                inp = shader.inputs[sname]
                for lk in list(inp.links):
                    links.remove(lk)
                links.new(tex_node.outputs['Color'], inp)
                break

    # ── make sure obj_name actually has this material ──────────────────────
    obj = bpy.data.objects.get(obj_name)
    if obj and hasattr(obj, 'data') and hasattr(obj.data, 'materials'):
        slot_mats = [ms.material for ms in obj.material_slots]
        if mat not in slot_mats:
            if obj.material_slots:
                obj.material_slots[0].material = mat
            else:
                try:
                    obj.data.materials.append(mat)
                except Exception:
                    pass

    # ── CRITICAL: invalidate GPU shader cache ─────────────────────────────
    _tag_material_update(mat)
    _tag_objects_using(mat)
    _force_viewport_redraw()
    MP.needs_depsgraph_update = True        # picked up by _sync_tick

    print(f"[MP] ✓ '{img.name}'  →  obj:'{obj_name}'  mat:'{mat_name}'")
    return True


def _resolve_pending(tex_hash: str, img: "bpy.types.Image"):
    with MP.pending_lock:
        targets = MP.pending.pop(tex_hash, [])
    for obj_name, mat_name, img_name in targets:
        _assign_texture(obj_name, mat_name, img)


def _queue_pending(tex_hash: str, obj_name: str, mat_name: str, img_name: str):
    with MP.pending_lock:
        MP.pending.setdefault(tex_hash, []).append((obj_name, mat_name, img_name))


def _replay_manifest(folder: str):
    """
    Re-apply every texture recorded in the manifest whose file already exists
    in the sync folder.  Queues anything whose file hasn't arrived yet.
    """
    entries = _load_manifest(folder)
    for tex_hash, e in entries.items():
        img_name = e.get("img_name", "")
        mat_name = e.get("mat_name", "")
        obj_name = e.get("obj_name", "")
        filename = e.get("filename", "")
        if not all((img_name, mat_name, obj_name, filename)):
            continue
        save_path = os.path.join(bpy.path.abspath(folder), filename)
        if not os.path.exists(save_path):
            continue

        img = bpy.data.images.get(img_name)
        if img is None:
            try:
                img       = bpy.data.images.load(save_path)
                img.name  = img_name
                img.source = 'FILE'
            except Exception as exc:
                print(f"[MP] replay load '{img_name}': {exc}")
                continue
        else:
            img.source   = 'FILE'
            img.filepath = save_path
            try:
                img.reload()
            except Exception:
                pass

        if not _assign_texture(obj_name, mat_name, img):
            _queue_pending(tex_hash, obj_name, mat_name, img_name)


# ═════════════════════════════════════════════════════════════════════════════
#  TEXTURE PUSH helpers
# ═════════════════════════════════════════════════════════════════════════════

def _file_hash(filepath: str) -> str:
    try:
        st = os.stat(filepath)
        return f"{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        return ""


def _save_filename(img_name: str, h: str, ext: str) -> str:
    return f"{img_name.replace('/', '_').replace(chr(92), '_')}__{h}{ext}"


def _build_tex_push(img_name: str, filepath: str,
                    mat_name: str, obj_name: str) -> "dict | None":
    try:
        size = os.path.getsize(filepath)
        if size > TEX_LIMIT_MB * 1024 * 1024:
            print(f"[MP] skip '{img_name}' – {size//1024//1024} MB > limit")
            return None
        h   = _file_hash(filepath)
        ext = os.path.splitext(filepath)[1] or ".png"
        fn  = _save_filename(img_name, h, ext)
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"type": "tex_push", "img_name": img_name, "mat_name": mat_name,
                "obj_name": obj_name, "hash": h, "ext": ext, "filename": fn, "b64": b64}
    except Exception as e:
        print(f"[MP] build_tex_push: {e}")
        return None


def _collect_tex_pushes(skip: set) -> list:
    msgs, seen = [], set()
    for obj in bpy.data.objects:
        mat = obj.active_material
        if not mat or not mat.use_nodes:
            continue
        tex_node = next(
            (n for n in mat.node_tree.nodes
             if n.type == 'TEX_IMAGE' and n.image), None)
        if not tex_node:
            continue
        img      = tex_node.image
        filepath = bpy.path.abspath(img.filepath) if img.filepath else ""
        if not filepath or not os.path.exists(filepath):
            continue
        h = _file_hash(filepath)
        if not h or h in skip or h in seen:
            continue
        seen.add(h)
        push = _build_tex_push(img.name, filepath, mat.name, obj.name)
        if push:
            msgs.append(push)
    return msgs


def _install_texture(msg: dict, sync_folder: str):
    """
    Save binary → update manifest → load image datablock →
    assign directly to the exact object/material → invalidate depsgraph.
    """
    if not sync_folder:
        print("[MP] No Texture Folder set – skipping texture.")
        return

    folder    = bpy.path.abspath(sync_folder)
    os.makedirs(folder, exist_ok=True)

    img_name  = msg["img_name"]
    mat_name  = msg["mat_name"]
    obj_name  = msg["obj_name"]
    tex_hash  = msg["hash"]
    ext       = msg.get("ext", ".png")
    filename  = msg.get("filename") or _save_filename(img_name, tex_hash, ext)
    save_path = os.path.join(folder, filename)

    # save binary
    if not os.path.exists(save_path):
        try:
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(msg["b64"]))
            print(f"[MP] saved → {save_path}")
        except Exception as e:
            print(f"[MP] install_texture write: {e}")
            return

    # update manifest
    _manifest_add(sync_folder, tex_hash, img_name, mat_name, obj_name, filename)

    # load / refresh image datablock
    img = bpy.data.images.get(img_name)
    if img is None:
        try:
            img        = bpy.data.images.load(save_path)
            img.name   = img_name
            img.source = 'FILE'
        except Exception as e:
            print(f"[MP] load image '{img_name}': {e}")
            return
    else:
        img.source   = 'FILE'
        img.filepath = save_path
        try:
            img.reload()
        except Exception:
            pass

    # assign directly; queue if object/material not ready yet
    if not _assign_texture(obj_name, mat_name, img):
        _queue_pending(tex_hash, obj_name, mat_name, img_name)

    _resolve_pending(tex_hash, img)


# ═════════════════════════════════════════════════════════════════════════════
#  OBJECT STATE helpers
# ═════════════════════════════════════════════════════════════════════════════

def _obj_to_state(obj, sync_mesh: bool = False) -> dict:
    loc  = [round(v, PREC) for v in obj.location]
    mode = obj.rotation_mode
    rot  = ([round(v, PREC) for v in obj.rotation_quaternion] if mode == 'QUATERNION' else
            [round(v, PREC) for v in obj.rotation_axis_angle]  if mode == 'AXIS_ANGLE'  else
            [round(v, PREC) for v in obj.rotation_euler])
    state: dict = {"type": obj.type, "loc": loc,
                   "rot_mode": mode, "rot": rot,
                   "scale": [round(v, PREC) for v in obj.scale]}

    mat = obj.active_material
    if mat:
        state["mat_name"] = mat.name
        if mat.use_nodes:
            try:
                nodes    = mat.node_tree.nodes
                bsdf     = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                tex_node = next((n for n in nodes
                                 if n.type == 'TEX_IMAGE' and n.image), None)

                # Only send mat_color if there is NO texture wired to Base Color.
                # If a texture is wired, the default_value is irrelevant and
                # sending it would cause a solid-colour flash on the receiver.
                has_tex_wired = (
                    tex_node is not None and
                    bsdf is not None and
                    any(lk.to_socket == bsdf.inputs.get("Base Color")
                        for lk in (tex_node.outputs[0].links
                                   if tex_node.outputs else []))
                )
                if bsdf and not has_tex_wired:
                    c = bsdf.inputs["Base Color"].default_value
                    state["mat_color"] = [round(v, PREC) for v in c]

                if tex_node:
                    img      = tex_node.image
                    filepath = bpy.path.abspath(img.filepath) if img.filepath else ""
                    if filepath and os.path.exists(filepath):
                        state["tex"] = {
                            "img_name": img.name,
                            "mat_name": mat.name,
                            "obj_name": obj.name,
                            "hash":     _file_hash(filepath),
                            "ext":      os.path.splitext(filepath)[1] or ".png",
                            "filepath": filepath,
                        }
            except Exception:
                pass

    if obj.animation_data and obj.animation_data.action:
        try:
            act    = obj.animation_data.action
            curves = [{"path": fc.data_path, "idx": fc.array_index,
                       "kps": [(round(kp.co[0], PREC), round(kp.co[1], PREC))
                               for kp in fc.keyframe_points]}
                      for fc in act.fcurves]
            state["anim"] = {"name": act.name, "curves": curves}
        except Exception as e:
            print(f"[MP] anim ({obj.name}): {e}")

    if sync_mesh and obj.type == 'MESH':
        try:
            if threading.current_thread() is threading.main_thread():
                if obj.mode == 'EDIT':
                    obj.update_from_editmode()
            n = len(obj.data.vertices)
            if n:
                raw = [0.0] * (n * 3)
                obj.data.vertices.foreach_get("co", raw)
                state["verts"] = [round(v, PREC) for v in raw]
                state["faces"] = [list(p.vertices) for p in obj.data.polygons]
        except Exception as e:
            print(f"[MP] mesh ({obj.name}): {e}")

    return state


def _anim_sig(anim: dict) -> str:
    return "|".join(
        f"{fc['path']}[{fc['idx']}]:{len(fc['kps'])}:"
        f"{fc['kps'][-1][0]:.2f}:{fc['kps'][-1][1]:.4f}"
        if fc.get("kps") else f"{fc['path']}[{fc['idx']}]:0"
        for fc in anim.get("curves", [])
    )


def _state_key(s: dict) -> tuple:
    tex = s.get("tex", {})
    return (
        tuple(s.get("loc",   [])),
        s.get("rot_mode", ""),
        tuple(s.get("rot",   [])),
        tuple(s.get("scale", [])),
        hash(tuple(s["verts"]))  if "verts"     in s else 0,
        tuple(s["mat_color"])    if "mat_color" in s else (),
        tex.get("hash", ""),
        _anim_sig(s["anim"])     if "anim"      in s else "",
        s.get("deleted", False),
    )


def _collect_full(sync_mesh=False) -> dict:
    return {obj.name: _obj_to_state(obj, sync_mesh) for obj in bpy.data.objects}


def _collect_delta(sync_mesh=False) -> dict:
    changed = {}
    current = {obj.name for obj in bpy.data.objects}
    with MP.baseline_lock:
        baseline = dict(MP.obj_baseline)
    for name, old in baseline.items():
        if name not in current and not old.get("deleted"):
            changed[name] = {"deleted": True}
    for obj in bpy.data.objects:
        new  = _obj_to_state(obj, sync_mesh)
        prev = baseline.get(obj.name)
        if prev is None or _state_key(prev) != _state_key(new):
            changed[obj.name] = new
    return changed


# ═════════════════════════════════════════════════════════════════════════════
#  APPLY STATE  (main thread only)
# ═════════════════════════════════════════════════════════════════════════════

def _apply_keyframes(obj, anim: dict):
    if not anim.get("name"):
        return
    if not obj.animation_data:
        obj.animation_data_create()
    act = bpy.data.actions.get(anim["name"]) or bpy.data.actions.new(anim["name"])
    if obj.animation_data.action is not act:
        obj.animation_data.action = act
    incoming = {(fc["path"], fc["idx"]) for fc in anim.get("curves", [])}
    for fc in list(act.fcurves):
        if (fc.data_path, fc.array_index) not in incoming:
            act.fcurves.remove(fc)
    for fc_data in anim.get("curves", []):
        path, idx, kps = fc_data["path"], fc_data["idx"], fc_data["kps"]
        ex = act.fcurves.find(path, index=idx)
        if ex:
            act.fcurves.remove(ex)
        if not kps:
            continue
        fc = act.fcurves.new(path, index=idx)
        fc.keyframe_points.add(len(kps))
        for i, (frame, value) in enumerate(kps):
            kp = fc.keyframe_points[i]
            kp.co_ui = (frame, value)
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = kp.handle_right_type = 'AUTO'
        fc.update()


def _state_to_obj(obj_name: str, obj, state: dict, sync_folder: str):
    try:
        # Transform
        obj.location = state["loc"]
        mode = state.get("rot_mode", "XYZ")
        if obj.rotation_mode != mode:
            obj.rotation_mode = mode
        if mode == 'QUATERNION':
            obj.rotation_quaternion = state["rot"]
        elif mode == 'AXIS_ANGLE':
            obj.rotation_axis_angle = state["rot"]
        else:
            obj.rotation_euler = state["rot"]
        obj.scale = state["scale"]

        # Material
        if "mat_name" in state:
            mn  = state["mat_name"]
            mat = bpy.data.materials.get(mn) or bpy.data.materials.new(mn)
            if not mat.use_nodes:
                mat.use_nodes = True
            if not obj.material_slots:
                obj.data.materials.append(mat)
            elif obj.material_slots[0].material != mat:
                obj.material_slots[0].material = mat

            # Only set the solid colour if there is no texture for this object
            if "mat_color" in state and "tex" not in state:
                try:
                    bsdf = next((n for n in mat.node_tree.nodes
                                 if n.type == 'BSDF_PRINCIPLED'), None)
                    if bsdf:
                        bsdf.inputs["Base Color"].default_value = state["mat_color"]
                except Exception:
                    pass

        # Texture – direct assignment via manifest
        tex = state.get("tex")
        if tex:
            h        = tex["hash"]
            img_name = tex["img_name"]
            mat_name = tex["mat_name"]

            img = bpy.data.images.get(img_name)
            img_ready = (img and img.filepath
                         and os.path.exists(bpy.path.abspath(img.filepath)))

            if img_ready:
                _assign_texture(obj_name, mat_name, img)
            elif sync_folder:
                entries  = _load_manifest(sync_folder)
                entry    = entries.get(h, {})
                filename = entry.get("filename", "")
                sp = (os.path.join(bpy.path.abspath(sync_folder), filename)
                      if filename else "")
                if sp and os.path.exists(sp):
                    if img is None:
                        try:
                            img        = bpy.data.images.load(sp)
                            img.name   = img_name
                            img.source = 'FILE'
                        except Exception as e2:
                            print(f"[MP] load from manifest: {e2}")
                    else:
                        img.source   = 'FILE'
                        img.filepath = sp
                        try:
                            img.reload()
                        except Exception:
                            pass
                    if img:
                        _assign_texture(obj_name, mat_name, img)
                else:
                    _queue_pending(h, obj_name, mat_name, img_name)
            else:
                _queue_pending(h, obj_name, mat_name, img_name)

        # Keyframes
        if state.get("anim"):
            _apply_keyframes(obj, state["anim"])

        # Mesh
        if "verts" in state and obj.type == 'MESH':
            verts, faces = state["verts"], state.get("faces", [])
            n = len(verts) // 3
            if n != len(obj.data.vertices) or len(faces) != len(obj.data.polygons):
                if obj.mode != 'EDIT':
                    obj.data.clear_geometry()
                    obj.data.from_pydata(
                        [(verts[i], verts[i+1], verts[i+2])
                         for i in range(0, len(verts), 3)], [], faces)
                    obj.data.update()
            else:
                if obj.mode == 'EDIT':
                    bm = bmesh.from_edit_mesh(obj.data)
                    if len(bm.verts) * 3 == len(verts):
                        for i, v in enumerate(bm.verts):
                            v.co.x = verts[i*3]
                            v.co.y = verts[i*3+1]
                            v.co.z = verts[i*3+2]
                        bmesh.update_edit_mesh(obj.data)
                else:
                    obj.data.vertices.foreach_set("co", verts)
                    obj.data.update()

    except Exception as e:
        print(f"[MP] state_to_obj({obj_name}): {e}")


def _apply_objects(objects_data: dict, sync_folder: str, update_baseline=True):
    for name, state in objects_data.items():
        if state.get("deleted"):
            obj = bpy.data.objects.get(name)
            if obj:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass
            if update_baseline:
                with MP.baseline_lock:
                    MP.obj_baseline.pop(name, None)
            continue

        obj = bpy.data.objects.get(name)
        if obj is None:
            otype = state.get("type", "EMPTY")
            try:
                if otype == 'MESH':
                    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
                elif otype == 'LIGHT':
                    obj = bpy.data.objects.new(name,
                              bpy.data.lights.new(name, 'POINT'))
                else:
                    obj = bpy.data.objects.new(name, None)
                bpy.context.collection.objects.link(obj)
            except Exception as e:
                print(f"[MP] create '{name}': {e}")
                continue

        _state_to_obj(name, obj, state, sync_folder)

        if update_baseline:
            with MP.baseline_lock:
                MP.obj_baseline[name] = state

        # Object/material now exist – drain any pending texture for this hash
        tex = state.get("tex")
        if tex:
            h = tex["hash"]
            with MP.pending_lock:
                waiting = list(MP.pending.get(h, []))
            for o, m, i in waiting:
                img = bpy.data.images.get(i)
                if img:
                    _assign_texture(o, m, img)


# ═════════════════════════════════════════════════════════════════════════════
#  NETWORK  utilities
# ═════════════════════════════════════════════════════════════════════════════

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"

def _encode_code(ip, port):
    return base64.b32encode(f"{ip}:{port}".encode()).decode().rstrip("=")

def _decode_code(code):
    code = code.upper().strip() + "=" * ((8 - len(code.upper().strip()) % 8) % 8)
    raw  = base64.b32decode(code).decode()
    ip, p = raw.rsplit(":", 1); return ip, int(p)

def _send(sock, data) -> bool:
    try:
        payload = json.dumps(data).encode()
        sock.sendall(struct.pack(">I", len(payload)) + payload); return True
    except Exception:
        return False

def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: return None
        buf += c
    return buf

def _recv(sock):
    try:
        hdr = _recvall(sock, 4)
        if hdr is None: return None
        length = struct.unpack(">I", hdr)[0]
        if length > 200*1024*1024: return None
        payload = _recvall(sock, length)
        return json.loads(payload.decode()) if payload else None
    except Exception:
        return None

def _broadcast(exclude, msg):
    with MP.conn_lock: targets = list(MP.client_conns.keys())
    for c in targets:
        if c is not exclude: _send(c, msg)


# ═════════════════════════════════════════════════════════════════════════════
#  HOST threads
# ═════════════════════════════════════════════════════════════════════════════

def _host_client_thread(conn, addr):
    peer_name = "???"
    try:
        while MP.connected:
            msg = _recv(conn)
            if msg is None: break
            t = msg.get("type")
            if t == "join":
                peer_name = msg.get("name", "???")[:24]
                colour    = COLOURS[MP.colour_idx % len(COLOURS)]
                MP.colour_idx += 1
                with MP.conn_lock:  MP.client_conns[conn] = {"name": peer_name}
                with MP.peer_lock:  MP.peer_data[peer_name] = {"colour": colour}
                _broadcast(conn, {"type":"peer_joined","name":peer_name,"colour":list(colour)})
                with MP.peer_lock:
                    ex = {n:list(d.get("colour",[1,.5,0,1]))
                          for n,d in MP.peer_data.items() if n!=peer_name}
                _send(conn, {"type":"peer_list","peers":ex})
                with MP.q_lock: MP.in_queue.append({"type":"_peer_join","_conn":conn})
            elif t in ("cursor","camera"):
                msg["name"] = peer_name
                with MP.q_lock: MP.in_queue.append(msg)
                _broadcast(conn, msg)
            elif t in ("delta","full_state"):
                msg["name"] = peer_name
                _broadcast(conn, msg)
                with MP.q_lock: MP.in_queue.append(msg)
            elif t in ("tex_push","tex_manifest"):
                _broadcast(conn, msg)
                with MP.q_lock: MP.in_queue.append(msg)
    except Exception as e:
        print(f"[MP] client-thread: {e}")
    finally:
        with MP.conn_lock:  MP.client_conns.pop(conn, None)
        with MP.peer_lock:  MP.peer_data.pop(peer_name, None)
        try: conn.close()
        except Exception: pass
        _broadcast(None, {"type":"peer_left","name":peer_name})


def _host_accept_thread():
    while MP.connected and MP.server_sock:
        try:
            MP.server_sock.settimeout(1.0)
            conn, addr = MP.server_sock.accept()
            threading.Thread(target=_host_client_thread,
                             args=(conn,addr), daemon=True).start()
        except socket.timeout: continue
        except Exception: break


def _client_recv_thread():
    while MP.connected and MP.my_sock:
        msg = _recv(MP.my_sock)
        if msg is None:
            MP.connected = False; MP.status = "⚠ Lost connection"; break
        with MP.q_lock: MP.in_queue.append(msg)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN THREAD: queue + local state
# ═════════════════════════════════════════════════════════════════════════════

def _process_queue():
    with MP.q_lock:
        msgs = MP.in_queue[:]
        MP.in_queue.clear()

    sf = ""
    try: sf = bpy.context.scene.mp_settings.sync_folder
    except Exception: pass

    for msg in msgs:
        t    = msg.get("type")
        name = msg.get("name","???")

        if t == "peer_joined":
            with MP.peer_lock:
                MP.peer_data.setdefault(name,{})["colour"] = \
                    tuple(msg.get("colour",[1,.5,0,1]))
            MP.status = f"● {name} joined"

        elif t == "peer_list":
            for n,c in msg.get("peers",{}).items():
                with MP.peer_lock:
                    MP.peer_data.setdefault(n,{})["colour"] = tuple(c)

        elif t == "peer_left":
            with MP.peer_lock: MP.peer_data.pop(name,None)
            MP.status = f"● {name} left"

        elif t == "cursor":
            with MP.peer_lock:
                MP.peer_data.setdefault(name,{})["cursor"] = msg.get("pos")

        elif t == "camera":
            with MP.peer_lock:
                d = MP.peer_data.setdefault(name,{})
                d["cam_loc"] = msg.get("loc"); d["cam_rot"] = msg.get("rot")

        elif t in ("delta","full_state"):
            objs = msg.get("objects",{})
            if objs: _apply_objects(objs, sf, update_baseline=True)

        elif t == "tex_push":
            with MP.tex_lock: MP.sent_tex_hashes.add(msg.get("hash",""))
            _install_texture(msg, sf)

        elif t == "tex_manifest":
            extra = msg.get("entries",{})
            if extra and sf:
                existing = _load_manifest(sf)
                existing.update(extra)
                _save_manifest(sf, existing)
            if sf: _replay_manifest(sf)

        elif t == "_peer_join":
            target = msg.get("_conn")
            if not target: continue
            sm = False
            try: sm = bpy.data.scenes[0].mp_settings.sync_mesh
            except Exception: pass
            full = _collect_full(sm)
            _send(target, {"type":"full_state","name":"__host__","objects":full})
            if sf:
                entries = _load_manifest(sf)
                if entries:
                    _send(target, {"type":"tex_manifest","entries":entries})
            pushes = _collect_tex_pushes(skip=set())
            for p in pushes:
                _send(target, p)
                with MP.tex_lock: MP.sent_tex_hashes.add(p["hash"])
            print(f"[MP] sent {len(pushes)} textures to new peer.")


def _send_local_state():
    scene    = bpy.context.scene
    settings = scene.mp_settings
    name     = settings.user_name.strip() or "Anonymous"
    sm       = settings.sync_mesh
    sf       = settings.sync_folder

    c = scene.cursor.location
    cursor_msg = {"type":"cursor","name":name,
                  "pos":[round(c.x,PREC),round(c.y,PREC),round(c.z,PREC)]}

    cam_msg = None
    if MP.cached_region_3d:
        r3d = MP.cached_region_3d
        vm  = r3d.view_matrix.inverted()
        loc = vm.translation; rot = vm.to_quaternion()
        cam_msg = {"type":"camera","name":name,
                   "loc":[round(v,PREC) for v in loc],
                   "rot":[round(v,PREC) for v in [rot.w,rot.x,rot.y,rot.z]]}

    changed = _collect_delta(sm)
    delta_msg = None
    if changed:
        delta_msg = {"type":"delta","name":name,"objects":changed}
        with MP.baseline_lock:
            for n,s in changed.items():
                MP.obj_baseline.pop(n,None) if s.get("deleted") \
                    else MP.obj_baseline.update({n:s})

    tex_msgs = []
    if changed:
        with MP.tex_lock: already = set(MP.sent_tex_hashes)
        for state in changed.values():
            if state.get("deleted"): continue
            tex = state.get("tex")
            if not tex: continue
            h = tex["hash"]
            if not h or h in already: continue
            fp = tex.get("filepath","")
            if not fp: continue
            push = _build_tex_push(tex["img_name"], fp,
                                   tex["mat_name"], tex["obj_name"])
            if push:
                tex_msgs.append(push)
                if sf: _manifest_add(sf, h, tex["img_name"],
                                     tex["mat_name"], tex["obj_name"],
                                     push["filename"])
                with MP.tex_lock: MP.sent_tex_hashes.add(h)

    manifest_msg = None
    if tex_msgs and sf:
        entries = _load_manifest(sf)
        if entries:
            manifest_msg = {"type":"tex_manifest","entries":entries}

    messages = ([m for m in [cursor_msg, cam_msg, delta_msg] if m]
                + tex_msgs + ([manifest_msg] if manifest_msg else []))

    if MP.is_host:
        with MP.conn_lock: conns = list(MP.client_conns.keys())
        for m in messages:
            for c in conns: _send(c, m)
    elif MP.my_sock:
        for m in messages: _send(MP.my_sock, m)


# ─────────────────────────────────────────────────────────────────────────────
#  SYNC TIMER
# ─────────────────────────────────────────────────────────────────────────────

def _sync_tick():
    if not MP.connected:
        MP.timer_active = False; return None

    _process_queue()
    _send_local_state()

    # If any texture was installed this tick, run a full depsgraph update
    # so the viewport shader cache is rebuilt with the new image.
    if MP.needs_depsgraph_update:
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass
        MP.needs_depsgraph_update = False

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D": area.tag_redraw()

    return SYNC_INTERVAL

def _start_timer():
    if not MP.timer_active:
        MP.timer_active = True
        bpy.app.timers.register(_sync_tick, first_interval=SYNC_INTERVAL)

def _stop_timer():
    MP.timer_active = False
    try: bpy.app.timers.unregister(_sync_tick)
    except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
#  VIEWPORT DRAW
# ─────────────────────────────────────────────────────────────────────────────

def _draw_peers():
    ctx = bpy.context
    if not ctx or not ctx.space_data or ctx.space_data.type != "VIEW_3D": return
    MP.cached_region    = ctx.region
    MP.cached_region_3d = ctx.space_data.region_3d
    if not MP.connected: return
    region = ctx.region; rv3d = ctx.region_data
    if not region or not rv3d: return
    my_name = ctx.scene.mp_settings.user_name.strip()
    with MP.peer_lock: snap = {n:dict(d) for n,d in MP.peer_data.items()}
    for n, d in snap.items():
        if n == my_name: continue
        col = d.get("colour",(1,.5,0,1))
        cp  = d.get("cursor")
        if cp:
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(
                     region, rv3d, mathutils.Vector(cp))
            if p2:
                _draw_crosshair(p2, col, CURSOR_R)
                _draw_label(n, p2[0]+CURSOR_R+4, p2[1]-6, col)
        cl = d.get("cam_loc")
        if cl:
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(
                     region, rv3d, mathutils.Vector(cl))
            if p2:
                _draw_circle(p2, col, CAM_R)
                _draw_label(f"{n} [cam]", p2[0]+CAM_R+4, p2[1]-6, (*col[:3],.6))
    pc = len(snap)
    blf.position(0,12,region.height-22,0); _blf_size(0,12)
    blf.color(0,.8,.8,.8,.9)
    blf.draw(0,f"MP  {pc} peer{'s' if pc!=1 else ''} online")

def _blf_size(f,s):
    try: blf.size(f,s,72)
    except TypeError: blf.size(f,s)

def _gpoly(verts,col,mode):
    sh=gpu.shader.from_builtin("UNIFORM_COLOR")
    bat=batch_for_shader(sh,mode,{"pos":verts})
    sh.bind();sh.uniform_float("color",col)
    gpu.state.blend_set("ALPHA");gpu.state.line_width_set(1.8)
    bat.draw(sh);gpu.state.blend_set("NONE")

def _draw_circle(c,col,r,s=20):
    _gpoly([(c[0]+r*math.cos(2*math.pi*i/s),c[1]+r*math.sin(2*math.pi*i/s))
            for i in range(s)],col,"LINE_LOOP")

def _draw_crosshair(c,col,r):
    _draw_circle(c,col,r); cx,cy=c
    sh=gpu.shader.from_builtin("UNIFORM_COLOR")
    bat=batch_for_shader(sh,"LINES",{"pos":[(cx-r,cy),(cx+r,cy),(cx,cy-r),(cx,cy+r)]})
    sh.bind();sh.uniform_float("color",col);gpu.state.line_width_set(1.5);bat.draw(sh)

def _draw_label(txt,x,y,col):
    blf.enable(0,blf.SHADOW);blf.shadow(0,3,0,0,0,.85)
    blf.shadow_offset(0,1,-1);blf.position(0,x,y,0)
    _blf_size(0,LABEL_SIZE);blf.color(0,*col)
    blf.draw(0,txt);blf.disable(0,blf.SHADOW)

def _start_draw():
    if MP.draw_handle is None:
        MP.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_peers,(),"WINDOW","POST_PIXEL")

def _stop_draw():
    if MP.draw_handle:
        try: bpy.types.SpaceView3D.draw_handler_remove(MP.draw_handle,"WINDOW")
        except Exception: pass
        MP.draw_handle = None


# ─────────────────────────────────────────────────────────────────────────────
#  DISCONNECT
# ─────────────────────────────────────────────────────────────────────────────

def _disconnect():
    MP.connected = False; MP.is_host = False
    for s in (MP.server_sock, MP.my_sock):
        if s:
            try: s.close()
            except Exception: pass
    MP.server_sock = MP.my_sock = None
    with MP.conn_lock:
        for c in list(MP.client_conns):
            try: c.close()
            except Exception: pass
        MP.client_conns.clear()
    with MP.peer_lock:    MP.peer_data.clear()
    with MP.q_lock:       MP.in_queue.clear()
    with MP.baseline_lock: MP.obj_baseline.clear()
    with MP.tex_lock:     MP.sent_tex_hashes.clear()
    with MP.pending_lock: MP.pending.clear()
    _stop_timer(); _stop_draw()
    MP.status = "Disconnected"; MP.colour_idx = 0


# ═════════════════════════════════════════════════════════════════════════════
#  OPERATORS
# ═════════════════════════════════════════════════════════════════════════════

class MP_OT_Host(Operator):
    bl_idname="mp.host"; bl_label="Host Session"; bl_options={"REGISTER"}
    def execute(self,ctx):
        if not ctx.scene.mp_settings.user_name.strip():
            self.report({"ERROR"},"Enter your name first!"); return{"CANCELLED"}
        if MP.connected:
            self.report({"WARNING"},"Disconnect first."); return{"CANCELLED"}
        try:
            srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            srv.bind(("",DEFAULT_PORT)); srv.listen(8)
            MP.server_sock=srv; MP.is_host=True; MP.connected=True
            with MP.baseline_lock:
                MP.obj_baseline=_collect_full(ctx.scene.mp_settings.sync_mesh)
            code=_encode_code(_get_local_ip(),DEFAULT_PORT)
            ctx.scene.mp_settings.room_code=code
            MP.status="Hosting  •  0 peers"
            threading.Thread(target=_host_accept_thread,daemon=True).start()
            _start_timer(); _start_draw()
            sf=ctx.scene.mp_settings.sync_folder
            if sf: _replay_manifest(sf)
            self.report({"INFO"},f"Hosting!  Code: {code}")
        except Exception as e:
            self.report({"ERROR"},str(e)); return{"CANCELLED"}
        return{"FINISHED"}


class MP_OT_Join(Operator):
    bl_idname="mp.join"; bl_label="Join Session"; bl_options={"REGISTER"}
    def execute(self,ctx):
        S=ctx.scene.mp_settings
        if not S.user_name.strip() or not S.room_code.strip():
            self.report({"ERROR"},"Enter name and room code!"); return{"CANCELLED"}
        if MP.connected:
            self.report({"WARNING"},"Disconnect first."); return{"CANCELLED"}
        try: ip,port=_decode_code(S.room_code.strip())
        except Exception:
            self.report({"ERROR"},"Invalid room code."); return{"CANCELLED"}
        try:
            sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            sock.settimeout(6.0); sock.connect((ip,port)); sock.settimeout(None)
            MP.my_sock=sock; MP.is_host=False; MP.connected=True
            MP.status="● Connected"
            _send(sock,{"type":"join","name":S.user_name.strip()})
            threading.Thread(target=_client_recv_thread,daemon=True).start()
            _start_timer(); _start_draw()
            if S.sync_folder: _replay_manifest(S.sync_folder)
            self.report({"INFO"},"Joined!")
        except Exception as e:
            self.report({"ERROR"},f"Could not connect: {e}"); return{"CANCELLED"}
        return{"FINISHED"}


class MP_OT_Disconnect(Operator):
    bl_idname="mp.disconnect"; bl_label="Disconnect"; bl_options={"REGISTER"}
    def execute(self,ctx): _disconnect(); return{"FINISHED"}


class MP_OT_CopyCode(Operator):
    bl_idname="mp.copy_code"; bl_label="Copy"; bl_options={"REGISTER"}
    def execute(self,ctx):
        ctx.window_manager.clipboard=ctx.scene.mp_settings.room_code
        return{"FINISHED"}


class MP_OT_SnapCam(Operator):
    bl_idname="mp.snap_cam"; bl_label="Snap to camera"; bl_options={"REGISTER"}
    peer_name: StringProperty(default="")
    def execute(self,ctx):
        with MP.peer_lock: d=dict(MP.peer_data.get(self.peer_name,{}))
        loc,rot=d.get("cam_loc"),d.get("cam_rot")
        if not loc or not rot: return{"CANCELLED"}
        r3d=ctx.space_data.region_3d
        r3d.view_location=mathutils.Vector(loc)
        r3d.view_rotation=mathutils.Quaternion([rot[0],rot[1],rot[2],rot[3]])
        return{"FINISHED"}


class MP_OT_PushAll(Operator):
    """Force-push the full scene + every texture to all peers."""
    bl_idname="mp.push_all"; bl_label="Push Full Scene + Textures"; bl_options={"REGISTER"}
    def execute(self,ctx):
        if not MP.connected: return{"CANCELLED"}
        my_name=ctx.scene.mp_settings.user_name.strip() or "Anonymous"
        sf=ctx.scene.mp_settings.sync_folder
        with MP.tex_lock: MP.sent_tex_hashes.clear()
        pushes=_collect_tex_pushes(skip=set())
        with MP.tex_lock:
            for p in pushes:
                MP.sent_tex_hashes.add(p["hash"])
                if sf: _manifest_add(sf,p["hash"],p["img_name"],
                                     p["mat_name"],p["obj_name"],p["filename"])
        full=_collect_full(ctx.scene.mp_settings.sync_mesh)
        msgs=[{"type":"full_state","name":my_name,"objects":full}]+pushes
        if sf:
            entries=_load_manifest(sf)
            if entries: msgs.append({"type":"tex_manifest","entries":entries})
        if MP.is_host:
            with MP.conn_lock: conns=list(MP.client_conns.keys())
            for m in msgs:
                for c in conns: _send(c,m)
        elif MP.my_sock:
            for m in msgs: _send(MP.my_sock,m)
        with MP.baseline_lock: MP.obj_baseline=dict(full)
        self.report({"INFO"},f"Pushed {len(full)} objects + {len(pushes)} textures.")
        return{"FINISHED"}


class MP_OT_ReplayManifest(Operator):
    """Re-apply all textures from the manifest right now."""
    bl_idname="mp.replay_manifest"; bl_label="Re-apply Textures from Manifest"
    bl_options={"REGISTER"}
    def execute(self,ctx):
        sf=ctx.scene.mp_settings.sync_folder
        if not sf:
            self.report({"WARNING"},"No Texture Folder set."); return{"CANCELLED"}
        _replay_manifest(sf)
        self.report({"INFO"},"Manifest re-applied.")
        return{"FINISHED"}


# ═════════════════════════════════════════════════════════════════════════════
#  PROPERTIES  &  UI
# ═════════════════════════════════════════════════════════════════════════════

class MPSettings(PropertyGroup):
    user_name:   StringProperty(name="Your Name", default="", maxlen=24)
    room_code:   StringProperty(name="Room Code", default="")
    sync_mesh:   BoolProperty(
        name="Sync Mesh Shapes",
        description="Sync vertex positions live (slow on high-poly meshes)",
        default=False)
    sync_folder: StringProperty(
        name="Texture Folder", subtype='DIR_PATH',
        description="Where textures + mp_textures.json are saved.\nSet BEFORE connecting.",
        default="")


class MP_PT_Main(Panel):
    bl_label="Multiplayer"; bl_idname="MP_PT_Main"
    bl_space_type="VIEW_3D"; bl_region_type="UI"; bl_category="Multiplayer"

    def draw(self,ctx):
        L=self.layout; S=ctx.scene.mp_settings
        b=L.box()
        b.label(text=MP.status,
                icon="RADIOBUT_ON" if MP.connected else "RADIOBUT_OFF")
        L.separator(factor=0.4)
        L.label(text="Identity",icon="USER")
        L.prop(S,"user_name",text="Name")
        L.separator(factor=0.4)
        L.label(text="Options",icon="MODIFIER")
        L.prop(S,"sync_mesh")
        L.label(text="Texture Folder  (mp_textures.json saved here):",icon="FILE_FOLDER")
        L.prop(S,"sync_folder",text="")
        L.separator()
        if not MP.connected:
            hb=L.box(); hb.label(text="Start a Session",icon="SOLO_ON")
            r=hb.row(); r.scale_y=1.35; r.operator("mp.host",icon="SOLO_ON")
            L.separator(factor=0.3)
            jb=L.box(); jb.label(text="Join a Session",icon="LINKED")
            jb.prop(S,"room_code",text="Code")
            r2=jb.row(); r2.scale_y=1.35; r2.operator("mp.join",icon="LINKED")
        else:
            if MP.is_host:
                cb=L.box(); cb.label(text="Share this code:",icon="KEY_HLT")
                r2=cb.row(align=True)
                r2.prop(S,"room_code",text="")
                r2.operator("mp.copy_code",icon="COPYDOWN",text="Copy")
                L.separator(factor=0.3)
            pb=L.box(); pb.label(text="Online",icon="COMMUNITY")
            pb.label(text=f"  You  ({S.user_name.strip() or 'Anonymous'})",icon="FUND")
            with MP.peer_lock: snap=dict(MP.peer_data)
            for pn in snap:
                pr=pb.row(align=True)
                pr.label(text=f"  ● {pn}",icon="FUND")
                op=pr.operator("mp.snap_cam",text="",icon="VIEW_CAMERA")
                op.peer_name=pn
            if not snap: pb.label(text="  Waiting for peers…",icon="TIME")
            L.separator(factor=0.3)
            L.operator("mp.push_all",        icon="EXPORT")
            L.operator("mp.replay_manifest", icon="FILE_REFRESH")
            dr=L.row(); dr.alert=True; dr.scale_y=1.2
            dr.operator("mp.disconnect",icon="X")


_CLASSES=(
    MPSettings, MP_OT_Host, MP_OT_Join, MP_OT_Disconnect,
    MP_OT_CopyCode, MP_OT_SnapCam, MP_OT_PushAll,
    MP_OT_ReplayManifest, MP_PT_Main,
)

def register():
    for c in _CLASSES: bpy.utils.register_class(c)
    bpy.types.Scene.mp_settings=bpy.props.PointerProperty(type=MPSettings)
    print("[MP] Blender Multiplayer Sync v5.1 registered.")

def unregister():
    _disconnect()
    for c in reversed(_CLASSES): bpy.utils.unregister_class(c)
    del bpy.types.Scene.mp_settings
    print("[MP] Blender Multiplayer Sync unregistered.")

if __name__=="__main__": register()
