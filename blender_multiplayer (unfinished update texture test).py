# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          BLENDER MULTIPLAYER SYNC  v4.2  –  blender_multiplayer.py     ║
# ║  Full bidirectional sync: objects · cursor · camera · meshes           ║
# ║  Proactive texture push · manifest map · node connect · keyframes      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# KEY FIX in v4.2
#   Textures were downloaded but never CONNECTED to the material node tree.
#   _connect_image_to_material() now:
#     • Finds or creates a TEX_IMAGE node in the material
#     • Sets node.image to the loaded image datablock
#     • Connects it to the first available color input
#       (Principled BSDF Base Color  OR  Emission Color)
#     • Handles the "Add Image as Plane" Emission setup specifically
#   It is called:
#     • inside _install_texture()   – right after saving the file
#     • inside _apply_manifest()    – when repairing missing files
#     • inside _state_to_obj()      – if the image is already in bpy.data.images
#   Pending connections (image not yet downloaded) are stored in
#   MP.pending_tex_connects and resolved on the next _install_texture() call.
#
# MANIFEST  mp_textures.json  –  lives in your Texture Folder.
# NETWORK   LAN out of box (TCP 19283).  Internet: port-forward or ZeroTier.

bl_info = {
    "name":        "Blender Multiplayer Sync",
    "author":      "Claude & LAGkit",
    "version":     (4, 2, 0),
    "blender":     (3, 0, 0),
    "location":    "View3D > N-Panel > Multiplayer",
    "description": "Real-time collaboration – objects, meshes, materials, textures, keyframes",
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

DEFAULT_PORT      = 19283
SYNC_INTERVAL     = 0.05
LABEL_SIZE        = 14
CURSOR_R          = 10
CAM_R             = 7
PREC              = 5
TEX_SIZE_LIMIT_MB = 32
MANIFEST_FILENAME = "mp_textures.json"

COLOURS = [
    (1.00, 0.42, 0.12, 1.0), (0.18, 0.85, 0.42, 1.0),
    (0.20, 0.55, 1.00, 1.0), (0.95, 0.15, 0.60, 1.0),
    (1.00, 0.90, 0.10, 1.0), (0.55, 0.20, 1.00, 1.0),
    (0.10, 0.88, 0.90, 1.0),
]


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

class _MP:
    server_sock     = None
    my_sock         = None
    client_conns    = {}
    conn_lock       = threading.Lock()

    peer_data       = {}
    peer_lock       = threading.Lock()

    in_queue        = []
    q_lock          = threading.Lock()

    obj_baseline    = {}
    baseline_lock   = threading.Lock()

    sent_tex_hashes = set()
    tex_lock        = threading.Lock()

    # { mat_name: image_name }  – materials that need a texture once it arrives
    pending_tex_connects = {}
    pending_lock         = threading.Lock()

    is_host         = False
    connected       = False
    status          = "Disconnected"
    colour_idx      = 0

    draw_handle     = None
    timer_active    = False

    cached_region    = None
    cached_region_3d = None

MP = _MP()


# ═════════════════════════════════════════════════════════════════════════════
#  TEXTURE  NODE  CONNECTION  (the core fix)
# ═════════════════════════════════════════════════════════════════════════════

def _connect_image_to_material(mat_name: str, img: "bpy.types.Image"):
    """
    Find the material, locate or create a TEX_IMAGE node, assign img,
    and wire it to whichever color socket is available:
      • Principled BSDF  → Base Color
      • Emission         → Color          (Add Image as Plane default)
      • Any other shader → first Color/RGB input found
    Safe to call multiple times (idempotent).
    """
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        return

    if not mat.use_nodes:
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # ── Find or create TEX_IMAGE node ────────────────────────────────────────
    tex_node = next((n for n in nodes if n.type == 'TEX_IMAGE'), None)
    if tex_node is None:
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-400, 300)

    # Always assign / reassign the image
    tex_node.image = img
    try:
        img.reload()
    except Exception:
        pass

    # ── Find the target shader node and its color input ───────────────────────
    # Priority: Principled → Emission → any BSDF → any node with Color input
    shader = (
        next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None) or
        next((n for n in nodes if n.type == 'EMISSION'),         None) or
        next((n for n in nodes if n.bl_idname.startswith('ShaderNodeBsdf')), None)
    )

    if shader is None:
        print(f"[MP] _connect_image: no shader node found in '{mat_name}'")
        return

    # Pick the right socket name
    for socket_name in ('Base Color', 'Color', 'Emission Color'):
        if socket_name in shader.inputs:
            color_input = shader.inputs[socket_name]
            break
    else:
        # Fall back: first socket that accepts color
        color_input = next(
            (s for s in shader.inputs
             if s.type in ('RGBA', 'VECTOR') and s.name.lower() != 'normal'),
            None,
        )

    if color_input is None:
        print(f"[MP] _connect_image: no suitable color input in '{mat_name}'")
        return

    # Remove existing links on that socket then connect
    for lnk in list(color_input.links):
        links.remove(lnk)
    links.new(tex_node.outputs['Color'], color_input)
    print(f"[MP] Connected '{img.name}' → '{mat_name}' ({color_input.name})")


def _resolve_pending_connects():
    """
    After any image is loaded, check whether there are pending material
    connections waiting for it and apply them.
    """
    with MP.pending_lock:
        pending = dict(MP.pending_tex_connects)

    resolved = []
    for mat_name, img_name in pending.items():
        img = bpy.data.images.get(img_name)
        if img and img.filepath and os.path.exists(bpy.path.abspath(img.filepath)):
            _connect_image_to_material(mat_name, img)
            resolved.append(mat_name)

    if resolved:
        with MP.pending_lock:
            for k in resolved:
                MP.pending_tex_connects.pop(k, None)


def _queue_pending_connect(mat_name: str, img_name: str):
    """Register a material→image connection to be made once the image arrives."""
    with MP.pending_lock:
        MP.pending_tex_connects[mat_name] = img_name


# ═════════════════════════════════════════════════════════════════════════════
#  TEXTURE  MANIFEST
# ═════════════════════════════════════════════════════════════════════════════

def _manifest_path(sync_folder: str) -> str:
    return os.path.join(bpy.path.abspath(sync_folder), MANIFEST_FILENAME)


def _load_manifest(sync_folder: str) -> dict:
    path = _manifest_path(sync_folder)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("entries", {})
    except Exception as e:
        print(f"[MP] manifest read: {e}")
        return {}


def _save_manifest(sync_folder: str, entries: dict):
    path = _manifest_path(sync_folder)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": entries}, f, indent=2)
    except Exception as e:
        print(f"[MP] manifest write: {e}")


def _add_to_manifest(sync_folder: str, tex_hash: str, entry: dict):
    entries = _load_manifest(sync_folder)
    entries[tex_hash] = entry
    _save_manifest(sync_folder, entries)


def _apply_manifest(sync_folder: str, extra_entries: dict = None):
    """
    1. For every image in bpy.data.images with a missing file, search the
       manifest (+ sync folder filesystem) and redirect its filepath.
    2. After fixing a filepath, call _connect_image_to_material() so the
       node tree actually uses the newly found image.
    3. Resolve any pending connections that were waiting for the image.
    """
    if not sync_folder:
        return
    folder = bpy.path.abspath(sync_folder)
    if not os.path.isdir(folder):
        return

    entries = _load_manifest(sync_folder)
    if extra_entries:
        entries.update(extra_entries)

    # ── Build lookup tables from manifest ────────────────────────────────────
    by_image_name = {}   # img_name           → abs_save_path
    by_name_stem  = {}   # stem(img_name)      → abs_save_path
    by_path_stem  = {}   # stem(original_path) → abs_save_path
    by_mat_name   = {}   # mat_name            → (abs_save_path, img_name)

    for h, e in entries.items():
        fname = e.get("filename", "")
        if not fname:
            continue
        abs_path = os.path.join(folder, fname)
        if not os.path.exists(abs_path):
            continue
        iname = e.get("image_name", "")
        orig  = e.get("original_path", "")
        mat   = e.get("mat_name", "")
        if iname:
            by_image_name[iname]                              = abs_path
            by_name_stem[os.path.splitext(iname)[0].lower()] = abs_path
        if orig:
            by_path_stem[
                os.path.splitext(os.path.basename(orig))[0].lower()] = abs_path
        if mat:
            by_mat_name[mat] = (abs_path, iname)

    # ── Filesystem fallback ───────────────────────────────────────────────────
    fs_by_stem = {}
    try:
        for fname in os.listdir(folder):
            if fname == MANIFEST_FILENAME:
                continue
            stem_plain = fname.split("__")[0].lower()
            stem_full  = os.path.splitext(fname)[0].lower()
            fpath      = os.path.join(folder, fname)
            fs_by_stem[stem_plain] = fpath
            fs_by_stem[stem_full]  = fpath
    except Exception:
        pass

    def _find_path(img):
        c = by_image_name.get(img.name)
        if c: return c
        c = by_name_stem.get(os.path.splitext(img.name)[0].lower())
        if c: return c
        if img.filepath:
            base = os.path.splitext(
                os.path.basename(bpy.path.abspath(img.filepath)))[0].lower()
            c = by_path_stem.get(base) or by_name_stem.get(base)
            if c: return c
        stem = os.path.splitext(img.name)[0].lower()
        return fs_by_stem.get(stem)

    # ── Repair missing images ─────────────────────────────────────────────────
    for img in bpy.data.images:
        if img.source not in ('FILE', 'SEQUENCE', 'MOVIE'):
            continue
        if img.filepath and os.path.exists(bpy.path.abspath(img.filepath)):
            continue                                   # already fine

        candidate = _find_path(img)
        if not candidate:
            continue

        img.filepath = candidate
        try:
            img.reload()
            print(f"[MP] repaired '{img.name}' → {candidate}")
        except Exception as e:
            print(f"[MP] reload '{img.name}': {e}")
            continue

        # Re-connect in every material that owns this image
        for mat in bpy.data.materials:
            if not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image == img:
                    _connect_image_to_material(mat.name, img)
                    break

    # ── Connect via mat_name lookup (for receiver who has mat but no node) ────
    for mat_name, (abs_path, img_name) in by_mat_name.items():
        img = bpy.data.images.get(img_name)
        if img is None:
            # Try loading from path
            if os.path.exists(abs_path):
                try:
                    img = bpy.data.images.load(abs_path)
                    img.name = img_name
                except Exception:
                    pass
        if img:
            _connect_image_to_material(mat_name, img)

    _resolve_pending_connects()


# ─────────────────────────────────────────────────────────────────────────────
#  TEXTURE  PUSH  helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tex_file_hash(filepath: str) -> str:
    try:
        st = os.stat(filepath)
        return f"{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        return ""


def _tex_save_filename(img_name: str, tex_hash: str, ext: str) -> str:
    safe = img_name.replace("/", "_").replace("\\", "_")
    return f"{safe}__{tex_hash}{ext}"


def _build_tex_push(img_name: str, filepath: str, mat_name: str = "") -> dict:
    try:
        size  = os.path.getsize(filepath)
        limit = TEX_SIZE_LIMIT_MB * 1024 * 1024
        if size > limit:
            print(f"[MP] skip '{img_name}' – {size//1024//1024} MB > limit.")
            return None
        tex_hash = _tex_file_hash(filepath)
        ext      = os.path.splitext(filepath)[1] or ".png"
        filename = _tex_save_filename(img_name, tex_hash, ext)
        with open(filepath, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type": "tex_push", "image_name": img_name,
            "hash": tex_hash, "ext": ext, "filename": filename,
            "original_path": filepath, "mat_name": mat_name,
            "data_b64": data_b64,
        }
    except Exception as e:
        print(f"[MP] _build_tex_push({img_name}): {e}")
        return None


def _collect_all_tex_pushes(skip_hashes: set) -> list:
    msgs, seen = [], set()
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != 'TEX_IMAGE' or not node.image:
                continue
            img      = node.image
            filepath = bpy.path.abspath(img.filepath) if img.filepath else ""
            if not filepath or not os.path.exists(filepath):
                continue
            h = _tex_file_hash(filepath)
            if not h or h in skip_hashes or h in seen:
                continue
            seen.add(h)
            push = _build_tex_push(img.name, filepath, mat.name)
            if push:
                msgs.append(push)
    return msgs


def _install_texture(msg: dict, sync_folder: str):
    """
    Save the received texture binary, update the manifest, load the image
    datablock, and immediately connect it to the material node tree.
    """
    if not sync_folder:
        print("[MP] No texture folder set – texture not saved.")
        return

    folder = bpy.path.abspath(sync_folder)
    os.makedirs(folder, exist_ok=True)

    img_name  = msg["image_name"]
    tex_hash  = msg["hash"]
    ext       = msg.get("ext", ".png")
    filename  = msg.get("filename") or _tex_save_filename(img_name, tex_hash, ext)
    mat_name  = msg.get("mat_name", "")
    save_path = os.path.join(folder, filename)

    # ── Save binary ───────────────────────────────────────────────────────────
    if not os.path.exists(save_path):
        try:
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(msg["data_b64"]))
            print(f"[MP] saved texture → {save_path}")
        except Exception as e:
            print(f"[MP] _install_texture write: {e}")
            return

    # ── Update manifest ───────────────────────────────────────────────────────
    _add_to_manifest(sync_folder, tex_hash, {
        "image_name":    img_name,
        "filename":      filename,
        "ext":           ext,
        "original_path": msg.get("original_path", ""),
        "mat_name":      mat_name,
    })

    # ── Load / update image datablock ─────────────────────────────────────────
    img = bpy.data.images.get(img_name)
    if img is None:
        try:
            img = bpy.data.images.load(save_path)
            img.name = img_name
        except Exception as e:
            print(f"[MP] load image '{img_name}': {e}")
            return
    else:
        img.filepath = save_path
        try:
            img.reload()
        except Exception:
            pass

    # ── Connect to material node ───────────────────────────────────────────────
    # 1. Use mat_name from the push message (most precise)
    if mat_name:
        mat = bpy.data.materials.get(mat_name)
        if mat:
            _connect_image_to_material(mat_name, img)
        else:
            _queue_pending_connect(mat_name, img_name)

    # 2. Also scan all materials for any TEX_IMAGE node that has
    #    the same image name but is still pointing to a missing path
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != 'TEX_IMAGE':
                continue
            # Node has no image, or image with same name, or missing file
            node_img = node.image
            if node_img is None:
                continue
            if node_img.name == img_name:
                node.image = img          # update to freshly loaded datablock
                _connect_image_to_material(mat.name, img)
                break
            # Also check if node_img has same filepath stem (cross-machine case)
            if node_img.filepath:
                abs_fp = bpy.path.abspath(node_img.filepath)
                if not os.path.exists(abs_fp):
                    stem_node = os.path.splitext(
                        os.path.basename(abs_fp))[0].lower()
                    stem_new  = os.path.splitext(img_name)[0].lower()
                    if stem_node == stem_new:
                        node.image = img
                        _connect_image_to_material(mat.name, img)
                        break

    # 3. Resolve any previously queued pending connections
    _resolve_pending_connects()

    # 4. Full manifest repair in case other images also became findable
    _apply_manifest(sync_folder)


# ═════════════════════════════════════════════════════════════════════════════
#  OBJECT STATE  helpers
# ═════════════════════════════════════════════════════════════════════════════

def _obj_to_state(obj, sync_mesh: bool = False) -> dict:
    loc  = [round(v, PREC) for v in obj.location]
    mode = obj.rotation_mode
    rot  = ([round(v, PREC) for v in obj.rotation_quaternion]
            if mode == 'QUATERNION' else
            [round(v, PREC) for v in obj.rotation_axis_angle]
            if mode == 'AXIS_ANGLE' else
            [round(v, PREC) for v in obj.rotation_euler])
    scale = [round(v, PREC) for v in obj.scale]
    state = {"type": obj.type, "loc": loc, "rot_mode": mode,
             "rot": rot, "scale": scale}

    if obj.active_material:
        mat = obj.active_material
        state["mat_name"] = mat.name
        if mat.use_nodes:
            try:
                nodes = mat.node_tree.nodes
                bsdf  = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf:
                    col = bsdf.inputs["Base Color"].default_value
                    state["mat_color"] = [round(v, PREC) for v in col]
                tex_node = next(
                    (n for n in nodes if n.type == 'TEX_IMAGE' and n.image), None)
                if tex_node:
                    img      = tex_node.image
                    filepath = bpy.path.abspath(img.filepath) if img.filepath else ""
                    if filepath and os.path.exists(filepath):
                        state["tex_name"]       = img.name
                        state["tex_hash"]       = _tex_file_hash(filepath)
                        state["tex_ext"]        = os.path.splitext(filepath)[1] or ".png"
                        state["tex_path_local"] = filepath
                        state["tex_mat"]        = mat.name
            except Exception:
                pass

    if obj.animation_data and obj.animation_data.action:
        try:
            act = obj.animation_data.action
            curves = [{"path": fc.data_path, "idx": fc.array_index,
                       "kps": [(round(kp.co[0], PREC), round(kp.co[1], PREC))
                               for kp in fc.keyframe_points]}
                      for fc in act.fcurves]
            state["anim"] = {"name": act.name, "curves": curves}
        except Exception as e:
            print(f"[MP] anim serialize ({obj.name}): {e}")

    if sync_mesh and obj.type == 'MESH':
        try:
            if threading.current_thread() is threading.main_thread():
                if obj.mode == 'EDIT':
                    obj.update_from_editmode()
            n = len(obj.data.vertices)
            if n:
                verts = [0.0] * (n * 3)
                obj.data.vertices.foreach_get("co", verts)
                state["verts"] = [round(v, PREC) for v in verts]
                state["faces"] = [list(p.vertices) for p in obj.data.polygons]
        except Exception as e:
            print(f"[MP] mesh serialize ({obj.name}): {e}")

    return state


def _anim_sig(anim: dict) -> str:
    curves = anim.get("curves", [])
    return "|".join(
        f"{fc['path']}[{fc['idx']}]:{len(fc['kps'])}:"
        f"{fc['kps'][-1][0]:.2f}:{fc['kps'][-1][1]:.4f}"
        if fc.get("kps") else f"{fc['path']}[{fc['idx']}]:0"
        for fc in curves
    )


def _state_key(s: dict) -> tuple:
    return (
        tuple(s.get("loc",   [])),
        s.get("rot_mode", ""),
        tuple(s.get("rot",   [])),
        tuple(s.get("scale", [])),
        hash(tuple(s["verts"])) if "verts"     in s else 0,
        tuple(s["mat_color"])   if "mat_color" in s else (),
        s.get("tex_hash", ""),
        _anim_sig(s["anim"])    if "anim"      in s else "",
        s.get("deleted", False),
    )


def _collect_full_scene(sync_mesh: bool = False) -> dict:
    return {obj.name: _obj_to_state(obj, sync_mesh) for obj in bpy.data.objects}


def _collect_delta(sync_mesh: bool = False) -> dict:
    changed = {}
    current = {obj.name for obj in bpy.data.objects}
    with MP.baseline_lock:
        baseline = dict(MP.obj_baseline)
    for name, state in baseline.items():
        if name not in current and not state.get("deleted"):
            changed[name] = {"deleted": True}
    for obj in bpy.data.objects:
        state = _obj_to_state(obj, sync_mesh)
        prev  = baseline.get(obj.name)
        if prev is None or _state_key(prev) != _state_key(state):
            changed[obj.name] = state
    return changed


# ─────────────────────────────────────────────────────────────────────────────
#  APPLY STATE
# ─────────────────────────────────────────────────────────────────────────────

def _apply_keyframes(obj, anim: dict):
    act_name = anim.get("name", "")
    if not act_name:
        return
    if not obj.animation_data:
        obj.animation_data_create()
    act = bpy.data.actions.get(act_name) or bpy.data.actions.new(act_name)
    if obj.animation_data.action is not act:
        obj.animation_data.action = act

    incoming = {(fc["path"], fc["idx"]) for fc in anim.get("curves", [])}
    for fc in list(act.fcurves):
        if (fc.data_path, fc.array_index) not in incoming:
            act.fcurves.remove(fc)

    for fc_data in anim.get("curves", []):
        path, idx, kps = fc_data["path"], fc_data["idx"], fc_data["kps"]
        existing = act.fcurves.find(path, index=idx)
        if existing:
            act.fcurves.remove(existing)
        if not kps:
            continue
        fc = act.fcurves.new(path, index=idx)
        fc.keyframe_points.add(len(kps))
        for i, (frame, value) in enumerate(kps):
            kp = fc.keyframe_points[i]
            kp.co_ui             = (frame, value)
            kp.interpolation     = 'BEZIER'
            kp.handle_left_type  = 'AUTO'
            kp.handle_right_type = 'AUTO'
        fc.update()


def _state_to_obj(obj, state: dict):
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
            mat = (bpy.data.materials.get(state["mat_name"])
                   or bpy.data.materials.new(state["mat_name"]))
            if not mat.use_nodes:
                mat.use_nodes = True
            if not obj.material_slots:
                obj.data.materials.append(mat)
            elif obj.material_slots[0].material != mat:
                obj.material_slots[0].material = mat
            if "mat_color" in state:
                try:
                    bsdf = next((n for n in mat.node_tree.nodes
                                 if n.type == 'BSDF_PRINCIPLED'), None)
                    if bsdf:
                        bsdf.inputs["Base Color"].default_value = state["mat_color"]
                except Exception:
                    pass

            # Texture: connect if image already loaded, else queue
            if "tex_name" in state:
                img_name = state["tex_name"]
                mat_name = state["mat_name"]
                img = bpy.data.images.get(img_name)
                if img and img.filepath and os.path.exists(
                        bpy.path.abspath(img.filepath)):
                    _connect_image_to_material(mat_name, img)
                else:
                    _queue_pending_connect(mat_name, img_name)

        # Keyframes
        if "anim" in state and state["anim"]:
            _apply_keyframes(obj, state["anim"])

        # Mesh geometry
        if "verts" in state and obj.type == 'MESH':
            verts, faces = state["verts"], state.get("faces", [])
            n_verts = len(verts) // 3
            if (n_verts != len(obj.data.vertices)
                    or len(faces) != len(obj.data.polygons)):
                if obj.mode != 'EDIT':
                    mesh = obj.data
                    mesh.clear_geometry()
                    mesh.from_pydata(
                        [(verts[i], verts[i+1], verts[i+2])
                         for i in range(0, len(verts), 3)], [], faces)
                    mesh.update()
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
        print(f"[MP] _state_to_obj({obj.name}): {e}")


def _apply_objects(objects_data: dict, update_baseline: bool = True):
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
            obj_type = state.get("type", "EMPTY")
            try:
                if obj_type == 'MESH':
                    mesh = bpy.data.meshes.new(name)
                    obj  = bpy.data.objects.new(name, mesh)
                elif obj_type == 'LIGHT':
                    light = bpy.data.lights.new(name, 'POINT')
                    obj   = bpy.data.objects.new(name, light)
                else:
                    obj = bpy.data.objects.new(name, None)
                bpy.context.collection.objects.link(obj)
            except Exception as e:
                print(f"[MP] create '{name}': {e}")
                continue

        _state_to_obj(obj, state)
        if update_baseline:
            with MP.baseline_lock:
                MP.obj_baseline[name] = state


# ═════════════════════════════════════════════════════════════════════════════
#  NETWORK  utilities
# ═════════════════════════════════════════════════════════════════════════════

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"

def _encode_code(ip, port):
    return base64.b32encode(f"{ip}:{port}".encode()).decode().rstrip("=")

def _decode_code(code):
    code = code.upper().strip()
    code += "=" * ((8 - len(code) % 8) % 8)
    raw  = base64.b32decode(code).decode()
    ip, port_str = raw.rsplit(":", 1)
    return ip, int(port_str)

def _send(sock, data: dict) -> bool:
    try:
        payload = json.dumps(data).encode("utf-8")
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        return True
    except Exception:
        return False

def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: return None
        buf += chunk
    return buf

def _recv(sock):
    try:
        hdr = _recvall(sock, 4)
        if hdr is None: return None
        length = struct.unpack(">I", hdr)[0]
        if length > 200 * 1024 * 1024: return None
        payload = _recvall(sock, length)
        return json.loads(payload.decode("utf-8")) if payload else None
    except Exception:
        return None

def _broadcast(exclude, msg):
    with MP.conn_lock:
        targets = list(MP.client_conns.keys())
    for c in targets:
        if c is not exclude:
            _send(c, msg)


# ═════════════════════════════════════════════════════════════════════════════
#  HOST  threads
# ═════════════════════════════════════════════════════════════════════════════

def _host_client_thread(conn, addr):
    peer_name = "???"
    try:
        while MP.connected:
            msg = _recv(conn)
            if msg is None: break
            mtype = msg.get("type")

            if mtype == "join":
                peer_name = msg.get("name", "???")[:24]
                colour    = COLOURS[MP.colour_idx % len(COLOURS)]
                MP.colour_idx += 1
                with MP.conn_lock:
                    MP.client_conns[conn] = {"name": peer_name}
                with MP.peer_lock:
                    MP.peer_data[peer_name] = {"colour": colour}
                _broadcast(conn, {"type": "peer_joined", "name": peer_name,
                                  "colour": list(colour)})
                with MP.peer_lock:
                    existing = {n: list(d.get("colour", [1,.5,0,1]))
                                for n, d in MP.peer_data.items() if n != peer_name}
                _send(conn, {"type": "peer_list", "peers": existing})
                with MP.q_lock:
                    MP.in_queue.append({"type": "_on_peer_join", "_conn": conn})

            elif mtype in ("cursor", "camera"):
                msg["name"] = peer_name
                with MP.q_lock: MP.in_queue.append(msg)
                _broadcast(conn, msg)

            elif mtype in ("objects", "full_state"):
                msg["name"] = peer_name
                _broadcast(conn, msg)
                with MP.q_lock: MP.in_queue.append(msg)

            elif mtype in ("tex_push", "tex_manifest"):
                _broadcast(conn, msg)
                with MP.q_lock: MP.in_queue.append(msg)

    except Exception as e:
        print(f"[MP] client-thread: {e}")
    finally:
        with MP.conn_lock: MP.client_conns.pop(conn, None)
        with MP.peer_lock: MP.peer_data.pop(peer_name, None)
        try: conn.close()
        except Exception: pass
        _broadcast(None, {"type": "peer_left", "name": peer_name})


def _host_accept_thread():
    while MP.connected and MP.server_sock:
        try:
            MP.server_sock.settimeout(1.0)
            conn, addr = MP.server_sock.accept()
            threading.Thread(target=_host_client_thread,
                             args=(conn, addr), daemon=True).start()
        except socket.timeout: continue
        except Exception: break


# ─────────────────────────────────────────────────────────────────────────────
#  CLIENT  receive thread
# ─────────────────────────────────────────────────────────────────────────────

def _client_recv_thread():
    while MP.connected and MP.my_sock:
        msg = _recv(MP.my_sock)
        if msg is None:
            MP.connected = False; MP.status = "⚠ Lost connection"; break
        with MP.q_lock: MP.in_queue.append(msg)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN-THREAD:  process queue  +  send local state
# ═════════════════════════════════════════════════════════════════════════════

def _process_queue():
    with MP.q_lock:
        msgs = MP.in_queue[:]
        MP.in_queue.clear()

    sync_folder = ""
    try: sync_folder = bpy.context.scene.mp_settings.sync_folder
    except Exception: pass

    for msg in msgs:
        mtype = msg.get("type")
        name  = msg.get("name", "???")

        if mtype == "peer_joined":
            with MP.peer_lock:
                MP.peer_data.setdefault(name, {})["colour"] = \
                    tuple(msg.get("colour", [1,.5,0,1]))
            MP.status = f"● {name} joined"

        elif mtype == "peer_list":
            for n, c in msg.get("peers", {}).items():
                with MP.peer_lock:
                    MP.peer_data.setdefault(n, {})["colour"] = tuple(c)

        elif mtype == "peer_left":
            with MP.peer_lock: MP.peer_data.pop(name, None)
            MP.status = f"● {name} left"

        elif mtype == "cursor":
            with MP.peer_lock:
                MP.peer_data.setdefault(name, {})["cursor"] = msg.get("pos")

        elif mtype == "camera":
            with MP.peer_lock:
                d = MP.peer_data.setdefault(name, {})
                d["cam_loc"] = msg.get("loc"); d["cam_rot"] = msg.get("rot")

        elif mtype in ("objects", "full_state"):
            objs = msg.get("objects", {})
            if objs: _apply_objects(objs, update_baseline=True)

        elif mtype == "tex_push":
            with MP.tex_lock:
                MP.sent_tex_hashes.add(msg.get("hash", ""))
            _install_texture(msg, sync_folder)

        elif mtype == "tex_manifest":
            extra = msg.get("entries", {})
            if extra and sync_folder:
                folder = bpy.path.abspath(sync_folder)
                os.makedirs(folder, exist_ok=True)
                existing = _load_manifest(sync_folder)
                existing.update(extra)
                _save_manifest(sync_folder, existing)
            _apply_manifest(sync_folder, extra_entries=extra)

        elif mtype == "_on_peer_join":
            target = msg.get("_conn")
            if not target: continue
            sync_mesh = False
            try: sync_mesh = bpy.data.scenes[0].mp_settings.sync_mesh
            except Exception: pass
            full = _collect_full_scene(sync_mesh)
            _send(target, {"type": "full_state", "name": "__host__",
                           "objects": full})
            if sync_folder:
                entries = _load_manifest(sync_folder)
                if entries:
                    _send(target, {"type": "tex_manifest", "entries": entries})
            pushes = _collect_all_tex_pushes(skip_hashes=set())
            for p in pushes:
                _send(target, p)
                with MP.tex_lock: MP.sent_tex_hashes.add(p["hash"])
            print(f"[MP] sent {len(pushes)} textures to new peer.")


def _send_local_state():
    scene     = bpy.context.scene
    settings  = scene.mp_settings
    my_name   = settings.user_name.strip() or "Anonymous"
    sync_mesh = settings.sync_mesh
    sf        = settings.sync_folder

    c = scene.cursor.location
    cursor_msg = {"type": "cursor", "name": my_name,
                  "pos": [round(c.x,PREC), round(c.y,PREC), round(c.z,PREC)]}

    camera_msg = None
    if MP.cached_region_3d:
        r3d  = MP.cached_region_3d
        vmat = r3d.view_matrix.inverted()
        loc  = vmat.translation; rot = vmat.to_quaternion()
        camera_msg = {"type": "camera", "name": my_name,
                      "loc": [round(v,PREC) for v in loc],
                      "rot": [round(v,PREC) for v in [rot.w,rot.x,rot.y,rot.z]]}

    changed = _collect_delta(sync_mesh)
    objects_msg = None
    if changed:
        objects_msg = {"type": "objects", "name": my_name, "objects": changed}
        with MP.baseline_lock:
            for n, s in changed.items():
                MP.obj_baseline.pop(n, None) if s.get("deleted") \
                    else MP.obj_baseline.update({n: s})

    tex_msgs = []
    if changed:
        with MP.tex_lock: already = set(MP.sent_tex_hashes)
        for state in changed.values():
            if state.get("deleted"): continue
            h        = state.get("tex_hash", "")
            filepath = state.get("tex_path_local", "")
            img_name = state.get("tex_name", "")
            mat_name = state.get("tex_mat", "")
            if h and filepath and img_name and h not in already:
                push = _build_tex_push(img_name, filepath, mat_name)
                if push:
                    tex_msgs.append(push)
                    if sf: _add_to_manifest(sf, h, {
                        "image_name": img_name, "filename": push["filename"],
                        "ext": push["ext"], "original_path": filepath,
                        "mat_name": mat_name})
                    with MP.tex_lock: MP.sent_tex_hashes.add(h)

    manifest_msg = None
    if tex_msgs and sf:
        entries = _load_manifest(sf)
        if entries:
            manifest_msg = {"type": "tex_manifest", "entries": entries}

    messages = ([m for m in [cursor_msg, camera_msg, objects_msg] if m]
                + tex_msgs + ([manifest_msg] if manifest_msg else []))

    if MP.is_host:
        with MP.conn_lock: conns = list(MP.client_conns.keys())
        for msg in messages:
            for c in conns: _send(c, msg)
    elif MP.my_sock:
        for msg in messages: _send(MP.my_sock, msg)


# ─────────────────────────────────────────────────────────────────────────────
#  SYNC TIMER
# ─────────────────────────────────────────────────────────────────────────────

def _sync_tick():
    if not MP.connected:
        MP.timer_active = False; return None
    _process_queue()
    _send_local_state()
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
#  VIEWPORT DRAW HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def _draw_peers():
    ctx = bpy.context
    if not ctx or not ctx.space_data or ctx.space_data.type != "VIEW_3D":
        return
    MP.cached_region    = ctx.region
    MP.cached_region_3d = ctx.space_data.region_3d
    if not MP.connected: return
    region = ctx.region; rv3d = ctx.region_data
    if not region or not rv3d: return
    my_name = ctx.scene.mp_settings.user_name.strip()
    with MP.peer_lock:
        snap = {n: dict(d) for n, d in MP.peer_data.items()}
    for name, data in snap.items():
        if name == my_name: continue
        colour = data.get("colour", (1,.5,0,1))
        cp = data.get("cursor")
        if cp:
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(
                     region, rv3d, mathutils.Vector(cp))
            if p2:
                _draw_crosshair(p2, colour, CURSOR_R)
                _draw_label(name, p2[0]+CURSOR_R+4, p2[1]-6, colour)
        cl = data.get("cam_loc")
        if cl:
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(
                     region, rv3d, mathutils.Vector(cl))
            if p2:
                _draw_circle(p2, colour, CAM_R)
                _draw_label(f"{name} [cam]", p2[0]+CAM_R+4, p2[1]-6,
                            (*colour[:3], 0.6))
    pc = len(snap)
    blf.position(0, 12, region.height-22, 0); _blf_size(0, 12)
    blf.color(0, 0.8, 0.8, 0.8, 0.9)
    blf.draw(0, f"MP  {pc} peer{'s' if pc!=1 else ''} online")

def _blf_size(fid, sz):
    try: blf.size(fid, sz, 72)
    except TypeError: blf.size(fid, sz)

def _draw_poly(verts, colour, mode):
    sh = gpu.shader.from_builtin("UNIFORM_COLOR")
    bat = batch_for_shader(sh, mode, {"pos": verts})
    sh.bind(); sh.uniform_float("color", colour)
    gpu.state.blend_set("ALPHA"); gpu.state.line_width_set(1.8)
    bat.draw(sh); gpu.state.blend_set("NONE")

def _draw_circle(center, colour, r, segs=20):
    _draw_poly([(center[0]+r*math.cos(2*math.pi*i/segs),
                 center[1]+r*math.sin(2*math.pi*i/segs))
                for i in range(segs)], colour, "LINE_LOOP")

def _draw_crosshair(center, colour, r):
    _draw_circle(center, colour, r)
    cx, cy = center
    sh = gpu.shader.from_builtin("UNIFORM_COLOR")
    bat = batch_for_shader(sh, "LINES",
              {"pos": [(cx-r,cy),(cx+r,cy),(cx,cy-r),(cx,cy+r)]})
    sh.bind(); sh.uniform_float("color", colour)
    gpu.state.line_width_set(1.5); bat.draw(sh)

def _draw_label(text, x, y, colour):
    blf.enable(0, blf.SHADOW); blf.shadow(0, 3, 0, 0, 0, 0.85)
    blf.shadow_offset(0, 1, -1); blf.position(0, x, y, 0)
    _blf_size(0, LABEL_SIZE); blf.color(0, *colour)
    blf.draw(0, text); blf.disable(0, blf.SHADOW)

def _start_draw_handler():
    if MP.draw_handle is None:
        MP.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_peers, (), "WINDOW", "POST_PIXEL")

def _stop_draw_handler():
    if MP.draw_handle:
        try: bpy.types.SpaceView3D.draw_handler_remove(MP.draw_handle, "WINDOW")
        except Exception: pass
        MP.draw_handle = None


# ─────────────────────────────────────────────────────────────────────────────
#  DISCONNECT
# ─────────────────────────────────────────────────────────────────────────────

def _full_disconnect():
    MP.connected = False; MP.is_host = False
    for sock in (MP.server_sock, MP.my_sock):
        if sock:
            try: sock.close()
            except Exception: pass
    MP.server_sock = None; MP.my_sock = None
    with MP.conn_lock:
        for c in list(MP.client_conns):
            try: c.close()
            except Exception: pass
        MP.client_conns.clear()
    with MP.peer_lock:   MP.peer_data.clear()
    with MP.q_lock:      MP.in_queue.clear()
    with MP.baseline_lock: MP.obj_baseline.clear()
    with MP.tex_lock:    MP.sent_tex_hashes.clear()
    with MP.pending_lock: MP.pending_tex_connects.clear()
    _stop_timer(); _stop_draw_handler()
    MP.status = "Disconnected"; MP.colour_idx = 0


# ═════════════════════════════════════════════════════════════════════════════
#  OPERATORS
# ═════════════════════════════════════════════════════════════════════════════

class MP_OT_Host(Operator):
    bl_idname = "mp.host_session"; bl_label = "Host Session"
    bl_options = {"REGISTER"}
    def execute(self, context):
        if not context.scene.mp_settings.user_name.strip():
            self.report({"ERROR"}, "Enter your name first!"); return {"CANCELLED"}
        if MP.connected:
            self.report({"WARNING"}, "Disconnect first."); return {"CANCELLED"}
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("", DEFAULT_PORT)); srv.listen(8)
            MP.server_sock = srv; MP.is_host = True; MP.connected = True
            with MP.baseline_lock:
                MP.obj_baseline = _collect_full_scene(
                    context.scene.mp_settings.sync_mesh)
            code = _encode_code(_get_local_ip(), DEFAULT_PORT)
            context.scene.mp_settings.room_code = code
            MP.status = "Hosting  •  0 peers"
            threading.Thread(target=_host_accept_thread, daemon=True).start()
            _start_timer(); _start_draw_handler()
            sf = context.scene.mp_settings.sync_folder
            if sf: _apply_manifest(sf)
            self.report({"INFO"}, f"Hosting!  Code: {code}")
        except Exception as e:
            self.report({"ERROR"}, str(e)); return {"CANCELLED"}
        return {"FINISHED"}


class MP_OT_Join(Operator):
    bl_idname = "mp.join_session"; bl_label = "Join Session"
    bl_options = {"REGISTER"}
    def execute(self, context):
        settings = context.scene.mp_settings
        name, code = settings.user_name.strip(), settings.room_code.strip()
        if not name or not code:
            self.report({"ERROR"}, "Enter name and room code!"); return {"CANCELLED"}
        if MP.connected:
            self.report({"WARNING"}, "Disconnect first."); return {"CANCELLED"}
        try:
            ip, port = _decode_code(code)
        except Exception:
            self.report({"ERROR"}, "Invalid room code."); return {"CANCELLED"}
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(6.0); sock.connect((ip, port)); sock.settimeout(None)
            MP.my_sock = sock; MP.is_host = False
            MP.connected = True; MP.status = "● Connected"
            _send(sock, {"type": "join", "name": name})
            threading.Thread(target=_client_recv_thread, daemon=True).start()
            _start_timer(); _start_draw_handler()
            sf = settings.sync_folder
            if sf: _apply_manifest(sf)
            self.report({"INFO"}, "Joined!")
        except Exception as e:
            self.report({"ERROR"}, f"Could not connect: {e}"); return {"CANCELLED"}
        return {"FINISHED"}


class MP_OT_Disconnect(Operator):
    bl_idname = "mp.disconnect"; bl_label = "Disconnect"; bl_options = {"REGISTER"}
    def execute(self, context): _full_disconnect(); return {"FINISHED"}


class MP_OT_CopyCode(Operator):
    bl_idname = "mp.copy_code"; bl_label = "Copy"; bl_options = {"REGISTER"}
    def execute(self, context):
        context.window_manager.clipboard = context.scene.mp_settings.room_code
        return {"FINISHED"}


class MP_OT_SnapCam(Operator):
    bl_idname = "mp.snap_cam"; bl_label = "Snap to camera"; bl_options = {"REGISTER"}
    peer_name: StringProperty(default="")
    def execute(self, context):
        with MP.peer_lock: data = dict(MP.peer_data.get(self.peer_name, {}))
        loc, rot = data.get("cam_loc"), data.get("cam_rot")
        if not loc or not rot: return {"CANCELLED"}
        r3d = context.space_data.region_3d
        r3d.view_location = mathutils.Vector(loc)
        r3d.view_rotation = mathutils.Quaternion([rot[0],rot[1],rot[2],rot[3]])
        return {"FINISHED"}


class MP_OT_PushScene(Operator):
    """Force-push full scene + all textures + manifest."""
    bl_idname = "mp.push_scene"; bl_label = "Push Full Scene Now"
    bl_options = {"REGISTER"}
    def execute(self, context):
        if not MP.connected: return {"CANCELLED"}
        my_name = context.scene.mp_settings.user_name.strip() or "Anonymous"
        sf      = context.scene.mp_settings.sync_folder
        with MP.tex_lock: MP.sent_tex_hashes.clear()
        tex_pushes = _collect_all_tex_pushes(skip_hashes=set())
        with MP.tex_lock:
            for p in tex_pushes:
                MP.sent_tex_hashes.add(p["hash"])
                if sf: _add_to_manifest(sf, p["hash"], {
                    "image_name": p["image_name"], "filename": p["filename"],
                    "ext": p["ext"], "original_path": p["original_path"],
                    "mat_name": p["mat_name"]})
        full = _collect_full_scene(context.scene.mp_settings.sync_mesh)
        msgs = [{"type": "full_state", "name": my_name, "objects": full}] + tex_pushes
        if sf:
            entries = _load_manifest(sf)
            if entries: msgs.append({"type": "tex_manifest", "entries": entries})
        if MP.is_host:
            with MP.conn_lock: conns = list(MP.client_conns.keys())
            for msg in msgs:
                for c in conns: _send(c, msg)
        elif MP.my_sock:
            for msg in msgs: _send(MP.my_sock, msg)
        with MP.baseline_lock: MP.obj_baseline = dict(full)
        self.report({"INFO"}, f"Pushed {len(full)} objects + {len(tex_pushes)} textures.")
        return {"FINISHED"}


class MP_OT_RepairMissing(Operator):
    """Re-run manifest repair + reconnect all texture nodes right now."""
    bl_idname = "mp.repair_missing"; bl_label = "Repair Missing Files"
    bl_options = {"REGISTER"}
    def execute(self, context):
        sf = context.scene.mp_settings.sync_folder
        if not sf:
            self.report({"WARNING"}, "No texture folder set."); return {"CANCELLED"}
        _apply_manifest(sf)
        self.report({"INFO"}, "Repair complete.")
        return {"FINISHED"}


# ═════════════════════════════════════════════════════════════════════════════
#  PROPERTIES  &  UI
# ═════════════════════════════════════════════════════════════════════════════

class MPSettings(PropertyGroup):
    user_name:   StringProperty(name="Your Name",   default="", maxlen=24)
    room_code:   StringProperty(name="Room Code",   default="")
    sync_mesh:   BoolProperty(
        name="Sync Mesh Shapes",
        description="Sync vertex positions live (slow on high-poly meshes)",
        default=False)
    sync_folder: StringProperty(
        name="Texture Folder", subtype='DIR_PATH',
        description=(
            "Folder for textures + mp_textures.json manifest.\n"
            "Set BEFORE connecting."),
        default="")


class MP_PT_Main(Panel):
    bl_label = "Multiplayer"; bl_idname = "MP_PT_Main"
    bl_space_type = "VIEW_3D"; bl_region_type = "UI"; bl_category = "Multiplayer"

    def draw(self, context):
        layout   = self.layout
        settings = context.scene.mp_settings

        sb = layout.box()
        sb.label(text=MP.status,
                 icon="RADIOBUT_ON" if MP.connected else "RADIOBUT_OFF")
        layout.separator(factor=0.4)
        layout.label(text="Identity", icon="USER")
        layout.prop(settings, "user_name", text="Name")
        layout.separator(factor=0.4)
        layout.label(text="Options", icon="MODIFIER")
        layout.prop(settings, "sync_mesh")
        layout.label(text="Texture Folder  (mp_textures.json saved here):",
                     icon="FILE_FOLDER")
        layout.prop(settings, "sync_folder", text="")
        layout.separator()

        if not MP.connected:
            hb = layout.box()
            hb.label(text="Start a Session", icon="SOLO_ON")
            r = hb.row(); r.scale_y = 1.35
            r.operator("mp.host_session", icon="SOLO_ON")
            layout.separator(factor=0.3)
            jb = layout.box()
            jb.label(text="Join a Session", icon="LINKED")
            jb.prop(settings, "room_code", text="Code")
            r2 = jb.row(); r2.scale_y = 1.35
            r2.operator("mp.join_session", icon="LINKED")
        else:
            if MP.is_host:
                cb = layout.box()
                cb.label(text="Share this code:", icon="KEY_HLT")
                row2 = cb.row(align=True)
                row2.prop(settings, "room_code", text="")
                row2.operator("mp.copy_code", icon="COPYDOWN", text="Copy")
                layout.separator(factor=0.3)
            pb = layout.box()
            pb.label(text="Online", icon="COMMUNITY")
            pb.label(text=f"  You  ({settings.user_name.strip() or 'Anonymous'})",
                     icon="FUND")
            with MP.peer_lock: snap = dict(MP.peer_data)
            for pname in snap:
                pr = pb.row(align=True)
                pr.label(text=f"  ● {pname}", icon="FUND")
                op = pr.operator("mp.snap_cam", text="", icon="VIEW_CAMERA")
                op.peer_name = pname
            if not snap:
                pb.label(text="  Waiting for peers…", icon="TIME")
            layout.separator(factor=0.3)
            layout.operator("mp.push_scene",     icon="EXPORT")
            layout.operator("mp.repair_missing", icon="FILE_REFRESH")
            dr = layout.row(); dr.alert = True; dr.scale_y = 1.2
            dr.operator("mp.disconnect", icon="X")


_CLASSES = (
    MPSettings, MP_OT_Host, MP_OT_Join, MP_OT_Disconnect,
    MP_OT_CopyCode, MP_OT_SnapCam, MP_OT_PushScene,
    MP_OT_RepairMissing, MP_PT_Main,
)

def register():
    for cls in _CLASSES: bpy.utils.register_class(cls)
    bpy.types.Scene.mp_settings = bpy.props.PointerProperty(type=MPSettings)
    print("[MP] Blender Multiplayer Sync v4.2 registered.")

def unregister():
    _full_disconnect()
    for cls in reversed(_CLASSES): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mp_settings
    print("[MP] Blender Multiplayer Sync unregistered.")

if __name__ == "__main__":
    register()
