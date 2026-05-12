# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          BLENDER MULTIPLAYER SYNC  v4.0  –  blender_multiplayer.py     ║
# ║  Full bidirectional sync: objects · cursor · camera · meshes           ║
# ║  Proactive texture push · auto find-missing-files · keyframe sync      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# INSTALL:  Edit > Preferences > Add-ons > Install…  →  pick this file  →  ✓
# USE:      N-panel → "Multiplayer" tab
#           Set Asset Folder BEFORE connecting – all textures land there.
#
# TEXTURE SYNC:
#   Textures are pushed PROACTIVELY – no request/response roundtrip.
#   • On join: host immediately pushes every texture in the scene.
#   • On any delta tick: new/changed textures are pushed automatically.
#   • After any texture lands, missing-file repair runs on the whole scene.
#   • Files larger than TEX_SIZE_LIMIT_MB are skipped (logged to console).
#
# KEYFRAME SYNC:
#   Full fcurve rebuild on every action change (clean, no stale points).
#
# NETWORK:  LAN out of the box (TCP 19283).
#           Internet: host port-forwards 19283, or use ZeroTier / Tailscale.

bl_info = {
    "name":        "Blender Multiplayer Sync",
    "author":      "Claude & LAGkit",
    "version":     (4, 0, 0),
    "blender":     (3, 0, 0),
    "location":    "View3D > N-Panel > Multiplayer",
    "description": "Real-time collaboration – objects, meshes, materials, textures, keyframes",
    "category":    "3D View",
}

import os
import base64
import hashlib
import bpy
import socket
import threading
import json
import struct
import math
import mathutils
import bpy_extras.view3d_utils
import blf
import gpu
import bmesh
from gpu_extras.batch import batch_for_shader
from bpy.props import StringProperty, BoolProperty
from bpy.types import Panel, Operator, PropertyGroup

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PORT        = 19283
SYNC_INTERVAL       = 0.05          # 20 Hz
LABEL_SIZE          = 14
CURSOR_R            = 10
CAM_R               = 7
PREC                = 5
TEX_SIZE_LIMIT_MB   = 32            # skip textures larger than this

COLOURS = [
    (1.00, 0.42, 0.12, 1.0),
    (0.18, 0.85, 0.42, 1.0),
    (0.20, 0.55, 1.00, 1.0),
    (0.95, 0.15, 0.60, 1.0),
    (1.00, 0.90, 0.10, 1.0),
    (0.55, 0.20, 1.00, 1.0),
    (0.10, 0.88, 0.90, 1.0),
]


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

class _MP:
    server_sock      = None
    my_sock          = None
    client_conns     = {}
    conn_lock        = threading.Lock()

    peer_data        = {}
    peer_lock        = threading.Lock()

    in_queue         = []
    q_lock           = threading.Lock()

    obj_baseline     = {}
    baseline_lock    = threading.Lock()

    # Hashes of textures we have already SENT this session.
    # Cleared on disconnect so a reconnect re-pushes everything.
    sent_tex_hashes  = set()
    tex_lock         = threading.Lock()

    is_host          = False
    connected        = False
    status           = "Disconnected"
    colour_idx       = 0

    draw_handle      = None
    timer_active     = False

    cached_region    = None
    cached_region_3d = None

MP = _MP()


# ─────────────────────────────────────────────────────────────────────────────
#  TEXTURE  helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tex_file_hash(filepath: str) -> str:
    """Stable hash based on file size + mtime – fast, no full read needed."""
    try:
        st = os.stat(filepath)
        return f"{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        return ""


def _tex_save_name(img_name: str, tex_hash: str, ext: str) -> str:
    """Canonical filename inside the sync folder."""
    safe = img_name.replace("/", "_").replace("\\", "_")
    return f"{safe}__{tex_hash}{ext}"


def _build_tex_push(img_name: str, filepath: str) -> dict | None:
    """
    Read a texture and return a ready-to-send tex_push message, or None
    if the file is missing / too large.
    """
    try:
        size = os.path.getsize(filepath)
        limit = TEX_SIZE_LIMIT_MB * 1024 * 1024
        if size > limit:
            print(f"[MP] Skipping texture '{img_name}' – {size//1024//1024} MB > limit {TEX_SIZE_LIMIT_MB} MB")
            return None
        tex_hash = _tex_file_hash(filepath)
        ext      = os.path.splitext(filepath)[1] or ".png"
        with open(filepath, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type":     "tex_push",
            "name":     img_name,
            "hash":     tex_hash,
            "ext":      ext,
            "data_b64": data_b64,
        }
    except Exception as e:
        print(f"[MP] _build_tex_push({img_name}): {e}")
        return None


def _collect_all_tex_pushes(skip_hashes: set) -> list:
    """
    Build tex_push messages for every image texture in the scene
    whose hash is not already in skip_hashes.
    Main-thread only.
    """
    msgs = []
    seen = set()
    for img in bpy.data.images:
        if not img.filepath:
            continue
        filepath = bpy.path.abspath(img.filepath)
        if not os.path.exists(filepath):
            continue
        tex_hash = _tex_file_hash(filepath)
        if not tex_hash or tex_hash in skip_hashes or tex_hash in seen:
            continue
        seen.add(tex_hash)
        msg = _build_tex_push(img.name, filepath)
        if msg:
            msgs.append(msg)
    return msgs


def _install_texture(msg: dict, sync_folder: str):
    """
    Save received tex_push data to sync_folder, load it into Blender,
    and repair ALL missing-file references that match the texture name.
    Main-thread only.
    """
    if not sync_folder:
        return
    folder = bpy.path.abspath(sync_folder)
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f"[MP] Cannot create sync folder: {e}")
            return

    img_name = msg["name"]
    tex_hash = msg["hash"]
    ext      = msg.get("ext", ".png")
    save_name = _tex_save_name(img_name, tex_hash, ext)
    save_path = os.path.join(folder, save_name)

    # Write file if not already there
    if not os.path.exists(save_path):
        try:
            raw = base64.b64decode(msg["data_b64"])
            with open(save_path, "wb") as f:
                f.write(raw)
            print(f"[MP] Saved texture → {save_path}")
        except Exception as e:
            print(f"[MP] _install_texture write error: {e}")
            return

    # ── Load / update image datablock ────────────────────────────────────────
    img = bpy.data.images.get(img_name)
    if img is None:
        try:
            img = bpy.data.images.load(save_path)
            img.name = img_name
        except Exception as e:
            print(f"[MP] load image error: {e}")
    else:
        if bpy.path.abspath(img.filepath) != save_path:
            img.filepath = save_path
        try:
            img.reload()
        except Exception:
            pass

    # ── Auto find-missing-files across the whole scene ────────────────────────
    # Strategy: any image whose filename stem matches the incoming texture name
    # and whose current filepath is missing gets redirected to save_path.
    incoming_stem = os.path.splitext(img_name)[0].lower()
    for other in bpy.data.images:
        if other == img or not other.filepath:
            continue
        abs_fp = bpy.path.abspath(other.filepath)
        if os.path.exists(abs_fp):
            continue                          # file found, leave it
        other_stem = os.path.splitext(os.path.basename(abs_fp))[0].lower()
        if other_stem == incoming_stem:
            other.filepath = save_path
            try:
                other.reload()
                print(f"[MP] Repaired missing → '{other.name}' now at {save_path}")
            except Exception:
                pass

    # ── Also scan sync_folder for anything else that might patch other images ─
    _scan_folder_for_missing(folder)


def _scan_folder_for_missing(folder: str):
    """
    Walk every image in bpy.data.images; if its file is missing,
    check sync_folder for a file whose stem matches and redirect it.
    """
    if not folder or not os.path.isdir(folder):
        return
    try:
        available = {os.path.splitext(f)[0].lower(): os.path.join(folder, f)
                     for f in os.listdir(folder)}
    except Exception:
        return

    for img in bpy.data.images:
        if not img.filepath:
            continue
        abs_fp = bpy.path.abspath(img.filepath)
        if os.path.exists(abs_fp):
            continue
        stem = os.path.splitext(os.path.basename(abs_fp))[0].lower()
        # Try exact stem
        candidate = available.get(stem)
        if not candidate:
            # Try prefix match (e.g. "wood" matches "wood__123_456")
            for k, v in available.items():
                if k.startswith(stem) or stem.startswith(k.split("__")[0]):
                    candidate = v
                    break
        if candidate and os.path.exists(candidate):
            img.filepath = candidate
            try:
                img.reload()
                print(f"[MP] find-missing patched '{img.name}' → {candidate}")
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECT STATE  helpers
# ─────────────────────────────────────────────────────────────────────────────

def _obj_to_state(obj, sync_mesh: bool = False) -> dict:
    """Serialize transform, material, animation, and optionally mesh geometry."""
    loc   = [round(v, PREC) for v in obj.location]
    mode  = obj.rotation_mode
    if mode == 'QUATERNION':
        rot = [round(v, PREC) for v in obj.rotation_quaternion]
    elif mode == 'AXIS_ANGLE':
        rot = [round(v, PREC) for v in obj.rotation_axis_angle]
    else:
        rot = [round(v, PREC) for v in obj.rotation_euler]
    scale = [round(v, PREC) for v in obj.scale]

    state: dict = {
        "type":     obj.type,
        "loc":      loc,
        "rot_mode": mode,
        "rot":      rot,
        "scale":    scale,
    }

    # ── Material / texture metadata (binary not included here) ────────────────
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
                if tex_node and tex_node.image:
                    img      = tex_node.image
                    filepath = bpy.path.abspath(img.filepath) if img.filepath else ""
                    if filepath and os.path.exists(filepath):
                        state["tex_name"] = img.name
                        state["tex_hash"] = _tex_file_hash(filepath)
                        state["tex_ext"]  = os.path.splitext(filepath)[1] or ".png"
                        # Local path used only by the SENDER to build the push
                        state["tex_path_local"] = filepath
            except Exception:
                pass

    # ── Animation / keyframes ─────────────────────────────────────────────────
    if obj.animation_data and obj.animation_data.action:
        try:
            act    = obj.animation_data.action
            curves = []
            for fc in act.fcurves:
                kps = [(round(kp.co[0], PREC), round(kp.co[1], PREC))
                       for kp in fc.keyframe_points]
                curves.append({
                    "path": fc.data_path,
                    "idx":  fc.array_index,
                    "kps":  kps,
                })
            state["anim"] = {"name": act.name, "curves": curves}
        except Exception as e:
            print(f"[MP] anim serialize ({obj.name}): {e}")

    # ── Mesh geometry ─────────────────────────────────────────────────────────
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
    """Deterministic fingerprint for an animation dict (used in _state_key)."""
    if not anim:
        return ""
    curves = anim.get("curves", [])
    # Include path names, keyframe count, and last keyframe value per curve
    parts = []
    for fc in curves:
        kps = fc.get("kps", [])
        last = kps[-1] if kps else (0, 0)
        parts.append(f"{fc['path']}[{fc['idx']}]:{len(kps)}:{last[0]:.3f}:{last[1]:.5f}")
    return "|".join(parts)


def _state_key(s: dict) -> tuple:
    """Hashable snapshot for change detection."""
    v_hash = hash(tuple(s["verts"]))  if "verts"     in s else 0
    m_col  = tuple(s["mat_color"])    if "mat_color" in s else ()
    t_hash = s.get("tex_hash", "")
    a_sig  = _anim_sig(s.get("anim", {}))
    return (
        tuple(s.get("loc",   [])),
        s.get("rot_mode", ""),
        tuple(s.get("rot",   [])),
        tuple(s.get("scale", [])),
        v_hash, m_col, t_hash, a_sig,
        s.get("deleted", False),
    )


def _collect_full_scene(sync_mesh: bool = False) -> dict:
    return {obj.name: _obj_to_state(obj, sync_mesh) for obj in bpy.data.objects}


def _collect_delta(sync_mesh: bool = False) -> dict:
    changed: dict = {}
    current_names = {obj.name for obj in bpy.data.objects}

    with MP.baseline_lock:
        baseline = dict(MP.obj_baseline)

    # Deletions
    for name, state in baseline.items():
        if name not in current_names and not state.get("deleted"):
            changed[name] = {"deleted": True}

    # Changed / new objects
    for obj in bpy.data.objects:
        state = _obj_to_state(obj, sync_mesh)
        prev  = baseline.get(obj.name)
        if prev is None or _state_key(prev) != _state_key(state):
            changed[obj.name] = state

    return changed


# ─────────────────────────────────────────────────────────────────────────────
#  APPLY STATE  (main-thread only)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_keyframes(obj, anim: dict):
    """
    Cleanly rebuild all fcurves from received anim data.
    Removes stale curves, inserts fresh keypoints.
    """
    act_name = anim.get("name", "")
    if not act_name:
        return

    if not obj.animation_data:
        obj.animation_data_create()

    act = bpy.data.actions.get(act_name)
    if act is None:
        act = bpy.data.actions.new(act_name)

    # Assign action
    if obj.animation_data.action is not act:
        obj.animation_data.action = act

    incoming_keys = {(fc["path"], fc["idx"]) for fc in anim.get("curves", [])}

    # Remove fcurves not in incoming data
    to_remove = [fc for fc in act.fcurves
                 if (fc.data_path, fc.array_index) not in incoming_keys]
    for fc in to_remove:
        act.fcurves.remove(fc)

    # Rebuild each curve
    for fc_data in anim.get("curves", []):
        path = fc_data["path"]
        idx  = fc_data["idx"]
        kps  = fc_data["kps"]

        existing = act.fcurves.find(path, index=idx)
        if existing:
            act.fcurves.remove(existing)

        if not kps:
            continue

        fc = act.fcurves.new(path, index=idx)
        fc.keyframe_points.add(len(kps))
        for i, (frame, value) in enumerate(kps):
            kp = fc.keyframe_points[i]
            kp.co_ui           = (frame, value)
            kp.interpolation   = 'BEZIER'
            kp.handle_left_type  = 'AUTO'
            kp.handle_right_type = 'AUTO'
        fc.update()


def _state_to_obj(obj, state: dict):
    """Apply transform, material, keyframes, and mesh to obj. Main-thread only."""
    try:
        # ── Transform ─────────────────────────────────────────────────────────
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

        # ── Material ──────────────────────────────────────────────────────────
        if "mat_name" in state:
            mat_name = state["mat_name"]
            mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
            if not mat.use_nodes:
                mat.use_nodes = True
            if len(obj.material_slots) == 0:
                obj.data.materials.append(mat)
            elif obj.material_slots[0].material != mat:
                obj.material_slots[0].material = mat

            if "mat_color" in state:
                try:
                    nodes = mat.node_tree.nodes
                    bsdf  = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                    if bsdf:
                        bsdf.inputs["Base Color"].default_value = state["mat_color"]
                except Exception:
                    pass

        # ── Keyframes ─────────────────────────────────────────────────────────
        if "anim" in state and state["anim"]:
            _apply_keyframes(obj, state["anim"])

        # ── Mesh geometry ─────────────────────────────────────────────────────
        if "verts" in state and obj.type == 'MESH':
            verts = state["verts"]
            faces = state.get("faces", [])
            n_verts = len(verts) // 3
            n_faces = len(faces)

            if (n_verts != len(obj.data.vertices) or
                    n_faces != len(obj.data.polygons)):
                if obj.mode != 'EDIT':
                    mesh = obj.data
                    mesh.clear_geometry()
                    v_list = [(verts[i], verts[i+1], verts[i+2])
                              for i in range(0, len(verts), 3)]
                    mesh.from_pydata(v_list, [], faces)
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
    """Apply a dict of {name: state} to the local scene. Main-thread only."""
    for name, state in objects_data.items():

        # Deletion
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

        # Creation if missing
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
                print(f"[MP] create object '{name}': {e}")
                continue

        # Apply state
        _state_to_obj(obj, state)
        if update_baseline:
            with MP.baseline_lock:
                MP.obj_baseline[name] = state


# ─────────────────────────────────────────────────────────────────────────────
#  NETWORK  utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _encode_code(ip: str, port: int) -> str:
    return base64.b32encode(f"{ip}:{port}".encode()).decode().rstrip("=")


def _decode_code(code: str):
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


def _recvall(sock, n: int):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _recv(sock):
    try:
        hdr = _recvall(sock, 4)
        if hdr is None:
            return None
        length = struct.unpack(">I", hdr)[0]
        if length > 200 * 1024 * 1024:     # 200 MB absolute cap
            return None
        payload = _recvall(sock, length)
        if payload is None:
            return None
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def _broadcast(exclude, msg: dict):
    with MP.conn_lock:
        targets = list(MP.client_conns.keys())
    for c in targets:
        if c is not exclude:
            _send(c, msg)


# ─────────────────────────────────────────────────────────────────────────────
#  HOST  threads
# ─────────────────────────────────────────────────────────────────────────────

def _host_client_thread(conn, addr):
    peer_name = "???"
    try:
        while MP.connected:
            msg = _recv(conn)
            if msg is None:
                break
            mtype = msg.get("type")

            if mtype == "join":
                peer_name = msg.get("name", "???")[:24]
                colour    = COLOURS[MP.colour_idx % len(COLOURS)]
                MP.colour_idx += 1

                with MP.conn_lock:
                    MP.client_conns[conn] = {"name": peer_name}
                with MP.peer_lock:
                    MP.peer_data[peer_name] = {"colour": colour}

                _broadcast(conn, {"type": "peer_joined",
                                  "name": peer_name,
                                  "colour": list(colour)})

                with MP.peer_lock:
                    existing = {n: list(d.get("colour", [1, .5, 0, 1]))
                                for n, d in MP.peer_data.items()
                                if n != peer_name}
                _send(conn, {"type": "peer_list", "peers": existing})

                # ── Full scene snapshot ────────────────────────────────────────
                sync_mesh = False
                try:
                    sync_mesh = bpy.data.scenes[0].mp_settings.sync_mesh
                except Exception:
                    pass
                full = _collect_full_scene(sync_mesh)
                _send(conn, {"type": "full_state",
                             "name": "__host__", "objects": full})

                # ── Proactively push ALL textures ─────────────────────────────
                # Run on the main-thread queue so bpy access is safe
                with MP.q_lock:
                    MP.in_queue.append({"type": "_push_textures_to",
                                        "_conn": conn})

            elif mtype in ("cursor", "camera"):
                msg["name"] = peer_name
                with MP.q_lock:
                    MP.in_queue.append(msg)
                _broadcast(conn, msg)

            elif mtype in ("objects", "full_state"):
                msg["name"] = peer_name
                _broadcast(conn, msg)
                with MP.q_lock:
                    MP.in_queue.append(msg)

            elif mtype == "tex_push":
                # Relay to all other clients; also apply locally via queue
                _broadcast(conn, msg)
                with MP.q_lock:
                    MP.in_queue.append(msg)

    except Exception as e:
        print(f"[MP] client-thread: {e}")
    finally:
        with MP.conn_lock:
            MP.client_conns.pop(conn, None)
        with MP.peer_lock:
            MP.peer_data.pop(peer_name, None)
        try:
            conn.close()
        except Exception:
            pass
        _broadcast(None, {"type": "peer_left", "name": peer_name})
        print(f"[MP] {peer_name} left")


def _host_accept_thread():
    while MP.connected and MP.server_sock:
        try:
            MP.server_sock.settimeout(1.0)
            conn, addr = MP.server_sock.accept()
            threading.Thread(target=_host_client_thread,
                             args=(conn, addr), daemon=True).start()
        except socket.timeout:
            continue
        except Exception:
            break


# ─────────────────────────────────────────────────────────────────────────────
#  CLIENT  receive thread
# ─────────────────────────────────────────────────────────────────────────────

def _client_recv_thread():
    while MP.connected and MP.my_sock:
        msg = _recv(MP.my_sock)
        if msg is None:
            MP.connected = False
            MP.status    = "⚠ Lost connection"
            break
        with MP.q_lock:
            MP.in_queue.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN-THREAD:  process queue  +  send local state
# ─────────────────────────────────────────────────────────────────────────────

def _process_queue():
    with MP.q_lock:
        msgs = MP.in_queue[:]
        MP.in_queue.clear()

    sync_folder = ""
    try:
        sync_folder = bpy.context.scene.mp_settings.sync_folder
    except Exception:
        pass

    for msg in msgs:
        mtype = msg.get("type")
        name  = msg.get("name", "???")

        if mtype == "peer_joined":
            colour = tuple(msg.get("colour", [1, .5, 0, 1]))
            with MP.peer_lock:
                MP.peer_data.setdefault(name, {})["colour"] = colour
            MP.status = f"● {name} joined"

        elif mtype == "peer_list":
            for n, c in msg.get("peers", {}).items():
                with MP.peer_lock:
                    MP.peer_data.setdefault(n, {})["colour"] = tuple(c)

        elif mtype == "peer_left":
            with MP.peer_lock:
                MP.peer_data.pop(name, None)
            MP.status = f"● {name} left"

        elif mtype == "cursor":
            with MP.peer_lock:
                MP.peer_data.setdefault(name, {})["cursor"] = msg.get("pos", [0,0,0])

        elif mtype == "camera":
            with MP.peer_lock:
                d = MP.peer_data.setdefault(name, {})
                d["cam_loc"] = msg.get("loc", [0,0,0])
                d["cam_rot"] = msg.get("rot", [1,0,0,0])

        elif mtype in ("objects", "full_state"):
            objs = msg.get("objects", {})
            if objs:
                _apply_objects(objs, update_baseline=True)

        elif mtype == "tex_push":
            # Save + install; mark hash as received so we don't re-request
            with MP.tex_lock:
                MP.sent_tex_hashes.add(msg.get("hash", ""))
            _install_texture(msg, sync_folder)

        elif mtype == "_push_textures_to":
            # Internal: host pushing all textures to a newly joined conn
            target_conn = msg.get("_conn")
            if target_conn:
                with MP.tex_lock:
                    already = set(MP.sent_tex_hashes)
                pushes = _collect_all_tex_pushes(skip_hashes=set())  # send everything to newcomer
                for push_msg in pushes:
                    _send(target_conn, push_msg)
                    with MP.tex_lock:
                        MP.sent_tex_hashes.add(push_msg["hash"])
                print(f"[MP] Pushed {len(pushes)} textures to newcomer.")


def _send_local_state():
    scene    = bpy.context.scene
    settings = scene.mp_settings
    my_name  = settings.user_name.strip() or "Anonymous"
    sync_mesh = settings.sync_mesh

    # 3D cursor
    c = scene.cursor.location
    cursor_msg = {
        "type": "cursor", "name": my_name,
        "pos":  [round(c.x, PREC), round(c.y, PREC), round(c.z, PREC)],
    }

    # Viewport camera
    camera_msg = None
    if MP.cached_region_3d:
        r3d  = MP.cached_region_3d
        vmat = r3d.view_matrix.inverted()
        loc  = vmat.translation
        rot  = vmat.to_quaternion()
        camera_msg = {
            "type": "camera", "name": my_name,
            "loc":  [round(v, PREC) for v in loc],
            "rot":  [round(v, PREC) for v in [rot.w, rot.x, rot.y, rot.z]],
        }

    # Object deltas
    changed = _collect_delta(sync_mesh)
    objects_msg = None
    if changed:
        objects_msg = {"type": "objects", "name": my_name, "objects": changed}
        with MP.baseline_lock:
            for n, s in changed.items():
                if s.get("deleted"):
                    MP.obj_baseline.pop(n, None)
                else:
                    MP.obj_baseline[n] = s

    # ── Proactive texture push for NEW textures in the delta ─────────────────
    tex_msgs = []
    if changed:
        with MP.tex_lock:
            already_sent = set(MP.sent_tex_hashes)

        for state in changed.values():
            if state.get("deleted"):
                continue
            tex_hash = state.get("tex_hash", "")
            filepath  = state.get("tex_path_local", "")
            img_name  = state.get("tex_name", "")
            if tex_hash and filepath and img_name and tex_hash not in already_sent:
                push = _build_tex_push(img_name, filepath)
                if push:
                    tex_msgs.append(push)
                    with MP.tex_lock:
                        MP.sent_tex_hashes.add(tex_hash)

    messages = [m for m in [cursor_msg, camera_msg, objects_msg] if m] + tex_msgs

    if MP.is_host:
        with MP.conn_lock:
            conns = list(MP.client_conns.keys())
        for msg in messages:
            for c in conns:
                _send(c, msg)
    elif MP.my_sock:
        for msg in messages:
            _send(MP.my_sock, msg)


# ─────────────────────────────────────────────────────────────────────────────
#  SYNC TIMER  (main thread – safe for all bpy writes)
# ─────────────────────────────────────────────────────────────────────────────

def _sync_tick():
    if not MP.connected:
        MP.timer_active = False
        return None

    _process_queue()
    _send_local_state()

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

    return SYNC_INTERVAL


def _start_timer():
    if not MP.timer_active:
        MP.timer_active = True
        bpy.app.timers.register(_sync_tick, first_interval=SYNC_INTERVAL)


def _stop_timer():
    MP.timer_active = False
    try:
        bpy.app.timers.unregister(_sync_tick)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  VIEWPORT DRAW HANDLER  (POST_PIXEL)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_peers():
    ctx = bpy.context
    if not ctx or not ctx.space_data or ctx.space_data.type != "VIEW_3D":
        return

    MP.cached_region    = ctx.region
    MP.cached_region_3d = ctx.space_data.region_3d

    if not MP.connected:
        return

    region = ctx.region
    rv3d   = ctx.region_data
    if not region or not rv3d:
        return

    settings = ctx.scene.mp_settings
    my_name  = settings.user_name.strip()

    with MP.peer_lock:
        snap = {n: dict(d) for n, d in MP.peer_data.items()}

    for name, data in snap.items():
        if name == my_name:
            continue
        colour = data.get("colour", (1, .5, 0, 1))

        cp = data.get("cursor")
        if cp:
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(
                     region, rv3d, mathutils.Vector(cp))
            if p2:
                _draw_crosshair(p2, colour, CURSOR_R)
                _draw_label(name, p2[0] + CURSOR_R + 4, p2[1] - 6, colour)

        cl = data.get("cam_loc")
        if cl:
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(
                     region, rv3d, mathutils.Vector(cl))
            if p2:
                _draw_circle(p2, colour, CAM_R)
                _draw_label(f"{name} [cam]",
                            p2[0] + CAM_R + 4, p2[1] - 6,
                            (*colour[:3], 0.6))

    peer_count = len(snap)
    blf.position(0, 12, region.height - 22, 0)
    _blf_size(0, 12)
    blf.color(0, 0.8, 0.8, 0.8, 0.9)
    blf.draw(0, f"MP  {peer_count} peer{'s' if peer_count != 1 else ''} online")


def _blf_size(fid, sz):
    try:
        blf.size(fid, sz, 72)
    except TypeError:
        blf.size(fid, sz)


def _draw_poly(verts, colour, mode):
    sh  = gpu.shader.from_builtin("UNIFORM_COLOR")
    bat = batch_for_shader(sh, mode, {"pos": verts})
    sh.bind()
    sh.uniform_float("color", colour)
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(1.8)
    bat.draw(sh)
    gpu.state.blend_set("NONE")


def _draw_circle(center, colour, r, segs=20):
    verts = [(center[0] + r * math.cos(2*math.pi*i/segs),
              center[1] + r * math.sin(2*math.pi*i/segs))
             for i in range(segs)]
    _draw_poly(verts, colour, "LINE_LOOP")


def _draw_crosshair(center, colour, r):
    _draw_circle(center, colour, r)
    cx, cy = center
    sh  = gpu.shader.from_builtin("UNIFORM_COLOR")
    bat = batch_for_shader(sh, "LINES",
              {"pos": [(cx-r,cy),(cx+r,cy),(cx,cy-r),(cx,cy+r)]})
    sh.bind()
    sh.uniform_float("color", colour)
    gpu.state.line_width_set(1.5)
    bat.draw(sh)


def _draw_label(text, x, y, colour):
    blf.enable(0, blf.SHADOW)
    blf.shadow(0, 3, 0, 0, 0, 0.85)
    blf.shadow_offset(0, 1, -1)
    blf.position(0, x, y, 0)
    _blf_size(0, LABEL_SIZE)
    blf.color(0, *colour)
    blf.draw(0, text)
    blf.disable(0, blf.SHADOW)


def _start_draw_handler():
    if MP.draw_handle is None:
        MP.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_peers, (), "WINDOW", "POST_PIXEL")


def _stop_draw_handler():
    if MP.draw_handle:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(MP.draw_handle, "WINDOW")
        except Exception:
            pass
        MP.draw_handle = None


# ─────────────────────────────────────────────────────────────────────────────
#  DISCONNECT
# ─────────────────────────────────────────────────────────────────────────────

def _full_disconnect():
    MP.connected = False
    MP.is_host   = False
    for sock in (MP.server_sock, MP.my_sock):
        if sock:
            try: sock.close()
            except Exception: pass
    MP.server_sock = None
    MP.my_sock     = None
    with MP.conn_lock:
        for c in list(MP.client_conns):
            try: c.close()
            except Exception: pass
        MP.client_conns.clear()
    with MP.peer_lock:
        MP.peer_data.clear()
    with MP.q_lock:
        MP.in_queue.clear()
    with MP.baseline_lock:
        MP.obj_baseline.clear()
    with MP.tex_lock:
        MP.sent_tex_hashes.clear()   # reset so reconnect re-pushes everything
    _stop_timer()
    _stop_draw_handler()
    MP.status     = "Disconnected"
    MP.colour_idx = 0


# ─────────────────────────────────────────────────────────────────────────────
#  OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

class MP_OT_Host(Operator):
    bl_idname = "mp.host_session"
    bl_label  = "Host Session"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not context.scene.mp_settings.user_name.strip():
            self.report({"ERROR"}, "Enter your name first!")
            return {"CANCELLED"}
        if MP.connected:
            self.report({"WARNING"}, "Disconnect first.")
            return {"CANCELLED"}
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("", DEFAULT_PORT))
            srv.listen(8)
            MP.server_sock = srv
            MP.is_host     = True
            MP.connected   = True

            with MP.baseline_lock:
                MP.obj_baseline = _collect_full_scene(
                    context.scene.mp_settings.sync_mesh)

            code = _encode_code(_get_local_ip(), DEFAULT_PORT)
            context.scene.mp_settings.room_code = code
            MP.status = "Hosting  •  0 peers"

            threading.Thread(target=_host_accept_thread, daemon=True).start()
            _start_timer()
            _start_draw_handler()

            # Pre-scan sync folder for existing textures (find-missing on start)
            sf = context.scene.mp_settings.sync_folder
            if sf:
                _scan_folder_for_missing(bpy.path.abspath(sf))

            self.report({"INFO"}, f"Hosting!  Code: {code}")
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class MP_OT_Join(Operator):
    bl_idname = "mp.join_session"
    bl_label  = "Join Session"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = context.scene.mp_settings
        name = settings.user_name.strip()
        code = settings.room_code.strip()
        if not name or not code:
            self.report({"ERROR"}, "Enter name and room code!")
            return {"CANCELLED"}
        if MP.connected:
            self.report({"WARNING"}, "Disconnect first.")
            return {"CANCELLED"}
        try:
            ip, port = _decode_code(code)
        except Exception:
            self.report({"ERROR"}, "Invalid room code.")
            return {"CANCELLED"}
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(6.0)
            sock.connect((ip, port))
            sock.settimeout(None)
            MP.my_sock   = sock
            MP.is_host   = False
            MP.connected = True
            MP.status    = "● Connected"
            _send(sock, {"type": "join", "name": name})
            threading.Thread(target=_client_recv_thread, daemon=True).start()
            _start_timer()
            _start_draw_handler()

            # Pre-scan sync folder for textures we already have
            sf = settings.sync_folder
            if sf:
                _scan_folder_for_missing(bpy.path.abspath(sf))

            self.report({"INFO"}, "Joined!")
        except Exception as e:
            self.report({"ERROR"}, f"Could not connect: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class MP_OT_Disconnect(Operator):
    bl_idname = "mp.disconnect"
    bl_label  = "Disconnect"
    bl_options = {"REGISTER"}

    def execute(self, context):
        _full_disconnect()
        return {"FINISHED"}


class MP_OT_CopyCode(Operator):
    bl_idname = "mp.copy_code"
    bl_label  = "Copy"
    bl_options = {"REGISTER"}

    def execute(self, context):
        context.window_manager.clipboard = context.scene.mp_settings.room_code
        return {"FINISHED"}


class MP_OT_SnapCam(Operator):
    """Teleport your viewport to a peer's camera position."""
    bl_idname = "mp.snap_cam"
    bl_label  = "Snap to camera"
    bl_options = {"REGISTER"}
    peer_name: StringProperty(default="")

    def execute(self, context):
        with MP.peer_lock:
            data = dict(MP.peer_data.get(self.peer_name, {}))
        loc = data.get("cam_loc")
        rot = data.get("cam_rot")
        if not loc or not rot:
            return {"CANCELLED"}
        r3d = context.space_data.region_3d
        r3d.view_location = mathutils.Vector(loc)
        r3d.view_rotation = mathutils.Quaternion([rot[0], rot[1], rot[2], rot[3]])
        return {"FINISHED"}


class MP_OT_PushScene(Operator):
    """Force-push your full scene + all textures to all peers right now."""
    bl_idname = "mp.push_scene"
    bl_label  = "Push Full Scene Now"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not MP.connected:
            return {"CANCELLED"}
        my_name = context.scene.mp_settings.user_name.strip() or "Anonymous"

        # Scene
        full = _collect_full_scene(context.scene.mp_settings.sync_mesh)
        scene_msg = {"type": "full_state", "name": my_name, "objects": full}

        # All textures (force re-send by clearing sent set)
        with MP.tex_lock:
            MP.sent_tex_hashes.clear()
        tex_pushes = _collect_all_tex_pushes(skip_hashes=set())
        with MP.tex_lock:
            for p in tex_pushes:
                MP.sent_tex_hashes.add(p["hash"])

        all_msgs = [scene_msg] + tex_pushes

        if MP.is_host:
            with MP.conn_lock:
                conns = list(MP.client_conns.keys())
            for msg in all_msgs:
                for c in conns:
                    _send(c, msg)
        elif MP.my_sock:
            for msg in all_msgs:
                _send(MP.my_sock, msg)

        with MP.baseline_lock:
            MP.obj_baseline = dict(full)

        self.report({"INFO"},
                    f"Pushed {len(full)} objects + {len(tex_pushes)} textures.")
        return {"FINISHED"}


class MP_OT_ScanFolder(Operator):
    """Re-scan sync folder right now and repair all missing file references."""
    bl_idname = "mp.scan_folder"
    bl_label  = "Repair Missing Files Now"
    bl_options = {"REGISTER"}

    def execute(self, context):
        sf = context.scene.mp_settings.sync_folder
        if not sf:
            self.report({"WARNING"}, "No sync folder set.")
            return {"CANCELLED"}
        folder = bpy.path.abspath(sf)
        _scan_folder_for_missing(folder)
        self.report({"INFO"}, f"Missing-file scan complete on {folder}")
        return {"FINISHED"}


# ─────────────────────────────────────────────────────────────────────────────
#  PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

class MPSettings(PropertyGroup):
    user_name: StringProperty(name="Your Name", default="", maxlen=24)
    room_code: StringProperty(name="Room Code", default="")
    sync_mesh: BoolProperty(
        name="Sync Mesh Shapes",
        description="Sync vertex positions live (can lag on high-poly meshes)",
        default=False,
    )
    sync_folder: StringProperty(
        name="Texture Folder",
        subtype='DIR_PATH',
        description=(
            "Folder where synced textures are saved.\n"
            "Set this BEFORE connecting.\n"
            "Missing-file repair runs automatically."
        ),
        default="",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  UI PANEL
# ─────────────────────────────────────────────────────────────────────────────

class MP_PT_Main(Panel):
    bl_label       = "Multiplayer"
    bl_idname      = "MP_PT_Main"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Multiplayer"

    def draw(self, context):
        layout   = self.layout
        settings = context.scene.mp_settings

        # Status
        sb = layout.box()
        sb.label(text=MP.status,
                 icon="RADIOBUT_ON" if MP.connected else "RADIOBUT_OFF")

        layout.separator(factor=0.4)
        layout.label(text="Identity", icon="USER")
        layout.prop(settings, "user_name", text="Name")

        layout.separator(factor=0.4)
        layout.label(text="Options", icon="MODIFIER")
        layout.prop(settings, "sync_mesh")

        tf = layout.row(align=True)
        tf.prop(settings, "sync_folder", text="Texture Folder")

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
            with MP.peer_lock:
                snap = dict(MP.peer_data)
            for pname in snap:
                pr = pb.row(align=True)
                pr.label(text=f"  ● {pname}", icon="FUND")
                op = pr.operator("mp.snap_cam", text="", icon="VIEW_CAMERA")
                op.peer_name = pname
            if not snap:
                pb.label(text="  Waiting for peers…", icon="TIME")

            layout.separator(factor=0.3)
            layout.operator("mp.push_scene",   icon="EXPORT")
            layout.operator("mp.scan_folder",  icon="FILE_REFRESH")
            dr = layout.row(); dr.alert = True; dr.scale_y = 1.2
            dr.operator("mp.disconnect", icon="X")


# ─────────────────────────────────────────────────────────────────────────────
#  REGISTER / UNREGISTER
# ─────────────────────────────────────────────────────────────────────────────

_CLASSES = (
    MPSettings,
    MP_OT_Host,
    MP_OT_Join,
    MP_OT_Disconnect,
    MP_OT_CopyCode,
    MP_OT_SnapCam,
    MP_OT_PushScene,
    MP_OT_ScanFolder,
    MP_PT_Main,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mp_settings = bpy.props.PointerProperty(type=MPSettings)
    print("[MP] Blender Multiplayer Sync v4 registered.")


def unregister():
    _full_disconnect()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mp_settings
    print("[MP] Blender Multiplayer Sync unregistered.")


if __name__ == "__main__":
    register()
