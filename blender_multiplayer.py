# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          BLENDER MULTIPLAYER SYNC  v3.2  –  blender_multiplayer.py     ║
# ║  Full bidirectional sync: objects · cursor · camera · meshes · mats    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# INSTALL:  Edit > Preferences > Add-ons > Install…  →  pick this file  →  ✓
# USE:      N-panel (press N in 3D View) → "Multiplayer" tab
#
# NETWORK:  LAN works out of the box (TCP 19283).
#           Internet: host must port-forward 19283, or use ZeroTier / Tailscale.

bl_info = {
    "name":        "Blender Multiplayer Sync",
    "author":      "Claude (Updated)",
    "version":     (3, 2, 0),
    "blender":     (3, 0, 0),
    "location":    "View3D > N-Panel > Multiplayer",
    "description": "Real-time full-scene collaboration – objects, cursor, camera, meshes, materials",
    "category":    "3D View",
}

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

DEFAULT_PORT   = 19283
SYNC_INTERVAL  = 0.05          # 20 Hz update rate
LABEL_SIZE     = 14
CURSOR_R       = 10
CAM_R          = 7
PREC           = 5             # decimal places for transform rounding

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
    server_sock   = None
    my_sock       = None
    client_conns  = {}
    conn_lock     = threading.Lock()
    peer_data     = {}
    peer_lock     = threading.Lock()
    in_queue      = []
    q_lock        = threading.Lock()
    obj_baseline  = {}
    baseline_lock = threading.Lock()
    is_host       = False
    connected     = False
    status        = "Disconnected"
    colour_idx    = 0
    draw_handle   = None
    timer_active  = False
    cached_region    = None
    cached_region_3d = None

MP = _MP()


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECT STATE  helpers
# ─────────────────────────────────────────────────────────────────────────────

def _obj_to_state(obj, sync_mesh=False):
    """Serialize one object's full transform, type, material, and optional mesh data to a dict."""
    loc  = [round(v, PREC) for v in obj.location]
    mode = obj.rotation_mode
    if mode == 'QUATERNION':
        rot = [round(v, PREC) for v in obj.rotation_quaternion]
    elif mode == 'AXIS_ANGLE':
        rot = [round(v, PREC) for v in obj.rotation_axis_angle]
    else:
        rot = [round(v, PREC) for v in obj.rotation_euler]
    scale = [round(v, PREC) for v in obj.scale]

    state = {
        "type": obj.type,
        "loc": loc, 
        "rot_mode": mode, 
        "rot": rot, 
        "scale": scale
    }

    # Material Sync (Base Color from Principled BSDF)
    if obj.active_material and obj.active_material.use_nodes:
        try:
            nodes = obj.active_material.node_tree.nodes
            bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if bsdf:
                color = bsdf.inputs["Base Color"].default_value
                state["mat_color"] = [round(v, PREC) for v in color]
        except Exception:
            pass

    # Mesh Geometry Sync
    if sync_mesh and obj.type == 'MESH':
        if threading.current_thread() is threading.main_thread():
            if obj.mode == 'EDIT':
                obj.update_from_editmode()
        
        num_verts = len(obj.data.vertices)
        if num_verts > 0:
            verts = [0.0] * (num_verts * 3)
            obj.data.vertices.foreach_get("co", verts)
            state["verts"] = [round(v, PREC) for v in verts]
            state["faces"] = [list(p.vertices) for p in obj.data.polygons]

    return state


def _state_to_obj(obj, state):
    """Apply a serialized transform, material, and mesh shape to a Blender object."""
    try:
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

        # Material Sync
        if "mat_color" in state:
            mat_name = f"MP_Mat_{obj.name}"
            mat = obj.active_material
            if not mat or mat.name != mat_name:
                mat = bpy.data.materials.get(mat_name)
                if not mat:
                    mat = bpy.data.materials.new(name=mat_name)
                    mat.use_nodes = True
                if len(obj.material_slots) == 0:
                    obj.data.materials.append(mat)
                else:
                    obj.material_slots[0].material = mat
            
            try:
                nodes = mat.node_tree.nodes
                bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf:
                    bsdf.inputs["Base Color"].default_value = state["mat_color"]
            except Exception:
                pass

        # Mesh Vertex Sync & Reconstruction
        if "verts" in state and obj.type == 'MESH':
            verts = state["verts"]
            faces = state.get("faces", [])
            
            # Reconstruct geometry if topology doesn't match (New object or extreme edit)
            if len(verts) // 3 != len(obj.data.vertices) or len(faces) != len(obj.data.polygons):
                if obj.mode != 'EDIT':  # Can't rebuild base geometry safely while in edit mode
                    mesh = obj.data
                    mesh.clear_geometry()
                    v_tuples = [(verts[i], verts[i+1], verts[i+2]) for i in range(0, len(verts), 3)]
                    mesh.from_pydata(v_tuples, [], faces)
                    mesh.update()
            else:
                # Same topology, apply fast live updates
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
        print(f"[MP] apply_state({obj.name}): {e}")


def _state_key(s):
    """Hashable snapshot of a state dict for quick change detection."""
    v_hash = hash(tuple(s["verts"])) if "verts" in s else 0
    m_color = tuple(s["mat_color"]) if "mat_color" in s else 0
    deleted = s.get("deleted", False)
    return (tuple(s.get("loc", [])), s.get("rot_mode"), tuple(s.get("rot", [])), tuple(s.get("scale", [])), v_hash, m_color, deleted)


def _collect_full_scene(sync_mesh=False):
    """Serialize every object in the blend file."""
    return {obj.name: _obj_to_state(obj, sync_mesh) for obj in bpy.data.objects}


def _collect_delta(sync_mesh=False):
    """Return objects whose properties changed, or objects that were deleted."""
    changed = {}
    current_names = {obj.name for obj in bpy.data.objects}
    
    with MP.baseline_lock:
        baseline = dict(MP.obj_baseline)

    # Detect Deletions
    for name, state in baseline.items():
        if name not in current_names and not state.get("deleted"):
            changed[name] = {"deleted": True}

    # Detect Transformations / Mesh / Material changes
    for obj in bpy.data.objects:
        state = _obj_to_state(obj, sync_mesh)
        prev  = baseline.get(obj.name)
        if prev is None or _state_key(prev) != _state_key(state):
            changed[obj.name] = state
            
    return changed


def _apply_objects(objects_data, update_baseline=True):
    for name, state in objects_data.items():
        
        # Handle Deletion
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
            
        # Handle Creation
        obj = bpy.data.objects.get(name)
        if obj is None:
            obj_type = state.get("type", "EMPTY")
            try:
                if obj_type == 'MESH':
                    mesh = bpy.data.meshes.new(name)
                    obj = bpy.data.objects.new(name, mesh)
                elif obj_type == 'LIGHT':
                    light = bpy.data.lights.new(name, 'POINT')
                    obj = bpy.data.objects.new(name, light)
                else:
                    obj = bpy.data.objects.new(name, None)
                
                # Link to active collection
                bpy.context.collection.objects.link(obj)
            except Exception as e:
                print(f"[MP] Error creating object {name}: {e}")
                continue

        # Apply state updates
        if obj:
            _state_to_obj(obj, state)
            if update_baseline:
                with MP.baseline_lock:
                    MP.obj_baseline[name] = state


# ─────────────────────────────────────────────────────────────────────────────
#  NETWORK  utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def _encode_code(ip, port):
    import base64
    return base64.b32encode(f"{ip}:{port}".encode()).decode().rstrip("=")

def _decode_code(code):
    import base64
    code = code.upper().strip()
    code += "=" * ((8 - len(code) % 8) % 8)
    raw  = base64.b32decode(code).decode()
    ip, port_str = raw.rsplit(":", 1)
    return ip, int(port_str)

def _send(sock, data):
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
        if length > 50 * 1024 * 1024:      # 50 MB cap for heavy mesh data
            return None
        payload = _recvall(sock, length)
        if payload is None:
            return None
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None

def _broadcast(exclude, msg):
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

                _broadcast(conn, {"type": "peer_joined", "name": peer_name, "colour": list(colour)})

                with MP.peer_lock:
                    existing = {n: list(d.get("colour", [1, .5, 0, 1]))
                                for n, d in MP.peer_data.items() if n != peer_name}
                _send(conn, {"type": "peer_list", "peers": existing})

                # Safe attempt to get sync_mesh setting from the main scene
                sync_mesh = False
                try:
                    if bpy.data.scenes:
                        sync_mesh = bpy.data.scenes[0].mp_settings.sync_mesh
                except Exception:
                    pass

                full = _collect_full_scene(sync_mesh=sync_mesh)
                _send(conn, {"type": "full_state", "name": "__host__", "objects": full})

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

    except Exception as e:
        print(f"[MP] client-thread error: {e}")
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


def _host_accept_thread():
    while MP.connected and MP.server_sock:
        try:
            MP.server_sock.settimeout(1.0)
            conn, addr = MP.server_sock.accept()
            threading.Thread(target=_host_client_thread, args=(conn, addr), daemon=True).start()
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
            MP.status    = "Lost connection"
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

    for msg in msgs:
        mtype = msg.get("type")
        name  = msg.get("name", "???")

        if mtype == "peer_joined":
            colour = tuple(msg.get("colour", [1, .5, 0, 1]))
            with MP.peer_lock:
                if name not in MP.peer_data:
                    MP.peer_data[name] = {}
                MP.peer_data[name]["colour"] = colour
            MP.status = f"● {name} joined"
        elif mtype == "peer_list":
            for n, c in msg.get("peers", {}).items():
                with MP.peer_lock:
                    if n not in MP.peer_data:
                        MP.peer_data[n] = {}
                    MP.peer_data[n]["colour"] = tuple(c)
        elif mtype == "peer_left":
            with MP.peer_lock:
                MP.peer_data.pop(name, None)
            MP.status = f"● {name} left"
        elif mtype == "cursor":
            with MP.peer_lock:
                if name not in MP.peer_data:
                    MP.peer_data[name] = {}
                MP.peer_data[name]["cursor"] = msg.get("pos", [0, 0, 0])
        elif mtype == "camera":
            with MP.peer_lock:
                if name not in MP.peer_data:
                    MP.peer_data[name] = {}
                MP.peer_data[name]["cam_loc"] = msg.get("loc", [0, 0, 0])
                MP.peer_data[name]["cam_rot"] = msg.get("rot", [1, 0, 0, 0])
        elif mtype in ("objects", "full_state"):
            objects_data = msg.get("objects", {})
            if objects_data:
                _apply_objects(objects_data, update_baseline=True)


def _send_local_state():
    scene     = bpy.context.scene
    settings  = scene.mp_settings
    my_name   = settings.user_name.strip() or "Anonymous"
    sync_mesh = settings.sync_mesh

    # 3D cursor
    c = scene.cursor.location
    cursor_msg = {
        "type": "cursor", "name": my_name,
        "pos":  [round(c.x, PREC), round(c.y, PREC), round(c.z, PREC)],
    }

    # viewport camera
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

    # object & mesh deltas
    changed = _collect_delta(sync_mesh=sync_mesh)
    objects_msg = None
    if changed:
        objects_msg = {"type": "objects", "name": my_name, "objects": changed}
        with MP.baseline_lock:
            for name, state in changed.items():
                if state.get("deleted"):
                    MP.obj_baseline.pop(name, None)
                else:
                    MP.obj_baseline[name] = state

    messages = [m for m in [cursor_msg, camera_msg, objects_msg] if m]

    if MP.is_host:
        with MP.conn_lock:
            conns = list(MP.client_conns.keys())
        for msg in messages:
            for c in conns:
                _send(c, msg)
    elif MP.my_sock:
        for msg in messages:
            _send(MP.my_sock, msg)


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
#  VIEWPORT DRAW HANDLER
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
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, mathutils.Vector(cp))
            if p2:
                _draw_crosshair(p2, colour, CURSOR_R)
                _draw_label(name, p2[0] + CURSOR_R + 4, p2[1] - 6, colour)

        cl = data.get("cam_loc")
        if cl:
            p2 = bpy_extras.view3d_utils.location_3d_to_region_2d(region, rv3d, mathutils.Vector(cl))
            if p2:
                _draw_circle(p2, colour, CAM_R)
                _draw_label(f"{name} [cam]", p2[0] + CAM_R + 4, p2[1] - 6, (*colour[:3], 0.6))

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
    verts = [(center[0] + r * math.cos(2 * math.pi * i / segs), center[1] + r * math.sin(2 * math.pi * i / segs)) for i in range(segs)]
    _draw_poly(verts, colour, "LINE_LOOP")

def _draw_crosshair(center, colour, r):
    _draw_circle(center, colour, r)
    cx, cy = center
    sh  = gpu.shader.from_builtin("UNIFORM_COLOR")
    bat = batch_for_shader(sh, "LINES", {"pos": [(cx-r, cy), (cx+r, cy), (cx, cy-r), (cx, cy+r)]})
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
        MP.draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_peers, (), "WINDOW", "POST_PIXEL")

def _stop_draw_handler():
    if MP.draw_handle:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(MP.draw_handle, "WINDOW")
        except Exception:
            pass
        MP.draw_handle = None


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
    _stop_timer()
    _stop_draw_handler()
    MP.status     = "Disconnected"
    MP.colour_idx = 0


# ─────────────────────────────────────────────────────────────────────────────
#  OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

class MP_OT_Host(Operator):
    bl_idname, bl_label, bl_options = "mp.host_session", "Host Session", {"REGISTER"}
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
            MP.server_sock, MP.is_host, MP.connected = srv, True, True
            
            with MP.baseline_lock:
                MP.obj_baseline = _collect_full_scene(context.scene.mp_settings.sync_mesh)
                
            code = _encode_code(_get_local_ip(), DEFAULT_PORT)
            context.scene.mp_settings.room_code = code
            MP.status = "Hosting  •  0 peers"
            threading.Thread(target=_host_accept_thread, daemon=True).start()
            _start_timer(); _start_draw_handler()
            self.report({"INFO"}, f"Hosting!  Code: {code}")
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class MP_OT_Join(Operator):
    bl_idname, bl_label, bl_options = "mp.join_session", "Join Session", {"REGISTER"}
    def execute(self, context):
        settings = context.scene.mp_settings
        name, code = settings.user_name.strip(), settings.room_code.strip()
        if not name or not code:
            self.report({"ERROR"}, "Enter name and room code!")
            return {"CANCELLED"}
        if MP.connected:
            self.report({"WARNING"}, "Disconnect first.")
            return {"CANCELLED"}
        try:
            ip, port = _decode_code(code)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(6.0)
            sock.connect((ip, port))
            sock.settimeout(None)
            MP.my_sock, MP.is_host, MP.connected, MP.status = sock, False, True, "● Connected"
            _send(sock, {"type": "join", "name": name})
            threading.Thread(target=_client_recv_thread, daemon=True).start()
            _start_timer(); _start_draw_handler()
            self.report({"INFO"}, "Joined!")
        except Exception as e:
            self.report({"ERROR"}, f"Could not connect: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class MP_OT_Disconnect(Operator):
    bl_idname, bl_label, bl_options = "mp.disconnect", "Disconnect", {"REGISTER"}
    def execute(self, context):
        _full_disconnect()
        return {"FINISHED"}

class MP_OT_CopyCode(Operator):
    bl_idname, bl_label, bl_options = "mp.copy_code", "Copy", {"REGISTER"}
    def execute(self, context):
        context.window_manager.clipboard = context.scene.mp_settings.room_code
        return {"FINISHED"}

class MP_OT_SnapCam(Operator):
    bl_idname, bl_label, bl_options = "mp.snap_cam", "Snap to camera", {"REGISTER"}
    peer_name: StringProperty(default="")
    def execute(self, context):
        with MP.peer_lock: data = dict(MP.peer_data.get(self.peer_name, {}))
        loc, rot = data.get("cam_loc"), data.get("cam_rot")
        if not loc or not rot: return {"CANCELLED"}
        r3d = context.space_data.region_3d
        r3d.view_location = mathutils.Vector(loc)
        r3d.view_rotation = mathutils.Quaternion([rot[0], rot[1], rot[2], rot[3]])
        return {"FINISHED"}

class MP_OT_PushScene(Operator):
    bl_idname, bl_label, bl_options = "mp.push_scene", "Push Full Scene Now", {"REGISTER"}
    def execute(self, context):
        if not MP.connected: return {"CANCELLED"}
        my_name = context.scene.mp_settings.user_name.strip() or "Anonymous"
        full = _collect_full_scene(context.scene.mp_settings.sync_mesh)
        msg = {"type": "full_state", "name": my_name, "objects": full}
        if MP.is_host:
            with MP.conn_lock: conns = list(MP.client_conns.keys())
            for c in conns: _send(c, msg)
        elif MP.my_sock:
            _send(MP.my_sock, msg)
        with MP.baseline_lock:
            MP.obj_baseline = dict(full)
        self.report({"INFO"}, f"Pushed {len(full)} objects.")
        return {"FINISHED"}


# ─────────────────────────────────────────────────────────────────────────────
#  PROPERTIES & UI
# ─────────────────────────────────────────────────────────────────────────────

class MPSettings(PropertyGroup):
    user_name : StringProperty(name="Your Name", default="", maxlen=24)
    room_code : StringProperty(name="Room Code",  default="")
    sync_mesh : BoolProperty(name="Sync Mesh Shapes (Edit Mode)", default=False, 
                             description="Sync mesh vertex positions (can lag on high-poly meshes)")


class MP_PT_Main(Panel):
    bl_label       = "Multiplayer"
    bl_idname      = "MP_PT_Main"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Multiplayer"

    def draw(self, context):
        layout   = self.layout
        settings = context.scene.mp_settings

        sb = layout.box()
        sb.label(text=MP.status, icon="RADIOBUT_ON" if MP.connected else "RADIOBUT_OFF")

        layout.separator(factor=0.4)
        layout.label(text="Identity", icon="USER")
        layout.prop(settings, "user_name", text="Name")
        
        layout.separator(factor=0.4)
        layout.label(text="Features", icon="MODIFIER")
        layout.prop(settings, "sync_mesh")
        
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
            pb.label(text=f"  You  ({settings.user_name.strip() or 'Anonymous'})", icon="FUND")
            with MP.peer_lock: snap = dict(MP.peer_data)
            for pname in snap:
                pr = pb.row(align=True)
                pr.label(text=f"  ● {pname}", icon="FUND")
                op = pr.operator("mp.snap_cam", text="", icon="VIEW_CAMERA")
                op.peer_name = pname
            if not snap:
                pb.label(text="  Waiting for peers…", icon="TIME")

            layout.separator(factor=0.3)
            layout.operator("mp.push_scene", icon="EXPORT")
            dr = layout.row(); dr.alert = True; dr.scale_y = 1.2
            dr.operator("mp.disconnect", icon="X")

_CLASSES = (MPSettings, MP_OT_Host, MP_OT_Join, MP_OT_Disconnect, MP_OT_CopyCode, MP_OT_SnapCam, MP_OT_PushScene, MP_PT_Main)

def register():
    for cls in _CLASSES: bpy.utils.register_class(cls)
    bpy.types.Scene.mp_settings = bpy.props.PointerProperty(type=MPSettings)

def unregister():
    _full_disconnect()
    for cls in reversed(_CLASSES): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mp_settings

if __name__ == "__main__":
    register()