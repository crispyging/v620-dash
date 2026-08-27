#!/usr/bin/env python3
"""
V620 telemetry dashboard: system power + CPU load/temps + GPU temps + fans,
logged together so you can see what spikes power and how thermals track load.
  - Power:     IPMI DCMI (ipmitool)
  - CPU temps: IPMI sensors (CPU1/CPU2/System/Peripheral)
  - CPU util:  /proc/stat (host-wide)
  - GPU temps: amdgpu sysfs (edge/junction/mem), PER CARD, keyed by PCI address
  - GPU memory: amdgpu sysfs VRAM/GTT counters + board power, PER CARD, drawn
                as an occupancy bar with one segment per resident model
  - Fans:      IPMI sensors, PER FAN (not just the max - see below)
  - Fan control: reads gpu-fan-control's fan-status.json, writes fan-curve.conf

WHERE THE VRAM BAR'S NUMBERS COME FROM
  The LENGTH of the bar is always amdgpu's own mem_info_vram_used. The named
  pieces inside it are labels for that length and never define it, so the panel
  cannot be talked into showing a card as empty when it isn't. Anything the
  labels can't account for stays visible as "other". See the VRAM occupancy
  section for the three sources and the order they're trusted in.

WHY PER-FAN MATTERS
  v1 logged only fan_rpm_max. That column structurally cannot show you the
  problem you're most likely to have: the BMC declares a fan FAILED when it
  drops below its Lower Critical threshold, and blasts every zone to 100%.
  The fan that trips it is by definition the SLOWEST one - which a max column
  hides completely. fan_rpm_min is the diagnostic number.

HOW FAN CONTROL WORKS FROM HERE
  This app never touches /dev/ipmi0 for fan control. gpu-fan-control owns the
  BMC; two writers on the fan zones is exactly the failure mode we're chasing.
  Instead this app writes fan-curve.conf, which the daemon re-reads every cycle
  and re-validates with its own whitelist parser. One writer, no new trust.

Runs in a privileged container with /dev/ipmi0 passed in. No sign-in, no cloud.
"""
import subprocess, re, threading, time, os, csv, datetime, glob, json, tempfile
import calendar
import urllib.request, urllib.error
from flask import Flask, jsonify, Response, request

# Bumped by hand whenever this file changes in a way you'd want to see landed.
# It is printed under the page title and returned by /api/power, so "did my new
# app.py actually take effect" is a five-second question instead of a hunt: if
# the version on the page isn't the version in the file you just copied, the
# container is running a different file (wrong path, or a Dockerfile baking an
# old copy into the image - see the README).
APP_VERSION = "v620-dash 0.1.0 (2026-08-27) · public build"

POLL      = int(os.environ.get("POLL_SECONDS", "5"))
DATA_DIR  = os.environ.get("DATA_DIR", "/data")
FAN_DIR   = os.environ.get("FAN_DIR", "/fanctl")   # gpu-fan-control's folder
MAXPOINTS = 360

FAN_STATUS = os.path.join(FAN_DIR, "fan-status.json")
FAN_CONF   = os.path.join(FAN_DIR, "fan-curve.conf")
# Same override the fan daemon takes, so both can be pointed at a fake sysfs
# tree for testing without touching the real hardware.
GPU_SYS    = os.environ.get("GPU_SYS", "/sys/class/hwmon")

# --- VRAM panel ------------------------------------------------------------
# Ollama's HTTP API, used ONLY for the names of the models it currently holds
# resident. It is never trusted for how full a card is - amdgpu sysfs is the
# ground truth for that, and it accounts for ComfyUI, the framebuffer and
# anything else Ollama has no idea about. Ollama supplies labels, not sizes.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
# Some boxes run Ollama; some are llama.cpp-lane only. An empty OLLAMA_URL
# means "lane-only": every Ollama-touching endpoint refuses cleanly instead
# of erroring, and the UI hides the Ollama furniture (Models tab, ollama-pull
# form, HF destination picker, bench load driver) rather than showing panels
# that can never work. Set a real OLLAMA_URL to get all of it back.
OLLAMA_ENABLED = bool(OLLAMA_URL)
OLLAMA_OFF_MSG = ("Ollama is not configured on this box (OLLAMA_URL is empty "
                  "in the compose). This is a llama.cpp-lane box — use the "
                  "Lane tab / HF import instead.")
# Which physical card Ollama is pinned to: a PCI address ("0000:04:00.0") or
# the short slot form ("04:00.0"). This is the answer to a problem that has no
# clean solution - /api/ps does not say which GPU a model landed on, and the
# HIP device index Ollama was started with is NOT the DRM card number, so
# HIP_VISIBLE_DEVICES=0 cannot be turned into a card by counting. If you know
# the answer, put it here. Leave it empty and the app works it out or says it
# is guessing.
OLLAMA_PIN = os.environ.get("OLLAMA_GPU_PCI", "").strip()
# Overridable for the same reason GPU_SYS is: so the fdinfo reader can be
# pointed at a fake tree in tests.
PROC_ROOT  = os.environ.get("PROC_ROOT", "/proc")
# The thermal-test tab is hidden by default. Nothing about it was removed —
# the state machine, the abort logic and the API routes are all still here and
# still work. This only controls whether the tab is rendered. Set
# SHOW_THERMAL_TEST=1 in the compose to get it back.
SHOW_THERMAL_TEST = os.environ.get("SHOW_THERMAL_TEST", "0").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# SHAREABLE BUILD: site name, access control, and dangerous-feature gating.
#
# This dashboard reads real hardware and, if you turn it on, RUNS SCRIPTS ON
# THE HOST AS ROOT. It has no user model and no CSRF protection. Treat the
# token as a root password for the machine.
#
#   SITE_NAME       what the page calls this box            (default "V620")
#   DASH_TOKEN      shared secret required for every request
#   DASH_BIND       interface to listen on   (default 0.0.0.0)
#   DASH_PORT       port                     (default 8500)
#   ENABLE_SCRIPTS  1 to expose the host-script runner  (default OFF)
#
# If DASH_TOKEN is empty the app REFUSES to listen on anything but loopback.
# That is deliberate: an unauthenticated instance on a LAN is a remote root
# shell with a web UI.
# ---------------------------------------------------------------------------
SITE_NAME   = os.environ.get("SITE_NAME", "V620").strip() or "V620"
DASH_TOKEN  = os.environ.get("DASH_TOKEN", "").strip()
DASH_BIND   = os.environ.get("DASH_BIND", "0.0.0.0").strip()
DASH_PORT   = int(os.environ.get("DASH_PORT", "8500"))
ENABLE_SCRIPTS = os.environ.get("ENABLE_SCRIPTS", "0").lower() in ("1", "true", "yes")

import hmac as _hmac
import secrets as _secrets

_COOKIE = "v620dash"

def _token_ok(supplied):
    """Constant-time compare. No token configured -> loopback-only, see below."""
    if not DASH_TOKEN:
        return True
    return bool(supplied) and _hmac.compare_digest(str(supplied), DASH_TOKEN)

_LOGIN_PAGE = """<!doctype html><meta charset=utf-8>
<title>__SITE_NAME__ — sign in</title>
<style>body{background:#0b0e12;color:#dfe6ee;font:15px/1.5 system-ui,sans-serif;
display:grid;place-items:center;height:100vh;margin:0}
form{background:#151b23;padding:26px 28px;border:1px solid #263041;border-radius:10px;min-width:320px}
h1{font-size:16px;margin:0 0 14px}input{width:100%;padding:9px;border-radius:6px;
border:1px solid #2b3648;background:#0b0e12;color:#dfe6ee;box-sizing:border-box}
button{margin-top:12px;width:100%;padding:9px;border:0;border-radius:6px;
background:#2f6fd0;color:#fff;font-weight:600;cursor:pointer}
.e{color:#e0575b;font-size:13px;margin-top:10px}</style>
<form method=post><h1>__SITE_NAME__ — telemetry</h1>
<input name=token type=password placeholder="access token" autofocus>
<button>Sign in</button>__ERR__</form>"""

app  = Flask(__name__)
lock = threading.Lock()

@app.route("/login", methods=["GET", "POST"])
def _login():
    err = ""
    if request.method == "POST":
        if _token_ok((request.form.get("token") or "").strip()):
            r = app.make_response(Response("", 302, {"Location": "/"}))
            # Session cookie: not persisted to disk, not sent cross-site.
            r.set_cookie(_COOKIE, DASH_TOKEN or "open", httponly=True,
                         samesite="Strict", max_age=60 * 60 * 12)
            return r
        err = '<div class="e">Wrong token.</div>'
    page = _LOGIN_PAGE.replace("__SITE_NAME__", SITE_NAME).replace("__ERR__", err)
    return Response(page, mimetype="text/html")

@app.before_request
def _require_token():
    if not DASH_TOKEN or request.path == "/login":
        return None
    if _token_ok(request.cookies.get(_COOKIE)):
        return None
    # Bearer token for scripted/API access (curl, monitoring).
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and _token_ok(auth[7:].strip()):
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "unauthorized"}), 401
    return Response("", 302, {"Location": "/login"})

state = {"current": None, "ts": None, "samples": [], "logging": False,
         "logfile": None, "stats": None, "error": None,
         "gpus": [], "cpu": {}, "util": None, "fans": {}, "fan": None,
         "gpumem": [], "gpumem_meta": {}, "ram": {},
         "cols": [], "gpu_ids": [], "fan_ids": [],
         # Sticky supersets: every GPU / fan id seen since startup, in a stable
         # order. The CSV schema is fixed when logging starts, so if a card is
         # momentarily absent at that instant we would silently lose its column
         # for the whole run. Accumulating means a transient dropout can't do
         # that - same reasoning as the daemon's SEEN_GPUS failsafe.
         "seen_gpu_ids": [], "seen_fan_ids": []}

# ---------------- readers ----------------
def read_watts():
    try:
        out = subprocess.run(["ipmitool","dcmi","power","reading"],
                             capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "Instantaneous" in line:
                m = re.search(r"(\d+)\s*Watts", line)
                if m: return int(m.group(1)), None
        return None, "no DCMI reading"
    except Exception as e:
        return None, str(e)

def read_temps():
    """IPMI temperature sensors -> {'CPU1':40,'CPU2':41,'System':35,...}."""
    temps = {}
    try:
        out = subprocess.run(["ipmitool","sdr","type","Temperature"],
                             capture_output=True, text=True, timeout=12)
        lines = out.stdout.splitlines()
        if not lines:
            out = subprocess.run(["ipmitool","sdr"], capture_output=True, text=True, timeout=12)
            lines = out.stdout.splitlines()
        for line in lines:
            if "degrees C" not in line:
                continue
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*degrees C", line)
            if not m:
                continue
            name = line.split("|")[0].strip().replace(" Temp","").strip()
            temps[name] = float(m.group(1))
    except Exception:
        pass
    return temps

def read_fans():
    """IPMI fan sensors -> {'FAN1':3600,'FAN2':3700,...} in RPM (skips 'no reading')."""
    fans = {}
    try:
        out = subprocess.run(["ipmitool","sdr","type","Fan"],
                             capture_output=True, text=True, timeout=12)
        for line in out.stdout.splitlines():
            if "RPM" not in line:
                continue
            m = re.search(r"(\d+)\s*RPM", line)
            if not m:
                continue
            name = line.split("|")[0].strip()
            fans[name] = int(m.group(1))
    except Exception:
        pass
    return fans

def make_util_reader():
    prev = {"t": None, "i": None}
    def read():
        try:
            with open("/proc/stat") as f:
                p = [int(x) for x in f.readline().split()[1:]]
            idle = p[3] + (p[4] if len(p) > 4 else 0)
            total = sum(p)
            pt, pi = prev["t"], prev["i"]
            prev["t"], prev["i"] = total, idle
            if pt is None:
                return None
            dt, di = total - pt, idle - pi
            return None if dt <= 0 else round(100.0*(dt-di)/dt, 1)
        except Exception:
            return None
    return read

def read_gpu_temps():
    """Every amdgpu card, keyed by PCI address.

    Sorted by PCI address, NOT by hwmon index. hwmon numbers are handed out in
    probe order and can swap between boots (and sort wrong lexically anyway -
    hwmon10 before hwmon2). The PCI address belongs to the physical slot, so
    "GPU 0" in a log from last week means the same card today.
    """
    gpus = []
    for hw in glob.glob(os.path.join(GPU_SYS, "hwmon*")):
        try:
            if open(os.path.join(hw,"name")).read().strip() != "amdgpu":
                continue
        except Exception:
            continue
        temps, card, pci = {}, "", ""
        try:
            pci = os.path.basename(os.path.realpath(os.path.join(hw,"device")))
        except Exception:
            pass
        for c in glob.glob(os.path.join(hw,"device","drm","card*")):
            card = os.path.basename(c); break
        for tf in sorted(glob.glob(os.path.join(hw,"temp*_input"))):
            base = tf[:-6]
            try: lbl = open(base+"_label").read().strip().lower()
            except Exception: lbl = os.path.basename(base)
            try: temps[lbl] = round(int(open(tf).read().strip())/1000.0, 1)
            except Exception: pass
        # amdgpu normally labels these (temp1=edge, temp2=junction). If the
        # labels are missing we fall back to the positional meaning, which is
        # exactly what gpu-fan-control.sh uses - otherwise the daemon would be
        # reading a junction temp the dashboard couldn't name, and the CSV's
        # gpuN_junction_c columns would come out empty.
        if temps and "junction" not in temps:
            if "temp2" in temps: temps["junction"] = temps["temp2"]
            if "temp1" in temps and "edge" not in temps:
                temps["edge"] = temps["temp1"]
        if temps:
            temps["pci"]  = pci or os.path.basename(hw)
            temps["card"] = card or "amdgpu"
            # short human label: the bus part of the PCI address is what's
            # printed on the riser diagram, e.g. 0000:04:00.0 -> "04:00.0"
            temps["slot"] = (pci[5:] if pci.startswith("0000:") else pci) or temps["card"]
            gpus.append(temps)
    gpus.sort(key=lambda g: g["pci"])
    return gpus

def hottest_gpu(gpus):
    hot = None
    for g in gpus:
        v = g.get("junction", g.get("edge"))
        if v is not None:
            hot = v if hot is None else max(hot, v)
    return hot

# ---------------- VRAM occupancy ----------------
# Three sources, in descending order of how much they can be trusted:
#
#   1. amdgpu sysfs   - how many bytes the driver says are allocated on each
#                       card. Ground truth. The bar's length is only ever this.
#   2. /proc fdinfo   - which PROCESS holds them, per card. Authoritative when
#                       readable, which needs `pid: host` in the compose file.
#   3. Ollama /api/ps - human names for the models. Knows nothing about
#                       ComfyUI, nothing about the framebuffer, and does not
#                       say which card anything landed on.
#
# The panel never lets a lower source overrule a higher one. Whatever sysfs
# says is allocated but nothing can name shows up as "other", and that gap is
# the most interesting number on the panel when it is large.

def _readint(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None

def read_gpu_mem():
    """Per-card VRAM, GTT, busy% and power draw, keyed by PCI address.

    The counters live in the PCI device directory, which is where the hwmon
    node's `device` symlink already points - the same hop read_gpu_temps()
    makes to find `device/drm/card*`. Going through hwmon rather than straight
    to /sys/class/drm keeps GPU_SYS as the single override for the whole app.

    Power is read here rather than in read_gpu_temps() because it comes off the
    hwmon node itself (power1_*, microwatts) rather than the device directory.
    It is READ ONLY. Nothing in this app writes power1_cap - gpu-fan-control
    owns the card the way it owns the fans, and two writers is the failure mode
    this whole project exists to avoid.
    """
    out = {}
    for hw in glob.glob(os.path.join(GPU_SYS, "hwmon*")):
        try:
            if open(os.path.join(hw, "name")).read().strip() != "amdgpu":
                continue
        except Exception:
            continue
        dev = os.path.join(hw, "device")
        try:
            pci = os.path.basename(os.path.realpath(dev))
        except Exception:
            continue
        d = {}
        vt = _readint(os.path.join(dev, "mem_info_vram_total"))
        if vt:
            vu = _readint(os.path.join(dev, "mem_info_vram_used")) or 0
            d["vram_total"] = vt
            d["vram_used"]  = vu
            d["vram_free"]  = max(0, vt - vu)
        # GTT is host RAM the card can reach over PCIe. A model that spills
        # into it has not failed to load - it has quietly become several times
        # slower, which is worth being able to see.
        gt = _readint(os.path.join(dev, "mem_info_gtt_total"))
        if gt:
            d["gtt_total"] = gt
            d["gtt_used"]  = _readint(os.path.join(dev, "mem_info_gtt_used")) or 0
        for key, fn in (("busy", "gpu_busy_percent"), ("mem_busy", "mem_busy_percent")):
            v = _readint(os.path.join(dev, fn))
            if v is not None:
                d[key] = v
        # power1_average is the driver's own rolling figure and is the one to
        # prefer; power1_input is instantaneous and absent on some SKUs.
        for fn in ("power1_average", "power1_input"):
            v = _readint(os.path.join(hw, fn))
            if v is not None:
                d["power_w"] = round(v / 1e6, 1)
                break
        for key, fn in (("cap_w", "power1_cap"), ("cap_max_w", "power1_cap_max")):
            v = _readint(os.path.join(hw, fn))
            if v is not None:
                d[key] = round(v / 1e6)
        if d:
            out[pci] = d
    return out

# ---------------- system RAM ----------------
# Added 2026-08-27, the day Qwen3.8-Flash-Next made host RAM part of the
# model-serving story (~50 GB N-gram table lives there BY DESIGN). One slim
# bar, deliberately unlike the VRAM panel's density — see renderRam().
#
# The accounting is chosen so the bar's pieces can NEVER double-count:
#   * a process segment is its RssAnon — memory that is privately its own.
#   * page cache is its own band. A GGUF mmap'd by llama-server lives THERE,
#     not in the process's anon segment; the legend annotates big mappers
#     with their RssFile ("+N mapped") so the connection is visible without
#     counting those bytes twice.
#   * free is MemFree; "avail" (what a new allocation could actually get,
#     cache being reclaimable) is reported as a number, not a band.
RAM_SYS = os.environ.get("RAM_SYS", "/sys/devices/system/node")

def read_ram():
    mi = {}
    try:
        with open(os.path.join(PROC_ROOT, "meminfo")) as f:
            for line in f:
                k, _, v = line.partition(":")
                mi[k.strip()] = int(v.strip().split()[0]) * 1024  # kB -> bytes
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}
    total = mi.get("MemTotal", 0)
    free = mi.get("MemFree", 0)
    avail = mi.get("MemAvailable", 0)
    cache = mi.get("Buffers", 0) + mi.get("Cached", 0) + mi.get("SReclaimable", 0)
    used = max(0, total - free - cache)          # anon + shmem + kernel
    # Top consumers by private (anon) memory. With pid: host this walk sees
    # the whole box; without it, only this container - same graceful shape
    # as the VRAM panel's fdinfo reader.
    procs = []
    try:
        pids = [p for p in os.listdir(PROC_ROOT) if p.isdigit()]
    except Exception:
        pids = []
    for pid in pids:
        anon = rfile = 0
        try:
            with open(os.path.join(PROC_ROOT, pid, "status")) as f:
                for line in f:
                    if line.startswith("RssAnon"):
                        anon = int(line.split()[1]) * 1024
                    elif line.startswith("RssFile"):
                        rfile = int(line.split()[1]) * 1024
                    elif line.startswith("VmSwap"):
                        break
        except Exception:
            continue
        if anon + rfile < (1 << 29):             # < 512 MiB: not worth a name
            continue
        procs.append({"pid": int(pid), "comm": _proc_text(pid, "comm"),
                      "anon": anon, "mapped": rfile})
    procs.sort(key=lambda p: -(p["anon"] + p["mapped"]))
    # NUMA free per node — dual socket, and Flash-Next's table + experts care
    # which node they land on relative to the GPUs' node.
    nodes = []
    try:
        for nd in sorted(os.listdir(RAM_SYS)):
            if not nd.startswith("node"):
                continue
            nf = nt = None
            with open(os.path.join(RAM_SYS, nd, "meminfo")) as f:
                for line in f:
                    if "MemFree" in line:
                        nf = int(line.split()[-2]) * 1024
                    elif "MemTotal" in line:
                        nt = int(line.split()[-2]) * 1024
            if nf is not None:
                nodes.append({"node": nd, "free": nf, "total": nt})
    except Exception:
        pass
    return {"total": total, "free": free, "avail": avail, "cache": cache,
            "used": used, "swap_total": mi.get("SwapTotal", 0),
            "swap_free": mi.get("SwapFree", 0),
            "procs": procs[:4], "nodes": nodes}


# --- Ollama ----------------------------------------------------------------
# Backed off on failure. Ollama being down is a normal state - it is not a
# dependency of this dashboard - and a 2.5s connect timeout on every 5s poll
# would drag power, temps and fans down with it for a service that is merely
# off.
_ollama = {"next": 0.0, "fails": 0, "data": None, "error": None}

# Which model was last worked on, without an endpoint that reports it.
#
# Ollama has no "current requests" API - /api/ps is residency, and a model
# idling out its keep-alive looks exactly like one mid-generation. But the
# keep-alive timer itself is the tell: `expires_at` is pushed forward every
# time a request touches the model, and nothing else moves it. So a change in
# that string, poll over poll, means that model served something in the last
# few seconds. It needs no parsing and no clock agreement with the server -
# only that the value differs from the one seen last time.
#
# What this does NOT tell you is who asked. Everything arrives through Open
# WebUI, so Ollama sees one client for the whole house. It tells you which
# MODEL is working, which is the answerable half of the question: if you and
# two people are on different models, that is the answer.
_ollama_seen = {}                  # name -> {"exp": str, "t": float}
ACTIVE_WINDOW = 15.0               # generous next to a 5s poll

def read_ollama_ps():
    """(models, error). Each model: name, size, size_vram, expires_at, used_ago."""
    if not OLLAMA_ENABLED:
        # Not an error — there is nothing to ask. Returning (None, None)
        # keeps the VRAM panel quiet instead of warning every poll that a
        # service which was never installed is not answering.
        return None, None
    now = time.time()
    if now < _ollama["next"]:
        return _ollama["data"], _ollama["error"]
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/ps",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=2.5) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
        models = []
        for m in (body.get("models") or []):
            name = m.get("name") or m.get("model") or "?"
            exp = str(m.get("expires_at") or "")
            prev = _ollama_seen.get(name)
            if prev is None:
                # First sighting. We have no idea when it was last used and
                # will not pretend otherwise - used_ago stays None until the
                # value actually moves under observation.
                _ollama_seen[name] = {"exp": exp, "t": None}
            elif exp != prev["exp"]:
                _ollama_seen[name] = {"exp": exp, "t": now}
            seen_t = _ollama_seen[name]["t"]
            models.append({"name": name,
                           "size": int(m.get("size") or 0),
                           "size_vram": int(m.get("size_vram") or 0),
                           "expires_at": m.get("expires_at"),
                           "used_ago": (None if seen_t is None else round(now - seen_t, 1)),
                           "active": (seen_t is not None and now - seen_t <= ACTIVE_WINDOW)})
        live = {m["name"] for m in models}
        for gone in [n for n in _ollama_seen if n not in live]:
            _ollama_seen.pop(gone, None)      # unloaded; its history is meaningless
        models.sort(key=lambda m: -m["size_vram"])
        _ollama.update(data=models, error=None, fails=0, next=0.0)
        return models, None
    except Exception as e:
        _ollama["fails"] += 1
        back = min(60.0, 5.0 * (2 ** min(_ollama["fails"] - 1, 4)))  # 5,10,20,40,60
        _ollama.update(data=None, error=str(e), next=now + back)
        return None, _ollama["error"]

# --- /proc fdinfo ----------------------------------------------------------
# amdgpu reports allocations in KiB with an explicit suffix. Older kernels use
# drm-memory-vram; newer ones add drm-resident-memory-vram, which is the more
# accurate of the two (it excludes evicted pages), so it is preferred.
_FDINFO_VRAM_KEYS = ("drm-resident-memory-vram", "drm-memory-vram")
_UNIT = {"B": 1, "KIB": 1024, "MIB": 1024 ** 2, "GIB": 1024 ** 3}

def _fdinfo_bytes(v):
    parts = str(v).split()
    try:
        n = float(parts[0])
    except Exception:
        return None
    # The DRM fdinfo spec makes the suffix optional; amdgpu always emits KiB,
    # so that is the right default rather than bytes.
    return int(n * _UNIT.get(parts[1].upper() if len(parts) > 1 else "KIB", 1024))

# --- per-process GPU time --------------------------------------------------
# amdgpu publishes cumulative nanoseconds of engine time per DRM client in the
# same fdinfo file the VRAM figures come from. The delta between two reads,
# over the wall-clock time between them, is that client's share of the card -
# which is what answers "is the card busy with MY model or the other one".
#
# CAVEAT, and it is a real one: these counters track work submitted through
# amdgpu's DRM scheduler. ROCm compute goes through KFD user-mode queues, which
# on some kernels bypass that accounting entirely and leave the counters
# stubbornly at zero while the card is plainly at 99%. So this is reported when
# it moves and openly reported as unavailable when it doesn't, rather than
# being dressed up as an idle model. The Ollama-side activity signal below
# covers that case, and the card-level busy figure was never affected.
_ENGINE_PREFIX = "drm-engine-"
# dec/enc/jpeg are video blocks - Plex transcoding, not model work - and are
# left out so a transcode can't read as inference.
_ENGINE_KEYS = ("drm-engine-gfx", "drm-engine-compute")
_busy = {}                    # (pci, client id) -> {ns, t, pct}
_BUSY_MIN_WINDOW = 2.0        # never divide by a window shorter than this
_BUSY_MAX_AGE = 300.0

def _engine_ns(info):
    """Summed gfx+compute nanoseconds, or None if the kernel doesn't publish them."""
    tot = None
    for k in _ENGINE_KEYS:
        v = info.get(k)
        if v is None:
            continue
        try:
            n = int(str(v).split()[0])
        except Exception:
            continue
        tot = n if tot is None else tot + n
    return tot

def _client_busy(key, ns, now):
    """Percent of one card this client used since the last sample.

    Deliberately holds the previous answer rather than recomputing when called
    again quickly. Two browser tabs polling this endpoint would otherwise
    measure over a 100 ms window and produce numbers that swing between 0 and
    100 with nothing real behind them.
    """
    if ns is None:
        return None
    prev = _busy.get(key)
    if prev is None:
        _busy[key] = {"ns": ns, "t": now, "pct": None}
        return None            # no baseline yet. Say so; don't report zero.
    dt = now - prev["t"]
    if dt < _BUSY_MIN_WINDOW:
        return prev["pct"]
    dns = ns - prev["ns"]
    pct = None
    if dns >= 0:               # a counter that went backwards means a new
        pct = 100.0 * (dns / 1e9) / dt   # client reused the id. Skip that one.
        # gfx and compute are summed and can legitimately overlap, so cap
        # rather than print 140% of a card.
        pct = round(max(0.0, min(100.0, pct)), 1)
    _busy[key] = {"ns": ns, "t": now, "pct": pct}
    return pct

def _busy_prune(now):
    for k in [k for k, v in _busy.items() if now - v["t"] > _BUSY_MAX_AGE]:
        _busy.pop(k, None)

def _proc_text(pid, name, sep=None):
    try:
        with open(os.path.join(PROC_ROOT, pid, name), "rb") as f:
            s = f.read()
        if sep:
            s = s.replace(sep, b" ")
        return s.decode("utf-8", "replace").strip()
    except Exception:
        return ""

def read_drm_clients():
    """{pci: [{pid, comm, cmd, bytes}]} - who is holding VRAM on which card.

    This is the only source that can attribute VRAM to a consumer. Ollama's
    /api/ps knows about Ollama's own models and nothing else, so on a box where
    ComfyUI has a card to itself the API can never explain where 18 GB went.

    Returns {} - not an error - when /proc shows only this container's own
    processes, which is the default. `pid: host` in the compose file is what
    turns this on, and it grants nothing to a container that is already
    `privileged: true`.

    Dedup is on drm-client-id, NOT on file descriptor. A process typically
    holds several fds against the same DRM client and every one of them reports
    the full allocation; summing files instead of clients turns a 20 GB model
    into 60 GB and makes the bar nonsense.
    """
    out, seen, now = {}, set(), time.time()
    try:
        pids = [p for p in os.listdir(PROC_ROOT) if p.isdigit()]
    except Exception:
        return out
    for pid in pids:
        fddir = os.path.join(PROC_ROOT, pid, "fd")
        try:
            fds = os.listdir(fddir)
        except Exception:
            continue                      # exited, or not ours to read
        for fd in fds:
            # readlink is one cheap syscall and prunes almost every fd on the
            # box, so we only ever open fdinfo for descriptors that really do
            # point at a DRM node.
            try:
                if not os.readlink(os.path.join(fddir, fd)).startswith("/dev/dri/"):
                    continue
            except Exception:
                continue
            try:
                with open(os.path.join(PROC_ROOT, pid, "fdinfo", fd)) as f:
                    txt = f.read()
            except Exception:
                continue
            info = {}
            for line in txt.splitlines():
                k, sep, v = line.partition(":")
                if sep:
                    info[k.strip()] = v.strip()
            pci = info.get("drm-pdev")
            if not pci:
                continue
            key = (pci, info.get("drm-client-id") or ("pid" + pid))
            if key in seen:
                continue
            byt = None
            for k in _FDINFO_VRAM_KEYS:
                if k in info:
                    byt = _fdinfo_bytes(info[k])
                    break
            if not byt:
                continue
            seen.add(key)
            eng = _engine_ns(info)
            out.setdefault(pci, []).append(
                {"pid": int(pid), "comm": _proc_text(pid, "comm"),
                 "cmd": _proc_text(pid, "cmdline", b"\0"), "bytes": byt,
                 "eng": eng is not None,
                 "gpu_pct": _client_busy(key, eng, now)})
    _busy_prune(now)
    for v in out.values():
        v.sort(key=lambda c: -c["bytes"])
    return out

_drm = {"next": 0.0, "misses": 0, "data": {}}

def read_drm_clients_cached():
    """Same, but stops paying for a scan that structurally cannot succeed.

    Without `pid: host` this walk can never find anything, and there is no
    point doing it every five seconds forever. Three empty passes and it drops
    to once a minute - still often enough that the panel lights up on its own
    a minute after the compose file is fixed, rather than needing a restart.
    """
    now = time.time()
    if now < _drm["next"]:
        return _drm["data"]
    d = read_drm_clients()
    if d:
        _drm.update(data=d, misses=0, next=0.0)
    else:
        _drm["misses"] += 1
        _drm.update(data={}, next=now + (60.0 if _drm["misses"] >= 3 else 0.0))
    return _drm["data"]

# --- putting the bar together ----------------------------------------------
_CONSUMERS = (("ollama", "Ollama"), ("comfy", "ComfyUI"), ("llama-server", "llama.cpp"),
              ("llama-cli", "llama.cpp"), ("vllm", "vLLM"), ("plex", "Plex"),
              ("ffmpeg", "ffmpeg"), ("xorg", "desktop"), ("gnome-shell", "desktop"))

def _consumer_label(c):
    low = ((c.get("comm") or "") + " " + (c.get("cmd") or "")).lower()
    for needle, name in _CONSUMERS:
        if needle in low:
            return name
    return c.get("comm") or ("pid %d" % c["pid"])

# --- weights blob -> model name: the exact join ----------------------------
# /api/ps reports a model's MANIFEST digest, never the weights blob the runner
# actually has open, so those two cannot be joined on an id and the original
# code matched on size instead. Size matching breaks the moment
# OLLAMA_SCHED_SPREAD splits one model across two cards: /api/ps reports the
# 24 GiB total while fdinfo reports 12.3 here and 11.8 there, and neither half
# is within tolerance of the whole, so the segment keeps the generic label.
#
# There is a real id available. The runner process is started with
#     --model /root/.ollama/models/blobs/sha256-<digest>
# and every manifest under the models directory names that same digest as its
# layer of mediaType application/vnd.ollama.image.model. Reading the manifests
# gives digest -> name:tag. That join survives a split allocation, and it
# survives two resident models being nearly the same size, which is precisely
# where a size match stops merely failing and starts being wrong.
#
# It needs the models directory mounted (OLLAMA_MODELS_DIR). Without it, or on
# a runner whose command line doesn't carry the path, naming falls through to
# the two heuristic tiers below and says so.
_RUNNER_BLOB_RE = re.compile(r"blobs[/\\]sha256[-:]([0-9a-f]{64})")

_manifests = {"next": 0.0, "data": {}, "forced": 0.0}

def _manifest_name(root, path):
    """<root>/registry.ollama.ai/library/qwen3-35b/latest -> qwen3-35b:latest"""
    rel = os.path.relpath(path, root).replace(os.sep, "/").split("/")
    if len(rel) < 2:
        return None
    tag, parts = rel[-1], rel[:-1]
    if parts and "." in parts[0]:          # the registry host component
        parts = parts[1:]
    if parts[:1] == ["library"]:           # ollama.com's default namespace
        parts = parts[1:]
    return ("/".join(parts) + ":" + tag) if parts else None

def read_model_blob_names(force=False):
    """{weights-blob digest: "name:tag"}, read from the manifests on disk.

    Cheap - a few dozen small JSON files - but re-read at most once a minute
    because the answer only changes when a pull finishes or a model is deleted.
    `force` is for exactly that case: a digest we've never seen means a model
    arrived since the last read, so re-read immediately instead of showing the
    generic label for up to a minute. Rate-limited so an unresolvable digest
    (a loose GGUF run straight off disk, say) can't turn every poll into a walk.
    """
    now = time.time()
    if force and now - _manifests["forced"] < 10.0:
        force = False
    if not force and now < _manifests["next"]:
        return _manifests["data"]
    if force:
        _manifests["forced"] = now
    root = os.path.join(OLLAMA_MODELS_DIR or "", "manifests")
    out = {}
    if os.path.isdir(root):
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "rb") as f:
                        man = json.loads(f.read().decode("utf-8", "replace"))
                except Exception:
                    continue               # not a manifest, or not readable
                name = _manifest_name(root, p)
                if not name or not isinstance(man, dict):
                    continue
                for lay in (man.get("layers") or []):
                    if not isinstance(lay, dict):
                        continue
                    if lay.get("mediaType") != "application/vnd.ollama.image.model":
                        continue
                    d = str(lay.get("digest") or "").split(":")[-1].split("-")[-1]
                    if len(d) == 64:
                        out[d] = name
    _manifests.update(data=out, next=now + 60.0)
    return out

def _name_ollama_segments(cards, ps):
    """Turn the generic "Ollama" segments into model names. Three tiers.

    Each named segment records which tier named it, in `name_src`:

      manifest  the runner's --model blob digest joined against the manifests
                on disk. Exact. Correct under SCHED_SPREAD and correct when two
                resident models are the same size.
      pid       one loaded model is one runner process, so segments sharing a
                pid are one model. Sum them ACROSS cards, then match that total
                against /api/ps. Exact grouping; only the size step is a guess.
      size      the original per-segment match, which is what `pid` reduces to
                when a model sits whole on one card.

    A segment that nothing can name keeps the label "Ollama". Leaving it
    generic is honest; labelling it with whatever was nearest in size is not.
    """
    segs = [s for c in cards for s in c["segments"]
            if s.get("kind") == "proc" and s.get("label") == "Ollama"]
    if not segs:
        return
    pool = list(ps or [])
    by_name = {}
    for m in pool:
        by_name.setdefault(m["name"], m)
        by_name.setdefault(m["name"].split(":")[0], m)

    # Tier 1 - the real join.
    blobs = read_model_blob_names()
    if any(s.get("blob") and s["blob"] not in blobs for s in segs):
        blobs = read_model_blob_names(force=True)
    for s in segs:
        nm = blobs.get(s.get("blob") or "")
        if not nm:
            continue
        s.update(label=nm, kind="model", name_src="manifest")
        m = by_name.get(nm) or by_name.get(nm.split(":")[0])
        if m:
            s["model"] = m               # carries size vs size_vram for the UI
            if m in pool:
                pool.remove(m)

    # Tier 2 - group by runner pid, sum across cards, then match on size.
    groups = {}
    for s in segs:
        if s.get("kind") != "model":
            groups.setdefault(s.get("pid"), []).append(s)
    for _pid, g in sorted(groups.items(), key=lambda kv: -sum(s["bytes"] for s in kv[1])):
        if not pool:
            break
        total = sum(s["bytes"] for s in g)
        best = min(pool, key=lambda m: abs(m["size_vram"] - total))
        tol = max(512 * 1024 * 1024, 0.15 * (best["size_vram"] or 0))
        if best["size_vram"] and abs(best["size_vram"] - total) <= tol:
            for s in g:
                s.update(label=best["name"], kind="model", model=best,
                         name_src=("pid" if len(g) > 1 else "size"))
            pool.remove(best)

    # Where the rest of it lives. A split model otherwise shows as the same name
    # printed twice with nothing saying the two halves are one thing. This gives
    # each half the other's share so the panel can say "12.3 GiB here, 11.8 GiB
    # on 87:00.0" instead.
    grouped = {}
    for c in cards:
        for s in c["segments"]:
            if s.get("kind") == "model" and s.get("pid"):
                grouped.setdefault((s["pid"], s["label"]), []).append((c, s))
    for _key, items in grouped.items():
        if len(items) < 2:
            continue
        tot = sum(s["bytes"] for _c, s in items)
        for c, s in items:
            s["split"] = {"total": tot, "cards": len(items),
                          "parts": [{"slot": c2.get("slot") or c2.get("pci"),
                                     "bytes": s2["bytes"], "self": c2 is c}
                                    for c2, s2 in items]}

def compose_gpu_mem(gpus, mem, clients, ps, ps_err):
    """One bar per card: the segments that make up `used`, plus what's free.

    Attribution precedence, and the UI names whichever one it used:
      fdinfo       a real process, a real card, a real byte count.
      pin          OLLAMA_GPU_PCI says which card Ollama is on.
      single-card  there is only one card, so there is nothing to guess.
      inferred     best-fit placement. A guess, labelled as one.

    Cards come from read_gpu_temps() first so the ordering matches every other
    GPU panel and every CSV column, then any card that reports memory but no
    temperatures is appended rather than dropped.
    """
    cards, by_pci = [], {}
    for g in gpus:
        pci = g.get("pci")
        c = {"pci": pci, "slot": g.get("slot") or pci, "card": g.get("card"),
             "junction": g.get("junction"), "mem_c": g.get("mem"), "segments": []}
        c.update({k: (mem.get(pci) or {}).get(k) for k in
                  ("vram_total", "vram_used", "vram_free", "gtt_total", "gtt_used",
                   "busy", "mem_busy", "power_w", "cap_w", "cap_max_w")})
        cards.append(c); by_pci[pci] = c
    for pci, m in sorted(mem.items()):
        if pci in by_pci:
            continue
        c = {"pci": pci, "slot": pci, "card": None, "junction": None, "mem_c": None,
             "segments": []}
        c.update(m)
        cards.append(c); by_pci[pci] = c

    # 1. fdinfo - every consumer, named by process.
    from_fdinfo = False
    pin_bad = False
    for pci, cl in (clients or {}).items():
        c = by_pci.get(pci)
        if not c:
            continue
        for x in cl:
            cmd = x.get("cmd") or ""
            # The blob digest is pulled from the FULL command line, before the
            # truncation below. `ollama runner --model <path>` puts it early,
            # but the old 200-char cut was close enough to the flags that a
            # longer runner path would have silently eaten the one field this
            # whole join depends on.
            mb = _RUNNER_BLOB_RE.search(cmd)
            c["segments"].append({"label": _consumer_label(x), "bytes": x["bytes"],
                                  "kind": "proc", "src": "fdinfo", "pid": x["pid"],
                                  "blob": (mb.group(1) if mb else None),
                                  "eng": x.get("eng"), "gpu_pct": x.get("gpu_pct"),
                                  "detail": cmd[:400]})
            from_fdinfo = True

    if from_fdinfo:
        _name_ollama_segments(cards, ps)

    elif ps:
        target = None
        if OLLAMA_PIN:
            target = next((c for c in cards if OLLAMA_PIN in (c["pci"], c["slot"])), None)
            # A pin that resolves to no card is a typo in the compose file, not
            # a licence to fall back to guessing. Placing the models somewhere
            # anyway would hide the typo behind a plausible-looking bar. Leave
            # them unplaced (they still show up in `other`) and say so.
            if target is None:
                pin_bad = True
        else:
            withvram = [c for c in cards if c.get("vram_total")]
            if len(withvram) == 1:
                target = withvram[0]
        if target is not None:
            src = "pin" if OLLAMA_PIN else "single-card"
            for m in ps:
                if m["size_vram"]:
                    target["segments"].append({"label": m["name"], "bytes": m["size_vram"],
                                               "kind": "model", "src": src, "model": m})
        elif not OLLAMA_PIN:
            # Best fit: biggest model first, onto the card with the least spare
            # allocated space that still accounts for it. Tightest-fit rather
            # than first-fit, because on two identical cards first-fit would
            # pile everything onto card 0 and be confidently wrong.
            room = {c["pci"]: (c.get("vram_used") or 0) for c in cards}
            for m in sorted(ps, key=lambda m: -m["size_vram"]):
                if not m["size_vram"]:
                    continue
                fit = [c for c in cards if room.get(c["pci"], 0) >= m["size_vram"] * 0.95]
                if not fit:
                    continue
                fit.sort(key=lambda c: room[c["pci"]])
                c = fit[0]
                room[c["pci"]] -= m["size_vram"]
                c["segments"].append({"label": m["name"], "bytes": m["size_vram"],
                                      "kind": "model", "src": "inferred", "model": m})

    for c in cards:
        c["segments"].sort(key=lambda s: -s["bytes"])
        named = sum(s["bytes"] for s in c["segments"])
        used = c.get("vram_used")
        c["named"] = named
        if used is not None:
            # The named segments must never exceed what the driver says is
            # allocated. If they do, the attribution is wrong, and the panel
            # says so rather than quietly rescaling into something plausible.
            c["other"] = max(0, used - named)
            c["over"]  = max(0, named - used)
        c["src"] = ("fdinfo" if from_fdinfo else
                    (c["segments"][0]["src"] if c["segments"] else None))
    # How the model names were arrived at, so the note under the bars can say
    # which of them are facts and which are best guesses.
    named_by = {}
    unnamed = 0
    eng_seen = eng_moved = False
    for c in cards:
        for s in c["segments"]:
            # Activity, from whichever of the two signals can speak. The model
            # one is attached here rather than in the naming pass because it
            # applies to every segment of a split model, on both cards.
            m = s.get("model")
            if m:
                s["active"] = m.get("active")
                s["used_ago"] = m.get("used_ago")
            if s.get("eng"):
                eng_seen = True
                if s.get("gpu_pct"):
                    eng_moved = True
            if s.get("kind") == "model":
                k = s.get("name_src") or s.get("src") or "?"
                named_by[k] = named_by.get(k, 0) + 1
            elif s.get("label") == "Ollama":
                unnamed += 1
    meta = {"fdinfo": from_fdinfo, "pin": OLLAMA_PIN or None,
            "pin_unmatched": pin_bad,
            "ollama_disabled": not OLLAMA_ENABLED,
            "ollama_error": ps_err, "ollama_models": len(ps or []),
            "ollama_url": OLLAMA_URL,
            "named_by": named_by, "unnamed_ollama": unnamed,
            "engine_time": eng_seen, "engine_moving": eng_moved,
            "models_dir": bool(OLLAMA_MODELS_DIR and
                               os.path.isdir(os.path.join(OLLAMA_MODELS_DIR, "manifests"))),
            "can_unload": ALLOW_MODEL_UNLOAD and OLLAMA_ENABLED}
    return cards, meta

# ---------------- fan control bridge ----------------
# The daemon writes fan-status.json every cycle; we read it. We write
# fan-curve.conf; the daemon re-reads and re-validates it every cycle. Nothing
# here opens /dev/ipmi0 for fan control.

PRESETS = ("silent","quiet","balanced","cool","max")
CURVE_RE = re.compile(r"^\s*(\d{1,3}:\d{1,3})(\s+\d{1,3}:\d{1,3})*\s*$")

def read_fan_status():
    try:
        with open(FAN_STATUS) as f:
            d = json.load(f)
    except Exception:
        return None
    # If the daemon has stopped, the file lingers with stale numbers. Anything
    # older than three poll intervals means nobody is driving the fans from
    # here - which the UI needs to say loudly, since manual mode may still be
    # latched on the BMC.
    try:
        age = time.time() - float(d.get("ts", 0))
        d["age"] = round(age, 1)
        d["stale"] = age > max(30, 3 * int(d.get("interval", 8)))
    except Exception:
        d["age"], d["stale"] = None, True
    return d

def validate_fan_update(body):
    """Returns (updates_dict, error_string). Mirrors the daemon's own limits."""
    up = {}
    def num(key, lo, hi):
        if key not in body or body[key] is None or body[key] == "":
            return None
        try: v = int(body[key])
        except Exception: return "%s must be a whole number" % key
        if not (lo <= v <= hi): return "%s must be between %d and %d" % (key, lo, hi)
        up[key.upper()] = v
        return None

    if "preset" in body and body["preset"]:
        p = str(body["preset"]).strip().lower()
        if p not in PRESETS:
            return None, "preset must be one of: " + ", ".join(PRESETS)
        up["PRESET"] = p
    # 20 is a hard lower bound, not a style choice: the whole random-spike bug
    # was a floor low enough for a fan to fall under the BMC's Lower Critical
    # threshold. The UI warns below 40; below 20 we simply refuse.
    for args in (("min_duty",20,100), ("hyst",0,25), ("failsafe_duty",0,100),
                 ("interval",2,120), ("cpu_every",1,30), ("diag",0,1),
                 # v6 daemon keys: per-card power cap in watts
                 # (0 = card's own maximum), and the cooperative kill switch.
                 # DISABLED is a conf flag rather than a process kill because
                 # killing the daemon fires its fail-loud atexit and pins the
                 # fans at Full — the opposite of what a kill switch is for.
                 ("cap_w",0,300), ("disabled",0,1)):
        err = num(*args)
        if err: return None, err
    for key in ("gpu_curve","cpu_curve"):
        if key in body and body[key] is not None:
            v = str(body[key]).strip()
            if v and not CURVE_RE.match(v):
                return None, "%s must be space-separated temp:duty pairs, e.g. 55:40 70:55" % key
            for pair in v.split():
                if int(pair.split(":")[1]) > 100:
                    return None, "%s has a duty above 100%%" % key
            up[key.upper()] = v
    if not up:
        return None, "nothing to change"
    return up, None

MANAGED_HDR = "# --- written by the V620 telemetry dashboard ---"
# The keys a preset supplies. Picking a preset alone resets these to it.
# CAP_W and DISABLED are cleared by a preset pick on purpose: choosing any
# preset both re-enables a disabled daemon and returns the power cap to the
# preset's own value. "Press a preset button" is the recovery gesture.
OVERRIDE_KEYS = ("MIN_DUTY","HYST","FAILSAFE_DUTY","GPU_CURVE","CPU_CURVE",
                 "CAP_W","DISABLED")

def write_fan_conf(updates, clear=()):
    """Rewrite the managed keys in fan-curve.conf, preserving every comment.

    An existing ACTIVE line for a key is replaced in place. Commented-out
    defaults (`# MIN_DUTY=40`) are deliberately left alone - they're
    documentation - and the new value is appended in a marked block at the end.
    The daemon's parser is last-wins, so appending is correct.

    `clear` comments out active lines for those keys instead. That's what makes
    a bare preset click actually mean something: the conf beats the preset in
    the daemon's precedence, so a MIN_DUTY=45 left over from an earlier edit
    would silently defeat `cool`'s 50% floor. Commenting it out hands the key
    back to the preset. Nothing is deleted, so the old value stays readable.

    Written atomically: the daemon reloads on mtime change and must never see
    half a file.
    """
    try:
        with open(FAN_CONF) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []
    remaining = dict(updates)
    clear = set(clear) - set(remaining)
    out, seen = [], set()
    for ln in lines:
        m = re.match(r"^\s*([A-Z_]+)\s*=", ln)
        key = m.group(1) if m else None
        if key and key in remaining:
            out.append("%s=%s" % (key, remaining.pop(key))); seen.add(key)
        elif key and key in clear:
            out.append("# " + ln)     # hand this key back to the preset
        elif key and key in seen:
            out.append("# " + ln)     # a later duplicate would win - neutralise it
        else:
            out.append(ln)
    if remaining:
        if MANAGED_HDR not in out:
            out += ["", MANAGED_HDR]
        for k, v in remaining.items():
            out.append("%s=%s" % (k, v))
    body = "\n".join(out).rstrip() + "\n"

    d = os.path.dirname(FAN_CONF) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".fan-curve.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(body)
        os.replace(tmp, FAN_CONF)
    except Exception:
        try: os.unlink(tmp)
        except Exception: pass
        raise

# ---------------- CSV schema ----------------
# Columns are fixed when logging starts, from whatever hardware is present
# then. Per-GPU and per-fan columns are added alongside the original max
# columns, so old logs still load and new ones can tell the cards apart.
BASE_COLS = ["timestamp","watts","cpu_util_pct","cpu1_c","cpu2_c",
             "gpu_edge_c","gpu_junction_c"]

def build_cols(gpu_ids, fan_ids):
    cols = list(BASE_COLS)
    for i, _ in enumerate(gpu_ids):
        cols.append("gpu%d_junction_c" % i)
    for i, _ in enumerate(gpu_ids):
        cols.append("gpu%d_edge_c" % i)
    # VRAM and board power, per card. Both are what you actually want beside
    # junction temperature during a thermal run: the question is never "how hot
    # did it get" on its own, it's "how hot did it get, doing what".
    for i, _ in enumerate(gpu_ids):
        cols.append("gpu%d_vram_used_mb" % i)
    for i, _ in enumerate(gpu_ids):
        cols.append("gpu%d_power_w" % i)
    cols += ["fan_rpm_max","fan_rpm_min"]
    for f in fan_ids:
        cols.append("%s_rpm" % f.lower())
    cols += ["fan_duty_pct"]
    # Appended at the very end so every column an older parser knows about
    # keeps its exact position. Blank outside a thermal run.
    cols.append("test_phase")
    return cols

def build_row(now, w, util, cpu, gpus, fans, fan, gpu_ids, fan_ids, phase=None):
    by_pci = {g.get("pci"): g for g in gpus}
    ge = max([g.get("edge") for g in gpus if g.get("edge") is not None], default=None)
    gj = hottest_gpu(gpus)
    row = [now.isoformat(timespec="seconds"), w,
           "" if util is None else util,
           cpu.get("CPU1",""), cpu.get("CPU2",""),
           "" if ge is None else ge, "" if gj is None else gj]
    for pci in gpu_ids:
        g = by_pci.get(pci) or {}
        row.append(g.get("junction", ""))
    for pci in gpu_ids:
        g = by_pci.get(pci) or {}
        row.append(g.get("edge", ""))
    for pci in gpu_ids:
        g = by_pci.get(pci) or {}
        v = g.get("vram_used")
        row.append("" if v is None else round(v / (1024.0 * 1024.0)))
    for pci in gpu_ids:
        g = by_pci.get(pci) or {}
        row.append(g.get("power_w", ""))
    vals = [v for v in fans.values() if v is not None]
    row += [max(vals) if vals else "", min(vals) if vals else ""]
    for f in fan_ids:
        row.append(fans.get(f, ""))
    row.append(fan.get("duty", "") if fan and not fan.get("stale") else "")
    row.append(phase or "")
    return row

# ---------------- poller ----------------
# Logging is started and stopped from two places now - the button, and the
# thermal test - so the mechanics live here rather than inside a route.
def _start_log():
    os.makedirs(DATA_DIR, exist_ok=True)
    fn = os.path.join(DATA_DIR, "telemetry-%s.csv" %
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with lock:
        gpu_ids = list(state["seen_gpu_ids"])
        fan_ids = list(state["seen_fan_ids"])
        cols = build_cols(gpu_ids, fan_ids)
        state["gpu_ids"], state["fan_ids"], state["cols"] = gpu_ids, fan_ids, cols
    with open(fn, "w", newline="") as f:
        csv.writer(f).writerow(cols)
    with lock:
        state["logging"] = True
        state["logfile"] = fn
        state["stats"] = {"min": None, "max": None, "sum": 0, "n": 0,
                          "start": time.time(), "umax": None, "c1max": None,
                          "c2max": None, "gjmax": None, "fmax": None,
                          "fmin": None, "vmax": None, "gpwmax": None}
    return fn

def _stop_log():
    with lock:
        state["logging"] = False

# ---------------- thermal test ----------------
# A supervised thermal run: apply a known load, log everything, and stop the
# moment the machine looks unhappy.
#
# READ THIS BEFORE TRUSTING A GREEN RESULT. What this can and cannot do:
#
#   It CAN generate load through Ollama's HTTP API, because that is a service
#   this container can already reach. That is real, representative load - it is
#   what the machine actually does for a living, not a synthetic burn.
#
#   It CANNOT choose which card the load lands on. Ollama picks its GPU at
#   process start from HIP_VISIBLE_DEVICES; there is no per-request control and
#   no API that reports the choice. So this is not a "card 0 then card 1" test.
#   It applies load and REPORTS which cards actually got hot. For a single-card
#   run, start Ollama pinned to one card and run the test again.
#
#   It CANNOT stop a load it did not start. Observe-only mode (for a ComfyUI
#   render, say) still runs the full watchdog and still maxes the fans, but
#   ending the job is on you. The verdict says so rather than implying control.
#
#   It CANNOT power-cap the cards. gpu-fan-control owns the card the way it
#   owns the fan zones. The one protective action available here is raising the
#   fan preset, which it does by writing the same config file the fan panel
#   writes - never by touching /dev/ipmi0.
#
#   An Ollama inference load is NOT the worst case this machine can produce.
#   Sustained diffusion (ComfyUI/SDXL) is typically hotter. A pass here means
#   "survived the load we can actually generate", which is worth knowing and is
#   not the same claim as "thermally validated".

TEST_LIMITS = {           # key: (default, lo, hi)
    "baseline_s":  (60,   10,   600),
    "max_load_s":  (900,  60,   7200),
    "hold_s":      (300,  0,    3600),
    "cooldown_s":  (300,  30,   3600),
    "abort_c":     (95,   70,   99),     # never above the driver's own emergency
    "abort_watts": (1000, 300,  1600),
    "workers":     (2,    1,    8),
    "num_predict": (256,  32,   2048),
}
STEADY_BAND_C = 1.0       # junction spread that counts as thermally steady
STEADY_WINDOW_S = 90

test = {
    "phase": "idle",      # idle|baseline|load|hold|cooldown|done
    "started": None, "phase_started": None, "ended": None,
    "cfg": None, "mode": None, "abort": None, "verdict": None,
    "steady_at": None, "peak": {}, "notes": [],
    "logfile": None, "fan_restore": None, "load_errors": 0, "load_reqs": 0,
}
_test_stop = threading.Event()

def _tnote(msg):
    """Timestamped line for the run's own narrative. Kept short and factual."""
    test["notes"].append({"t": round(time.time()), "msg": msg})
    del test["notes"][:-60]

def validate_test_config(body):
    """Returns (cfg, error). Mirrors validate_fan_update's shape deliberately."""
    cfg = {}
    for key, (dflt, lo, hi) in TEST_LIMITS.items():
        v = body.get(key, dflt)
        if v in (None, ""):
            v = dflt
        try:
            v = int(v)
        except Exception:
            return None, "%s must be a whole number" % key
        if not (lo <= v <= hi):
            return None, "%s must be between %d and %d" % (key, lo, hi)
        cfg[key] = v
    mode = (body.get("mode") or "ollama").strip()
    if mode not in ("ollama", "observe"):
        return None, "mode must be 'ollama' or 'observe'"
    cfg["mode"] = mode
    cfg["model"] = (body.get("model") or "").strip()
    if mode == "ollama" and not cfg["model"]:
        return None, "pick a model to load the GPUs with"
    # Default ON. It is the only protective action available from here, and a
    # thermal test with the safety off by default is a trap.
    cfg["protect"] = bool(body.get("protect", True))
    return cfg, None

def ollama_tags():
    """Model names available to pull load from. Empty list if Ollama is down."""
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/tags",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3.0) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        return sorted(m.get("name") for m in (d.get("models") or []) if m.get("name"))
    except Exception:
        return []

_LOAD_PROMPT = ("Write a long, detailed technical explanation of how PCIe "
                "lane allocation works on a dual-socket server board. Do not "
                "stop early; keep going with more detail.")

def _load_worker(cfg):
    """One generation loop. Bounded requests so the stop flag is checked often.

    num_predict caps each request at a few seconds of work rather than letting
    one call run for minutes, because a load thread that cannot be interrupted
    promptly is the difference between an abort and a dead card.
    """
    body = json.dumps({
        "model": cfg["model"], "prompt": _LOAD_PROMPT, "stream": False,
        "keep_alive": "10m", "options": {"num_predict": cfg["num_predict"]},
    }).encode()
    while not _test_stop.is_set():
        try:
            req = urllib.request.Request(
                OLLAMA_URL + "/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
            with lock:
                test["load_reqs"] += 1
        except Exception as e:
            with lock:
                test["load_errors"] += 1
                if test["load_errors"] in (1, 10, 50):
                    _tnote("load request failed: %s" % str(e)[:120])
            # Back off rather than hammering a service that is refusing.
            if _test_stop.wait(2.0):
                return

def _max_fans():
    """Raise cooling the only way this app is allowed to: via the daemon's conf.

    Returns the preset that was in effect, so it can be handed back afterwards.
    """
    if not os.path.isdir(FAN_DIR):
        _tnote("cannot raise fans: %s is not mounted, so this run has no "
               "protective action available at all" % FAN_DIR)
        return None
    prev = None
    try:
        st = read_fan_status() or {}
        prev = st.get("preset")
        # Clear the explicit overrides too. A leftover MIN_DUTY=45 in the conf
        # outranks the preset in the daemon's precedence and would quietly
        # defeat max exactly when it matters most.
        write_fan_conf({"PRESET": "max"}, OVERRIDE_KEYS)
        _tnote("fans forced to max (was %s)" % (prev or "unknown"))
    except Exception as e:
        _tnote("could not raise fans: %s" % str(e)[:120])
        return None
    return prev

def _restore_fans():
    prev = test.get("fan_restore")
    if not prev:
        return
    try:
        write_fan_conf({"PRESET": prev}, OVERRIDE_KEYS)
        _tnote("fan preset restored to %s" % prev)
    except Exception as e:
        _tnote("could not restore fan preset: %s" % str(e)[:120])
    test["fan_restore"] = None

def test_watchdog(gpus, watts, fan, now):
    """Called from the poller every cycle. Returns an abort reason, or None.

    This lives in the poller rather than in the load threads on purpose: in
    observe-only mode there are no load threads, and a watchdog that only runs
    while we happen to be driving the load is not a watchdog.
    """
    cfg = test.get("cfg") or {}
    abort_c = cfg.get("abort_c", 95)

    # 1. A card we have already seen has stopped reporting. This is the exact
    #    signature of the failure that started this whole exercise: at 100 C a
    #    card fell off the PCIe bus and vanished from hwmon. It is NOT a
    #    throttle and it does not come back on its own, so it outranks every
    #    other check here.
    live = {g["pci"] for g in gpus}
    missing = [p for p in state["seen_gpu_ids"] if p not in live]
    if missing:
        return "GPU %s stopped reporting - that is a bus drop, not a throttle. " \
               "It will need a cold power cycle, not a reboot." % ", ".join(missing)

    # 2. Junction over the line.
    hot = [(g.get("junction"), g["pci"]) for g in gpus if g.get("junction") is not None]
    for j, pci in hot:
        if j >= abort_c:
            return "GPU %s junction hit %.0f C (limit %d)" % (pci, j, abort_c)

    # 3. Whole-machine power. The PSU budget is finite and shared with the CPUs.
    #    Read the limit ONCE. An earlier version compared against .get(...) but
    #    formatted the message with cfg["abort_watts"], so a watchdog running on
    #    a partial config crashed at exactly the moment it was supposed to stop
    #    the run — the worst possible failure mode for a safety check.
    abort_w = cfg.get("abort_watts", 1000)
    if watts is not None and watts >= abort_w:
        return "system power hit %.0f W (limit %d)" % (watts, abort_w)

    # 4. The fan daemon died mid-run. Carrying on would mean heating two cards
    #    with nothing managing cooling, possibly with the BMC latched at
    #    whatever duty was last commanded.
    if fan and fan.get("stale"):
        return "gpu-fan-control stopped reporting (%ss stale) - nothing is " \
               "managing the fans" % (fan.get("age"))
    return None

def _test_peak(gpus, watts, util):
    """Per-card and whole-machine peaks, kept per phase so cooldown is readable."""
    p = test["peak"]
    def up(k, v):
        if v is not None and (p.get(k) is None or v > p[k]):
            p[k] = v
    up("watts", watts)
    up("util", util)
    for g in gpus:
        pci = g.get("pci")
        up("j:" + pci, g.get("junction"))
        up("e:" + pci, g.get("edge"))
        up("w:" + pci, g.get("power_w"))
        up("v:" + pci, round((g.get("vram_used") or 0) / (1024.0 ** 3), 1) or None)

def _steady(hist):
    """True once junction has been within STEADY_BAND_C for the whole window."""
    if not hist:
        return False
    span = hist[-1][0] - hist[0][0]
    if span < STEADY_WINDOW_S:
        return False
    vals = [v for _, v in hist]
    return (max(vals) - min(vals)) <= STEADY_BAND_C

_jhist = []

def test_tick(gpus, watts, util, fan, now):
    """Advance the state machine. Called from the poller, already under lock."""
    ph = test["phase"]
    if ph in ("idle", "done"):
        return
    cfg = test["cfg"]
    _test_peak(gpus, watts, util)

    reason = test_watchdog(gpus, watts, fan, now)
    if reason and ph != "cooldown":
        _abort_test(reason, now)
        return

    el = now - (test["phase_started"] or now)
    if ph == "baseline":
        if el >= cfg["baseline_s"]:
            _enter("load", now)
            _start_load(cfg)
    elif ph == "load":
        j = hottest_gpu(gpus)
        if j is not None:
            _jhist.append((now, j))
            del _jhist[:-200]
            while _jhist and now - _jhist[0][0] > STEADY_WINDOW_S * 1.5:
                _jhist.pop(0)
        if _steady(_jhist):
            test["steady_at"] = round(now)
            _tnote("thermally steady at %.0f C after %ds of load" % (j, el))
            _enter("hold", now)
        elif el >= cfg["max_load_s"]:
            _tnote("load ran the full %ds without settling" % cfg["max_load_s"])
            _enter("hold", now)
    elif ph == "hold":
        if el >= cfg["hold_s"]:
            _stop_load()
            _enter("cooldown", now)
    elif ph == "cooldown":
        if el >= cfg["cooldown_s"]:
            _finish_test(now)

def _enter(phase, now):
    test["phase"] = phase
    test["phase_started"] = now
    _tnote("phase: %s" % phase)

def _start_load(cfg):
    if cfg["mode"] != "ollama":
        _tnote("observe-only: not generating any load, only watching")
        return
    _test_stop.clear()
    for _ in range(cfg["workers"]):
        threading.Thread(target=_load_worker, args=(cfg,), daemon=True).start()
    _tnote("%d load worker(s) started against %s" % (cfg["workers"], cfg["model"]))

def _stop_load():
    _test_stop.set()

def _abort_test(reason, now=None):
    # `now` is the timestamp of the sample that triggered the abort, threaded
    # through from the poller rather than re-read here. Reading the clock again
    # would put the phase on a slightly different timebase from the samples it
    # is measured against, which is wrong in production by milliseconds and
    # wrong in the tests by twenty years.
    now = time.time() if now is None else now
    _stop_load()
    test["abort"] = reason
    _tnote("ABORT: " + reason)
    if (test["cfg"] or {}).get("protect") and not test["fan_restore"]:
        test["fan_restore"] = _max_fans() or "balanced"
    # Deliberately go to cooldown rather than stopping dead: the recovery curve
    # after an abort is the most valuable part of the log, and killing the run
    # here would throw it away.
    _enter("cooldown", now)

def _finish_test(now=None):
    _stop_load()
    _restore_fans()
    test["phase"] = "done"
    test["ended"] = time.time() if now is None else now
    test["verdict"] = _verdict()
    if test.get("logfile"):
        state["logging"] = False
    _tnote("run finished")

def _verdict():
    """Plain-language result. Says what was actually shown, not more."""
    p, cfg = test["peak"], test["cfg"] or {}
    js = sorted(((k[2:], v) for k, v in p.items() if k.startswith("j:")),
                key=lambda kv: -kv[1])
    hottest = js[0] if js else None
    if test["abort"]:
        return {"ok": False, "headline": "Aborted", "detail": test["abort"]}
    if not hottest:
        return {"ok": False, "headline": "No result",
                "detail": "No GPU reported a junction temperature during the run."}
    pci, j = hottest
    margin = cfg.get("abort_c", 95) - j
    if cfg.get("mode") == "observe":
        lead = "Observed run completed."
    else:
        lead = "Completed %d generation requests." % test["load_reqs"]
    detail = ("%s Hottest card was %s at %.0f C junction, %.0f C below the %d C "
              "abort line. Peak system power %s W." %
              (lead, pci, j, margin, cfg.get("abort_c", 95),
               p.get("watts") if p.get("watts") is not None else "?"))
    if not test.get("steady_at"):
        detail += (" It never settled to a steady temperature, so this is a "
                   "floor on how hot it gets, not a ceiling - run it longer.")
    if margin < 5:
        return {"ok": False, "headline": "Passed, but only just", "detail": detail}
    return {"ok": True, "headline": "Passed", "detail": detail}

def start_test(cfg):
    """Returns (ok, error). Caller holds no lock."""
    with lock:
        if test["phase"] not in ("idle", "done"):
            return False, "a test is already running"
    if cfg["mode"] == "ollama":
        tags = ollama_tags()
        if not tags:
            return False, ("Ollama at %s isn't answering, so there is nothing to "
                           "drive the load with. Use observe-only mode if you are "
                           "running the load yourself." % OLLAMA_URL)
        if cfg["model"] not in tags:
            return False, "Ollama has no model called %r" % cfg["model"]
    # Refuse a run whose watchdog would abort it on the first sample. A stale
    # fan daemon is an abort condition, so starting anyway produces thirty
    # seconds of cooldown and a verdict that reads "Aborted" for a reason that
    # was already true before the button was pressed. Say it now, when it is
    # still a fixable sentence rather than a failed run.
    fan0 = read_fan_status()
    if fan0 and fan0.get("stale"):
        return False, ("gpu-fan-control last reported %ss ago, so nothing is "
                       "managing the fans and the run would abort on its first "
                       "sample. Start the daemon, or unmount /fanctl to run "
                       "without fan supervision." % fan0.get("age"))
    fn = _start_log()          # every run gets its own CSV, always
    del _jhist[:]
    with lock:
        test.update({"phase": "baseline", "started": time.time(),
                     "phase_started": time.time(), "ended": None,
                     "cfg": cfg, "mode": cfg["mode"], "abort": None,
                     "verdict": None, "steady_at": None, "peak": {}, "notes": [],
                     "logfile": os.path.basename(fn), "fan_restore": None,
                     "load_errors": 0, "load_reqs": 0})
        _tnote("run started in %s mode, logging to %s" % (cfg["mode"], test["logfile"]))
        _tnote("phase: baseline")
    return True, None

def stop_test(reason="stopped by hand"):
    with lock:
        if test["phase"] in ("idle", "done"):
            return
        _stop_load()
        _restore_fans()
        _tnote(reason)
        test["phase"] = "done"
        test["ended"] = time.time()
        test["verdict"] = _verdict() if test["peak"] else {
            "ok": False, "headline": "Stopped", "detail": reason}
        state["logging"] = False

def test_public():
    """The slice of test state the browser needs. Cheap; called every 2s."""
    cfg = test["cfg"] or {}
    now = time.time()
    return {
        "phase": test["phase"], "mode": test["mode"], "abort": test["abort"],
        "verdict": test["verdict"], "notes": test["notes"][-12:],
        "logfile": test["logfile"], "steady": bool(test["steady_at"]),
        "elapsed": int(now - test["started"]) if test["started"] else None,
        "phase_elapsed": int(now - test["phase_started"]) if test["phase_started"] else None,
        "phase_total": (cfg.get({"baseline": "baseline_s", "load": "max_load_s",
                                 "hold": "hold_s", "cooldown": "cooldown_s"}
                                .get(test["phase"], ""), None)),
        "peak": test["peak"], "cfg": cfg,
        "reqs": test["load_reqs"], "errors": test["load_errors"],
        "protect": bool(cfg.get("protect")),
    }

def poller():
    read_util = make_util_reader()
    read_util()  # prime
    while True:
        w, err = read_watts()
        cpu = read_temps()
        util = read_util()
        gpus = read_gpu_temps()
        # Memory is merged INTO the temperature dicts rather than carried
        # alongside them, so every consumer downstream - the CSV row builder,
        # /api/power, the existing GPU panel - gets the new fields for free and
        # nothing has to learn a second shape.
        gmem = read_gpu_mem()
        for g in gpus:
            g.update(gmem.get(g.get("pci")) or {})
        ps, ps_err = read_ollama_ps()
        clients = read_drm_clients_cached()
        # Pure function, no I/O: deliberately outside the lock.
        gpumem, gpumeta = compose_gpu_mem(gpus, gmem, clients, ps, ps_err)
        ram = read_ram()
        fans = read_fans()
        fan  = read_fan_status()
        now = time.time()
        with lock:
            state["error"] = err
            state["cpu"], state["util"], state["gpus"] = cpu, util, gpus
            state["fans"], state["fan"] = fans, fan
            state["gpumem"], state["gpumem_meta"] = gpumem, gpumeta
            state["ram"] = ram
            # Advance the thermal test before anything else touches the log, so
            # the row written below carries the phase this sample belongs to.
            # The watchdog lives here rather than in the load threads because
            # observe-only runs have no load threads, and a watchdog that only
            # runs while we happen to be driving the load is not a watchdog.
            try:
                test_tick(gpus, w, util, fan, now)
            except Exception as e:
                _tnote("watchdog error: %s" % str(e)[:120])
            for pci in (g["pci"] for g in gpus):
                if pci not in state["seen_gpu_ids"]:
                    state["seen_gpu_ids"] = sorted(state["seen_gpu_ids"] + [pci])
            for f in fans:
                if f not in state["seen_fan_ids"]:
                    state["seen_fan_ids"] = sorted(state["seen_fan_ids"] + [f])
            if w is not None:
                state["current"] = w
                state["ts"] = now
                state["samples"].append([round(now), w])
                if len(state["samples"]) > MAXPOINTS:
                    state["samples"] = state["samples"][-MAXPOINTS:]
            # The row is written whenever logging is on, NOT only when the power
            # reading came back. An earlier version nested the whole block under
            # `if w is not None`, so a single failed `ipmitool dcmi power
            # reading` threw away that sample's GPU temperatures as well - and
            # during a thermal run the temperatures are the entire point. A
            # missing wattage now leaves one blank cell instead of a hole in the
            # capture.
            s = state["stats"]
            if state["logging"] and s is not None:
                if w is not None:
                    s["min"] = w if s["min"] is None else min(s["min"], w)
                    s["max"] = w if s["max"] is None else max(s["max"], w)
                    s["sum"] += w; s["n"] += 1
                def upd(k, v):
                    if v is not None:
                        s[k] = v if s.get(k) is None else max(s[k], v)
                def updmin(k, v):
                    if v is not None:
                        s[k] = v if s.get(k) is None else min(s[k], v)
                upd("umax", util)
                upd("c1max", cpu.get("CPU1")); upd("c2max", cpu.get("CPU2"))
                upd("gjmax", hottest_gpu(gpus))
                vr = [g.get("vram_used") for g in gpus if g.get("vram_used")]
                upd("vmax", round(max(vr) / (1024.0 ** 3), 1) if vr else None)
                gw = [g.get("power_w") for g in gpus if g.get("power_w") is not None]
                upd("gpwmax", max(gw) if gw else None)
                vals = [v for v in fans.values() if v is not None]
                upd("fmax", max(vals) if vals else None)
                updmin("fmin", min(vals) if vals else None)
                try:
                    with open(state["logfile"], "a", newline="") as f:
                        csv.writer(f).writerow(build_row(
                            datetime.datetime.now().astimezone(),
                            "" if w is None else w,
                            util, cpu, gpus, fans, fan,
                            state["gpu_ids"], state["fan_ids"],
                            test["phase"] if test["phase"] not in ("idle", "done") else ""))
                except Exception as e:
                    state["error"] = f"log write: {e}"
        time.sleep(POLL)

# The poller is the only background thread this app has, and it starts on
# import. A test harness that drives the thermal state machine by hand needs to
# keep it out of the way: left running, it calls test_tick() a second time with
# real wall-clock timestamps against phases the harness started at a simulated
# t=1000, which finishes runs early and makes results flicker between runs.
# NO_POLLER=1 imports the module inert. It is never set in the container.
if os.environ.get("NO_POLLER", "").lower() not in ("1", "true", "yes"):
    threading.Thread(target=poller, daemon=True).start()

# ---------------- Model catalog, fit, and pulls ----------------
# Three things live in this section, and they exist because of one bad night:
# three concurrent Ollama pulls totalling 91.5 GiB were started from a browser,
# the TrueNAS web shell signed itself out when the tab lost focus, and all three
# downloads died somewhere between 11% and 31% with no way to see it and no way
# to resume from the UI that started them.
#
#   1. A REGISTRY CLIENT that asks Ollama's registry for the real byte count of
#      a tag before anything is downloaded, so "will this fit" is answered in
#      one HTTP call instead of after 11 GiB.
#   2. A FIT CALCULATOR that measures the cards actually in this machine (via
#      read_gpu_mem, the same amdgpu counters the VRAM panel uses) and judges a
#      size against them. It never assumes a card size.
#   3. A PULL MANAGER that owns each download in a thread INSIDE THIS CONTAINER.
#
# On (3), the thing that matters: Ollama cancels a pull when the HTTP client
# that asked for it goes away. That is why a pull started from a browser tab
# dies with the browser tab. Here the client is this app's background thread, so
# closing the browser, losing the LAN, or getting signed out of TrueNAS does
# nothing at all to the download. The honest limit is the other end of the same
# fact: if THIS container restarts, its threads die and the pulls stop too. They
# resume from the partial on the next attempt, but they do stop. Do not redeploy
# this project mid-pull, and remember watchtower can recreate a container
# without asking you first.

# Where Ollama keeps its blobs, if it is mounted. Optional in exactly the way
# FAN_DIR is optional: absent, the orphan/free-space panel says so and every
# other part of this section still works. It has to be read-write only because
# deleting a dead partial is one of the two things it is for.
OLLAMA_MODELS_DIR = os.environ.get("OLLAMA_MODELS_DIR", "/ollama-models")
# Model deletion is off unless you turn it on. Pulling is additive and a mistake
# costs disk; deleting is not and a mistake costs a re-download of 40 GB over a
# home connection. Different risk, different default.
ALLOW_MODEL_DELETE = os.environ.get("ALLOW_MODEL_DELETE", "0").lower() in ("1", "true", "yes")
# Unloading is a different animal and defaults ON. It frees VRAM and nothing
# else: the weights stay on disk and the next request that needs the model
# loads it again. The worst case is one slow reload, which is the same cost the
# keep-alive timer imposes anyway when it expires on its own.
ALLOW_MODEL_UNLOAD = os.environ.get("ALLOW_MODEL_UNLOAD", "1").lower() in ("1", "true", "yes")

REGISTRY = os.environ.get("OLLAMA_REGISTRY", "https://registry.ollama.ai/v2").rstrip("/")

GIB = 1073741824.0

# --- the KV cache estimate -------------------------------------------------
# This is measured, not guessed, but it is measured from ONE model on ONE
# machine and the UI says so everywhere it appears.
#
# The measurement: qwen3.6:35b-a3b at Q4_K_M is 20.6 GiB of weights. With
# OLLAMA_CONTEXT_LENGTH=32768, OLLAMA_FLASH_ATTENTION=1 and
# OLLAMA_KV_CACHE_TYPE=q8_0, the two cards reported 11.2 + 10.7 = 21.9 GiB
# resident. Everything that is not weights is therefore about 1.3 GiB: KV cache
# plus compute buffers, at 32K, on this configuration.
#
# Scaling it to other models is where the honesty runs out. KV size is
# ctx x layers x 2 x kv_heads x head_dim x bytes_per_element, and the registry
# manifest does not tell us layers or kv_heads. Context scales it linearly and
# that part is safe. For model size we use a square root, because attention
# geometry grows far more slowly than total parameters - especially for a MoE,
# where most of the parameter count is in experts that have no KV at all. A
# linear scale would badly over-reserve for a 235B MoE and a flat constant would
# under-reserve for a 70B dense model; the square root is a hedge between two
# wrong answers, and it is labelled as an estimate in the UI rather than dressed
# up as a measurement. If you want a different number, change KV_ANCHOR_GIB or
# type your own reserve into the Models tab - the field is there for exactly
# this reason.
KV_ANCHOR_GIB     = 1.3
KV_ANCHOR_CTX     = 32768
KV_ANCHOR_WEIGHTS = 20.6

def kv_estimate_gib(weights_gib, ctx, kv_bytes=1):
    """GiB of KV cache + compute buffers, estimated. See the note above."""
    try:
        w = max(1.0, float(weights_gib or 1.0))
        c = max(1, int(ctx or KV_ANCHOR_CTX))
    except Exception:
        return KV_ANCHOR_GIB
    scale = (w / KV_ANCHOR_WEIGHTS) ** 0.5
    # kv_bytes: 1 for q8_0 (what this box runs), 2 for fp16.
    return KV_ANCHOR_GIB * (c / float(KV_ANCHOR_CTX)) * scale * (kv_bytes or 1)

# --- formats that cannot run here ------------------------------------------
# The point of this table is to fail BEFORE the download, not after it. Every
# entry is a property of gfx1030 (or of not being an NVIDIA/Apple part), not a
# guess about size.
def tag_warning(tag):
    """(severity, reason) for a tag name. severity: 'bad' | 'warn' | ''."""
    t = (tag or "").lower()
    if "mlx" in t:
        return ("bad", "MLX is Apple Silicon only. It will not load on ROCm at all.")
    if "nvfp4" in t:
        return ("bad", "NVFP4 needs NVIDIA Blackwell tensor cores. Not gfx1030.")
    if "mxfp8" in t or re.search(r"(^|[-_])fp8", t):
        return ("bad", "FP8 is broken on gfx1030 — see ComfyUI issue #10388. "
                       "It downloads fine and then fails or produces garbage.")
    if "mtp" in t:
        return ("warn", "Multi-token prediction. Runtime support in this Ollama "
                        "build is unverified — it may simply be ignored, which "
                        "costs you nothing but the extra bytes.")
    if "bf16" in t or re.search(r"(^|[-_])f16", t):
        return ("warn", "Unquantised weights. Correct on this hardware, but "
                        "roughly 4x the size of a Q4 for very little quality.")
    return ("", "")

# --- registry --------------------------------------------------------------
# Ollama's registry speaks the standard OCI manifest API and needs no auth.
# Summing layers[].size gives the on-disk byte count, which is also very close
# to the VRAM the weights occupy once loaded.
#
# Cached for an hour because tag lists and manifests are immutable in practice
# and the Models tab would otherwise fire 30 requests every time you open it.
# Negative results are cached too, for much less time, so a model name that
# does not exist does not become a retry storm.
_reg_cache = {}
_reg_lock  = threading.Lock()
REG_TTL    = 3600.0
REG_TTL_MISS = 120.0

def _reg_get(path, timeout=6.0):
    now = time.time()
    with _reg_lock:
        hit = _reg_cache.get(path)
        if hit and now < hit[0]:
            return hit[1]
    val = None
    try:
        req = urllib.request.Request(REGISTRY + path, headers={
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, "
                      "application/vnd.oci.image.manifest.v1+json, application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            val = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        val = None
    with _reg_lock:
        _reg_cache[path] = (now + (REG_TTL if val is not None else REG_TTL_MISS), val)
    return val

def _repo_path(model):
    """library/<name> for official models, <ns>/<name> for everyone else."""
    m = (model or "").split(":")[0]
    return m if "/" in m else "library/" + m

def registry_tags(model):
    d = _reg_get("/%s/tags/list" % _repo_path(model))
    return sorted((d or {}).get("tags") or [])

def registry_size(model, tag):
    """Total bytes for a tag, or None. None means 'ask the catalog instead'."""
    d = _reg_get("/%s/manifests/%s" % (_repo_path(model), tag))
    if not d:
        return None
    try:
        layers = d.get("layers") or []
        total = sum(int(l.get("size") or 0) for l in layers)
        return total or None
    except Exception:
        return None

def registry_weight_bytes(model, tag):
    """Just the weights layer, which is what actually lands in VRAM.

    The manifest also carries the template, the licence text and the params
    blob. Those are kilobytes against tens of gigabytes so the distinction
    barely moves the number, but the fit question is about weights and it costs
    nothing to answer the question that was asked.
    """
    d = _reg_get("/%s/manifests/%s" % (_repo_path(model), tag))
    if not d:
        return None
    try:
        for l in (d.get("layers") or []):
            if "image.model" in (l.get("mediaType") or ""):
                return int(l.get("size") or 0) or None
    except Exception:
        pass
    return None

# --- the ollama.com library index (self-discovering) ------------------------
# This section replaces the HAND-WRITTEN repo list that used to be the only
# thing on the Models tab that wasn't live. Everything else there was already
# discovered: tag names from registry_tags(), byte sizes from the manifest
# fetch in _prefetch_sizes(), cards from amdgpu, installed models from Ollama.
# Only the NAMES were typed by hand, which is why gemma4 and laguna were
# missing from a browser that was otherwise reading the real registry.
#
# It is inlined here rather than imported because the compose binds this file
# ALONE into the container:
#     /srv/v620-dash/app.py => /app/app.py
# a single-file bind, not a directory. A second .py cannot ship beside it, and
# the deploy ritual stays what it has always been: copy one file, restart.
#
# DESIGN RULES -- do not relax these:
#   1. NO official JSON API exists. ollama/ollama#9142 is an open feature
#      request for /api/search. Scraping the index page is the only route.
#   2. The index is ONE PAGE with no pagination -- 234 entries on 2026-07-30.
#      One request gets the whole library.
#   3. Tags and byte sizes are NOT fetched here. 234 repos x ~15 tags x one
#      manifest each is thousands of requests. Those stay LAZY, on click,
#      handled by the registry code above that already does exactly that.
#   4. A BAD PARSE MUST NEVER REPLACE A GOOD CACHE. If ollama.com restyles the
#      page the regexes go quiet and return 3 entries instead of 234; writing
#      that over catalog.json would silently gut the browser. lib_write()
#      refuses, keeps the old file, and the UI shows the stale timestamp.
#   5. Stdlib only. This container has python3 and no curl, let alone bs4.
#
# MARKUP CONTRACT (verified against the live page 2026-07-30, 234/234 entries)
#     <a href="/library/NAME" class="group w-full space-y-5">
#       <div title="NAME"> <h2>...
#       <p class="max-w-lg break-words text-neutral-800 text-md">DESCRIPTION</p>
#       <span class="inline-flex ... text-indigo-600">vision</span>  <- capability
#       <span class="inline-flex ... bg-[#ddf4ff]">27b</span>        <- size
#       </svg><span >117.9M</span><span class="hidden sm:flex">&nbsp;Pulls</span>
#       </svg><span >93</span><span class="hidden sm:flex">&nbsp;Tags</span>
#       <span class="flex items-center" title="Nov 30, 2024 10:34 PM UTC">
#         ...<span >1 year ago</span>
#
#   NOTE the counts live inside a bare `<span >`, NOT as a text node after the
#   </svg>. An earlier build assumed the text-node form and returned None for
#   pulls and tag_count on all 234 entries while every other field parsed
#   perfectly -- a parser can be 83% right and still useless. Both forms are
#   accepted now. Entries are delimited by the /library/ anchor because the
#   href is semantic and the Tailwind class soup is not. Badges are classified
#   by BOTH colour and text shape and anything matching neither is REPORTED
#   rather than dropped, so a new badge kind shows up in the sync report
#   instead of vanishing.
import html as _htmlmod

LIB_UA        = "v620-dash/1.0"
LIB_URL       = os.environ.get("OLLAMA_LIBRARY_URL", "https://ollama.com/library")
LIB_TIMEOUT   = 30
LIB_PATH      = os.path.join(DATA_DIR, "catalog.json")
# Minimum entries a parse must yield before it may overwrite a cache that
# already holds more than this. 234 today; 60 is low enough to tolerate ollama
# pruning the library, high enough to catch a dead parse.
LIB_MIN_PLAUSIBLE = 60
# Automatic refresh is OFF by default. The button is always there; a box that
# reaches out to the internet on a schedule should be something you turned on.
LIB_AUTO_HOURS = float(os.environ.get("OLLAMA_LIBRARY_REFRESH_HOURS", "0") or 0)

LIB_REQUIRED = ("description", "pulls", "tag_count", "updated")
# Seen on the live index 2026-07-30: tools=89 thinking=39 vision=35 cloud=19
# embedding=12 audio=2, zero unknowns.
LIB_KNOWN_CAPS = {"vision", "tools", "thinking", "embedding", "audio", "cloud", "code"}

# Badge shapes: 1b, 1.5b, 270m, 8x7b, 671b, e4b (gemma4 effective), a3b
# (qwen3 activated), 16x17b.
LIB_SIZE_RE = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)?[kmbt]|[0-9]+x[0-9]+(?:\.[0-9]+)?b|[ea][0-9]+(?:\.[0-9]+)?b)$",
    re.I)
LIB_ANCHOR_RE = re.compile(r'<a\s+href="/library/([A-Za-z0-9][A-Za-z0-9._\-]*)"', re.I)
LIB_TITLE_RE  = re.compile(r'<div\s+title="([^"]+)"', re.I)
LIB_DESC_RE   = re.compile(r'<p\s+class="max-w-lg\b[^"]*">(.*?)</p>', re.I | re.S)
LIB_BADGE_RE  = re.compile(
    r'<span\s+class="inline-flex[^"]*?(?:bg-\[#([0-9a-fA-F]{3,8})\]|bg-([a-z]+-[0-9]{2,3}))?[^"]*?'
    r'(?:text-([a-z]+)-([0-9]{2,3}))?[^"]*">\s*(.*?)\s*</span>', re.I | re.S)
LIB_STAT_RE = re.compile(
    r'</svg>\s*(?:<span\s*>\s*)?([0-9][0-9.,]*\s*[KMBT]?)\s*(?:</span>\s*)?'
    r'<span\s+class="hidden sm:flex">(?:&nbsp;|\s)*([A-Za-z]+)', re.I)
LIB_STAT_FALLBACK_RE = re.compile(r'</svg>\s*(?:<span\s*>\s*)?([0-9][0-9.,]*\s*[KMBT]?)', re.I)
LIB_ABS_TS_RE = re.compile(r'title="([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}[^"]*)"')
LIB_REL_TS_RE = re.compile(r'<span\s*>([^<]*?\bago)\s*</span>', re.I)
LIB_TAG_STRIP_RE = re.compile(r"<[^>]+>")
LIB_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def lib_text(raw):
    if not raw:
        return ""
    s = LIB_TAG_STRIP_RE.sub(" ", raw)
    return re.sub(r"\s+", " ", _htmlmod.unescape(s)).strip()


def lib_num(raw):
    """'83.3M' -> 83300000. '91' -> 91. None if unparseable."""
    if not raw:
        return None
    s = raw.replace(",", "").strip()
    mult = 1
    if s and s[-1].upper() in LIB_SUFFIX:
        mult = LIB_SUFFIX[s[-1].upper()]
        s = s[:-1].strip()
    try:
        return int(round(float(s) * mult))
    except ValueError:
        return None


def lib_fetch(url=None, timeout=LIB_TIMEOUT):
    """Fetch the index. Raises -- callers decide what a failure means.

    The User-Agent is descriptive and honest. It is not a browser
    impersonation string: the point is that ollama.com can see who this is and
    block it if they want to, and if they ever do, the answer is to ask them,
    not to dress the request up as Chrome.
    """
    req = urllib.request.Request(url or LIB_URL,
                                 headers={"User-Agent": LIB_UA, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def lib_parse(doc):
    """(models, problems). Never raises on odd markup; a field it cannot find
    comes back None/[]. Each record carries `rank`, its 0-based page position --
    on the default sort that IS the popularity ranking, which makes it a usable
    sort key even on a day the pull counts fail to parse."""
    models, problems = [], []
    anchors = list(LIB_ANCHOR_RE.finditer(doc))
    if not anchors:
        return models, ["no /library/ anchors found at all -- markup changed"]

    seen = set()
    for i, m in enumerate(anchors):
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(doc)
        block = doc[m.start():end]
        # A real card carries an <h2>. Nav and footer links to /library/<x> do not.
        if "<h2" not in block:
            continue

        name = m.group(1)
        t = LIB_TITLE_RE.search(block)
        if t and t.group(1).strip():
            name = t.group(1).strip()
        if name in seen:
            problems.append("duplicate entry %r -- counted once" % name)
            continue
        seen.add(name)

        d = LIB_DESC_RE.search(block)
        caps, sizes, unknown = [], [], []
        for b in LIB_BADGE_RE.finditer(block):
            hexbg, palbg, tcol, tshade, label = b.groups()
            label = lib_text(label)
            if not label:
                continue
            low = label.lower()
            is_size_colour = bool(hexbg and hexbg.lower().startswith("ddf4ff")) or (tcol == "blue")
            if LIB_SIZE_RE.match(low) or is_size_colour:
                if low not in [s.lower() for s in sizes]:
                    sizes.append(label)
            elif low in LIB_KNOWN_CAPS:
                if low not in caps:
                    caps.append(low)
            else:
                unknown.append(label)
        if unknown:
            problems.append("%s: unclassified badge(s) %s -- add to LIB_KNOWN_CAPS or LIB_SIZE_RE"
                            % (name, ", ".join(repr(u) for u in sorted(set(unknown)))))

        stats = {}
        for s in LIB_STAT_RE.finditer(block):
            stats[s.group(2).strip().lower()] = lib_num(s.group(1))
        pulls, tags = stats.get("pulls"), stats.get("tags")
        if pulls is None or tags is None:
            # Positional fallback: the labels live in a `hidden sm:flex` span, so
            # a responsive-layout change could drop the labels and keep the
            # numbers. First number after an </svg> is pulls, second is tags.
            bare = LIB_STAT_FALLBACK_RE.findall(block)
            if pulls is None and len(bare) >= 1:
                pulls = lib_num(bare[0])
            if tags is None and len(bare) >= 2:
                tags = lib_num(bare[1])

        a = LIB_ABS_TS_RE.search(block)
        r = LIB_REL_TS_RE.search(block)
        rec = {"name": name, "repo": m.group(1),
               "url": "https://ollama.com/library/" + m.group(1),
               "description": lib_text(d.group(1)) if d else "",
               "capabilities": caps, "sizes": sizes,
               "pulls": pulls, "tag_count": tags,
               "updated": a.group(1) if a else None,
               "updated_rel": lib_text(r.group(1)) if r else None,
               "rank": len(models)}
        models.append(rec)
        for f in LIB_REQUIRED:
            if not rec.get(f):
                problems.append("%s: missing %s" % (name, f))
    return models, problems


def lib_build(doc, source=None, sort="popular"):
    models, problems = lib_parse(doc)
    return {"schema": 1, "source": source or LIB_URL, "sort": sort,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": len(models), "problems": problems, "models": models}


def lib_write(catalog, path=None, force=False):
    """Write atomically, refusing to clobber a healthier cache. RULE 4 lives here.

    Returns (written, message)."""
    path = path or LIB_PATH
    new_n, old_n = catalog.get("count", 0), 0
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old_n = json.load(f).get("count", 0)
        except Exception as e:
            catalog.setdefault("problems", []).append(
                "existing cache at %s was unreadable (%s) -- replacing it" % (path, e))
    if not force:
        if new_n == 0:
            return False, "REFUSED: parsed 0 entries, nothing to write."
        if new_n < LIB_MIN_PLAUSIBLE and old_n >= LIB_MIN_PLAUSIBLE:
            return False, ("REFUSED: parsed only %d entries, the cache holds %d. Keeping the "
                           "cache — the page layout probably changed." % (new_n, old_n))
        if old_n and new_n < old_n * 0.6:
            return False, ("REFUSED: parsed %d entries, a 40%%+ drop from the cached %d. "
                           "Keeping the cache." % (new_n, old_n))
    tmp = path + ".tmp"
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=1)
    os.replace(tmp, path)
    return True, "wrote %d models to %s (previous cache: %d)" % (new_n, path, old_n)


def lib_health(catalog):
    """Which REQUIRED fields are missing on more than 5% of records.

    The 5% is not arbitrary: capabilities and sizes are legitimately absent on
    half and an eighth of the library respectively, but description, pulls,
    tag_count and updated are universal on a healthy page, so a gap there is
    the contract breaking rather than the data being sparse.
    """
    ms = catalog.get("models") or []
    n = len(ms)
    if not n:
        return ["nothing parsed"]
    return ["%s missing on %d of %d" % (k, c, n)
            for k, c in ((k, sum(1 for m in ms if not m.get(k))) for k in LIB_REQUIRED)
            if 100.0 * c / n > 5]


# --- the in-memory library --------------------------------------------------
# Read from disk once at boot and after every sync. Serving /api/library out of
# a dict means the store paints with no network at all -- the internet is only
# touched when you press Sync.
LIBRARY = {"schema": 1, "count": 0, "models": [], "problems": [],
           "fetched_at": None, "source": None, "sort": "popular"}
LIB_LOCK = threading.Lock()
LIB_LAST = {"ok": None, "msg": "not synced yet", "at": None, "running": False}


def lib_load():
    """Load catalog.json into memory. Missing file is not an error -- it is the
    state before the first sync, and the UI says so with a button next to it."""
    global LIBRARY
    try:
        with open(LIB_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("models"), list):
            d.setdefault("count", len(d["models"]))
            with LIB_LOCK:
                LIBRARY = d
            return True
    except Exception:
        pass
    return False


def lib_sync(force=False):
    """Fetch, parse, guard, write, reload. Returns (ok, message)."""
    global LIBRARY
    with LIB_LOCK:
        if LIB_LAST.get("running"):
            return False, "a sync is already running"
        LIB_LAST["running"] = True
    try:
        try:
            doc = lib_fetch()
        except Exception as e:
            return False, "could not reach ollama.com (%s: %s)" % (type(e).__name__, e)
        cat = lib_build(doc, source=LIB_URL)
        bad = lib_health(cat)
        if bad and not force:
            # The parse came back structurally wrong. Do not write it, and say
            # which field broke -- that is the one thing that makes fixing the
            # regex a five-minute job instead of a hunt.
            return False, ("parsed %d entries but the markup contract looks broken: %s. "
                           "Kept the existing cache." % (cat["count"], "; ".join(bad)))
        ok, msg = lib_write(cat, force=force)
        if ok:
            lib_load()
        return ok, msg
    finally:
        with LIB_LOCK:
            LIB_LAST["running"] = False


def _library_rec(repo):
    """One library record by repo name, or None. Linear over 234 is nothing."""
    with LIB_LOCK:
        ms = (LIBRARY or {}).get("models") or []
    r = (repo or "").strip().lower()
    for m in ms:
        if (m.get("repo") or "").lower() == r:
            return m
    return None


def lib_age_hours():
    fa = (LIBRARY or {}).get("fetched_at")
    if not fa:
        return None
    try:
        t = calendar.timegm(time.strptime(fa, "%Y-%m-%dT%H:%M:%SZ"))
        return max(0.0, (time.time() - t) / 3600.0)
    except Exception:
        return None


lib_load()


# --- the curated catalog ---------------------------------------------------
# Two jobs. First, it is what the browser shows when the registry is
# unreachable - sizes from here are labelled as estimates, never as facts.
# Second, and more usefully, it carries the things the registry does NOT know:
# what the model is for, whether it is a MoE and how much of it is active per
# token, what its native context is, and what it is actually good at. That is
# the part you cannot get from a byte count.
#
# `gb` figures are as published on ollama.com and are decimal GB, matching what
# the site shows. The fit calculator converts to GiB before comparing to VRAM,
# because sysfs reports binary units and mixing the two is a 7% lie.
CATALOG = [
    {
        "repo": "qwen3.6", "name": "Qwen3.6", "vendor": "Alibaba", "license": "Apache 2.0",
        "ctx": 262144, "moe": True,
        "tasks": ["code", "long context", "general", "agentic"],
        "blurb": "The current generation of Qwen. The 35B is a mixture of experts "
                 "with only about 3B parameters active per token, which is why it "
                 "runs far faster than its size suggests. Strong at code and at "
                 "long-document work.",
        "tags": [
            {"tag": "35b-a3b-q4_K_M", "gb": 24, "note": "3B active. What is installed now (as an Unsloth Dynamic import)."},
            {"tag": "35b-a3b-q8_0",   "gb": 39, "note": "Same model at q8. Best quality that still fits comfortably."},
            {"tag": "35b-a3b-mtp-q8_0", "gb": 39, "note": "q8 plus multi-token prediction."},
            {"tag": "35b-a3b-mtp-q4_K_M", "gb": 23, "note": "Cheapest way to test whether MTP does anything here."},
            {"tag": "27b-q4_K_M",     "gb": 17, "note": "Dense 27B. Smaller and slower per token than the 35B MoE."},
            {"tag": "27b-q8_0",       "gb": 30, "note": "Dense 27B at q8."},
            {"tag": "35b-a3b-coding-nvfp4", "gb": 22, "note": "Coding tune — but NVFP4, so not for these cards."},
        ],
    },
    {
        "repo": "qwen3", "name": "Qwen3", "vendor": "Alibaba", "license": "Apache 2.0",
        "ctx": 32768, "moe": True,
        "tasks": ["general", "code"],
        "blurb": "Previous generation. Kept here because the 30B-A3B is a well "
                 "understood quantity and the 32B dense is a useful control when "
                 "you want to isolate whether a problem is the MoE path.",
        "tags": [
            {"tag": "30b-a3b",        "gb": 19, "note": "3B active MoE. Fast."},
            {"tag": "30b-a3b-q8_0",   "gb": 33, "note": "Same at q8."},
            {"tag": "32b",            "gb": 20, "note": "Dense 32B — every parameter reads every token."},
            {"tag": "32b-q8_0",       "gb": 35, "note": "Dense 32B at q8."},
            {"tag": "235b-a22b",      "gb": 142, "note": "Far past this machine."},
        ],
    },
    {
        "repo": "qwen3-coder", "name": "Qwen3 Coder", "vendor": "Alibaba", "license": "Apache 2.0",
        "ctx": 262144, "moe": True,
        "tasks": ["code", "agentic", "long context"],
        "blurb": "Qwen3 tuned hard for code and for agentic tool use. The 30B-A3B "
                 "is the sweet spot on this box: it fits on one card with room for "
                 "a long context, so nothing crosses the PCIe bus.",
        "tags": [
            {"tag": "30b",            "gb": 19, "note": "30B-A3B. Fits one card."},
            {"tag": "30b-a3b-q8_0",   "gb": 32, "note": "q8. Needs both cards."},
            {"tag": "480b",           "gb": 290, "note": "Not on this planet."},
        ],
    },
    {
        "repo": "gpt-oss", "name": "GPT-OSS", "vendor": "OpenAI", "license": "Apache 2.0",
        "ctx": 131072, "moe": True,
        "tasks": ["general", "reasoning"],
        "blurb": "OpenAI's open-weight release. The 20B is small enough to sit "
                 "beside something else on the same card, which makes it a good "
                 "second model for quick jobs. The 120B does not fit here.",
        "tags": [
            {"tag": "20b",  "gb": 14, "note": "Comfortable. Leaves room for a long context."},
            {"tag": "120b", "gb": 65, "note": "Over the 60 GiB total. Would spill to CPU."},
        ],
    },
    {
        "repo": "llama4", "name": "Llama 4", "vendor": "Meta", "license": "Llama 4 Community",
        "ctx": 1048576, "moe": True,
        "tasks": ["general", "long context", "vision"],
        "blurb": "Scout is the small one and it is still 67 GB, which is over this "
                 "machine's total VRAM. Listed so the answer is visible without "
                 "downloading two thirds of it to find out.",
        "tags": [
            {"tag": "scout", "gb": 67, "note": "Over budget. 67 GB against 60 GiB of card."},
        ],
    },
    {
        "repo": "gemma3", "name": "Gemma 3", "vendor": "Google", "license": "Gemma Terms",
        "ctx": 131072, "moe": False,
        "tasks": ["general", "vision"],
        "blurb": "Dense, multimodal, and unusually strong for its size. The 27B is "
                 "a good single-card general model when you want something that is "
                 "not a Qwen for comparison.",
        "tags": [
            {"tag": "27b",      "gb": 17, "note": "Dense 27B, q4."},
            {"tag": "27b-it-q8_0", "gb": 30, "note": "q8."},
            {"tag": "12b",      "gb": 8,  "note": "Small and quick."},
        ],
    },
    {
        "repo": "mistral-small", "name": "Mistral Small", "vendor": "Mistral AI", "license": "Apache 2.0",
        "ctx": 131072, "moe": False,
        "tasks": ["general", "code"],
        "blurb": "Dense 24B. Fits on one card with a lot of headroom, which makes "
                 "it a reasonable always-resident model for small tasks.",
        "tags": [
            {"tag": "24b", "gb": 15, "note": "Dense, one card, plenty of room."},
        ],
    },
    {
        "repo": "deepseek-r1", "name": "DeepSeek-R1", "vendor": "DeepSeek", "license": "MIT",
        "ctx": 131072, "moe": True,
        "tasks": ["reasoning", "code"],
        "blurb": "Reasoning model — it thinks visibly before answering, which costs "
                 "tokens and time but helps on hard problems. The distilled 32B is "
                 "the one that fits here.",
        "tags": [
            {"tag": "32b",  "gb": 20, "note": "Distilled onto Qwen2.5-32B. One card."},
            {"tag": "70b",  "gb": 43, "note": "Distilled onto Llama-70B. Spans both cards."},
            {"tag": "671b", "gb": 404, "note": "The real one. No."},
        ],
    },
    {
        "repo": "nomic-embed-text", "name": "Nomic Embed Text", "vendor": "Nomic", "license": "Apache 2.0",
        "ctx": 8192, "moe": False,
        "tasks": ["embedding"],
        "blurb": "An embedding model, not a chat model. This is what turns "
                 "documents into vectors for Open WebUI's knowledge bases. Tiny, "
                 "and worth having resident permanently.",
        "tags": [{"tag": "latest", "gb": 0.274, "note": "274 MB. Costs nothing to keep loaded."}],
    },
]

CATALOG_BY_REPO = {c["repo"]: c for c in CATALOG}

# --- reading a tag name ----------------------------------------------------
# Tag names are the only place parameter counts are published in a machine
# readable way. "35b-a3b-q8_0" means a 35B model with 3B active per token; the
# manifest says neither. Parsing is not elegant but it is the difference
# between a browser you can filter and a list you have to read.
_PARAM_RE  = re.compile(r"(?:^|[-_])(\d+(?:\.\d+)?)b(?:$|[-_])")
_ACTIVE_RE = re.compile(r"[-_]a(\d+(?:\.\d+)?)b(?:$|[-_])")
_QUANT_RE  = re.compile(r"(q\d+(?:_[a-z0-9]+)*|bf16|fp16|f16|fp8|mxfp8|nvfp4|int8|int4)",
                        re.IGNORECASE)

def tag_params_b(tag):
    """Total parameters in billions, from the tag name. None if unstated.

    Deliberately anchored to a separator so the 3 in "a3b" - which is the
    ACTIVE count, not the total - cannot be mistaken for the model's size.
    """
    m = _PARAM_RE.search((tag or "").lower())
    return float(m.group(1)) if m else None

def tag_active_b(tag):
    """Active parameters per token for an MoE tag, or None for a dense one."""
    m = _ACTIVE_RE.search((tag or "").lower())
    return float(m.group(1)) if m else None

def tag_quant(tag):
    m = _QUANT_RE.search(tag or "")
    return m.group(1).lower() if m else None

def catalog_size_bytes(repo, tag):
    """(bytes, src) for one tag. The size lookup, factored out of the endpoint
    so the family summary can reuse it without going back through HTTP."""
    b = registry_weight_bytes(repo, tag) or registry_size(repo, tag)
    if b:
        return b, "registry"
    c = CATALOG_BY_REPO.get(repo)
    for t in ((c or {}).get("tags") or []):
        if t["tag"] == tag:
            return int(t["gb"] * 1e9), "catalog"
    return None, "none"

# Which verdict is preferable, for picking a family's best tag. Lower is
# better. This ordering is the whole opinion of the tab: staying on one card
# beats every amount of extra quality you buy by spanning two.
VERDICT_RANK = {"one_card": 0, "one_card_tight": 1, "spans": 2, "too_big": 3, "unknown": 4}

def _prefetch_sizes(pairs, deadline=5.0, workers=6):
    """Warm the registry cache for many tags at once, then give up waiting.

    Sizing thirty tags one after another means thirty sequential six-second
    timeouts on a cold cache, which is a minute of blank page. These run
    concurrently and the request stops WAITING after `deadline` - but the
    threads are not cancelled, so whatever is still in flight lands in the
    hour-long cache and the next load is both instant and accurate. A first
    paint that falls back to catalog figures and then self-corrects is a much
    better failure than a page that hangs.
    """
    try:
        import concurrent.futures as _cf
    except Exception:
        return
    ex = _cf.ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(catalog_size_bytes, r, t) for r, t in pairs]
        _cf.wait(futs, timeout=deadline)
    except Exception:
        pass
    finally:
        ex.shutdown(wait=False)

def _tag_detail(repo, t, cards, ctx, reserve):
    tag = t["tag"] if isinstance(t, dict) else t
    note = t.get("note") if isinstance(t, dict) else None
    sev, why = tag_warning(tag)
    b, src = catalog_size_bytes(repo, tag)
    gib = (b / GIB) if b else None
    fit = (fit_verdict(gib, cards, ctx=ctx, kv_bytes=1, reserve_gib=reserve)
           if gib is not None else {"verdict": "unknown", "why": "No size available."})
    return {"tag": tag, "note": note, "warn": sev, "warn_why": why,
            "src": src, "bytes": b, "gib": (round(gib, 2) if gib is not None else None),
            "params_b": tag_params_b(tag), "active_b": tag_active_b(tag),
            "quant": tag_quant(tag), "fit": fit}

def _best_tag(tags):
    """The tag to put on the family card: the best quality that fits best.

    Sorted by verdict first, then by size DESCENDING - a bigger file at the
    same verdict is a better quantisation. The exception is too_big, where
    bigger is simply further out of reach, so that one sorts ascending.
    """
    runnable = [t for t in tags if t.get("warn") != "bad" and t.get("gib") is not None]
    if not runnable:
        return None
    def key(t):
        r = VERDICT_RANK.get(t["fit"].get("verdict"), 9)
        return (r, t["gib"] if r >= 3 else -t["gib"])
    return sorted(runnable, key=key)[0]

def family_summary(c, cards, ctx=None, reserve=None):
    """One family, reduced to what a card in the grid needs to show."""
    tags = [_tag_detail(c["repo"], t, cards, ctx, reserve) for t in (c.get("tags") or [])]
    best = _best_tag(tags)
    runnable = [t for t in tags if t.get("warn") != "bad"]
    sizes = [t["params_b"] for t in tags if t.get("params_b")]
    return {
        "repo": c["repo"], "name": c["name"], "vendor": c["vendor"],
        "license": c["license"], "ctx": c["ctx"], "moe": c["moe"],
        "blurb": c["blurb"], "tasks": c.get("tasks") or [],
        "tags_total": len(tags), "tags_runnable": len(runnable),
        "tags_blocked": len(tags) - len(runnable),
        "params_min": (min(sizes) if sizes else None),
        "params_max": (max(sizes) if sizes else None),
        "best": best,
        "verdict": (best or {}).get("fit", {}).get("verdict", "unknown"),
    }

# --- hardware + fit --------------------------------------------------------
def detected_cards():
    """[{slot, vram_total, vram_free}] for every amdgpu card, largest first.

    This is the 'checks your hardware' half. It reads the same
    mem_info_vram_total the VRAM panel reads, so if the two ever disagree one of
    them is broken and you will be able to tell. Nothing here is hardcoded to
    two cards or to 30 GiB - pull a card and the verdicts change.
    """
    mem  = read_gpu_mem()
    gpus = read_gpu_temps()
    slot_by_pci = {g.get("pci"): (g.get("slot") or g.get("pci")) for g in gpus}
    out = []
    for pci, m in mem.items():
        vt = m.get("vram_total")
        if not vt:
            continue
        out.append({
            "pci": pci,
            "slot": slot_by_pci.get(pci) or pci,
            "vram_total": vt,
            "vram_free": m.get("vram_free"),
            "vram_used": m.get("vram_used"),
        })
    out.sort(key=lambda c: -(c["vram_total"] or 0))
    return out

def fit_verdict(weights_gib, cards, ctx=None, kv_bytes=1, reserve_gib=None):
    """Where a model of this size lands on the cards that are actually present.

    Returns a dict the UI renders directly. The verdicts, and why they are
    ranked the way they are:

      one_card       weights + KV fit inside a single card. Nothing crosses the
                     bus at any point. This is the case to want.
      one_card_tight fits a card, but with under 2 GiB spare. It will load and
                     then fall over the first time you raise the context.
      spans          needs more than one card. Ollama splits by layer, so every
                     token walks across PCIe at every layer boundary. It works.
                     It is slower than the same weights on one card, and on this
                     box there is no xGMI bridge to soften it.
      too_big        exceeds total VRAM. Ollama will silently place the overflow
                     in host RAM, which is not an error and not a warning - it
                     is a model that suddenly runs at a few tokens a second.
    """
    if not cards:
        return {"verdict": "unknown", "why": "No amdgpu cards visible to this container.",
                "need_gib": None, "kv_gib": None}
    kv = reserve_gib if reserve_gib is not None else kv_estimate_gib(weights_gib, ctx, kv_bytes)
    need = (weights_gib or 0) + kv
    biggest = (cards[0]["vram_total"] or 0) / GIB
    total   = sum((c["vram_total"] or 0) for c in cards) / GIB
    # Per-card driver/framebuffer overhead. The V620s report ~30 GiB usable of a
    # nominal 32, and the framebuffer plus the driver's own allocations take a
    # further bite that does not appear in mem_info_vram_total's arithmetic.
    OVERHEAD = 0.8
    b = max(0.0, biggest - OVERHEAD)
    t = max(0.0, total - OVERHEAD * len(cards))
    if need <= b - 2.0:
        v, why = "one_card", "Fits one card with room to spare. Nothing crosses PCIe."
    elif need <= b:
        v, why = "one_card_tight", ("Fits one card, but with under 2 GiB spare. "
                                    "Raising the context will push it over.")
    elif need <= t:
        v, why = "spans", ("Needs more than one card. Ollama will split it by layer; "
                           "expect a latency tax at every layer boundary.")
    else:
        v, why = "too_big", ("Over total VRAM. Ollama will place the remainder in host "
                            "RAM without warning you, and it will crawl.")
    return {"verdict": v, "why": why,
            "need_gib": round(need, 1), "kv_gib": round(kv, 2),
            "biggest_gib": round(b, 1), "total_gib": round(t, 1),
            "cards": len(cards)}

# --- disk ------------------------------------------------------------------
def models_dir_state():
    """Is the blob directory mounted, and how much room is left on it."""
    d = OLLAMA_MODELS_DIR
    if not d or not os.path.isdir(d):
        return {"mounted": False, "path": d}
    out = {"mounted": True, "path": d}
    try:
        st = os.statvfs(d)
        out["free"]  = st.f_bavail * st.f_frsize
        out["total"] = st.f_blocks * st.f_frsize
    except Exception as e:
        out["error"] = str(e)
    return out

# A blob filename and nothing else. This is the jail: the delete endpoint will
# not touch a path that does not match, and will not touch one that resolves
# outside the mounted directory. Both checks, not either.
PARTIAL_RE = re.compile(r"^sha256-[0-9a-f]{64}-partial(-\d+)?$")

def _blobs_dir():
    d = OLLAMA_MODELS_DIR
    for cand in (os.path.join(d, "blobs"), d):
        if os.path.isdir(cand):
            # The real blobs directory has blobs in it.
            try:
                for n in os.listdir(cand):
                    if n.startswith("sha256-"):
                        return cand
            except Exception:
                pass
    return os.path.join(d, "blobs") if os.path.isdir(os.path.join(d, "blobs")) else None

def list_partials():
    """Half-finished downloads, with REAL progress rather than apparent size.

    This is the single most confusing thing about an interrupted Ollama pull and
    it wasted an evening: `ls -l` on a -partial file shows the TARGET size, not
    how much has arrived. Ollama creates the file at full length up front and
    writes chunks into it at offsets, so a download that is 11% done looks like
    a completed 36 GB file. The number that tells the truth is blocks actually
    allocated on disk - st_blocks * 512 - which is what `du` reports and what
    this function returns as `on_disk`.
    """
    bd = _blobs_dir()
    if not bd:
        return []
    out = []
    try:
        names = os.listdir(bd)
    except Exception:
        return []
    for n in names:
        if not n.endswith("-partial") or not PARTIAL_RE.match(n):
            continue
        p = os.path.join(bd, n)
        try:
            st = os.stat(p)
        except Exception:
            continue
        chunks = 0
        for m in names:
            if m.startswith(n + "-") and PARTIAL_RE.match(m):
                chunks += 1
        out.append({
            "name": n,
            "digest": n[7:7 + 12],
            "target": st.st_size,               # apparent size = what it will be
            "on_disk": st.st_blocks * 512,      # allocated blocks = what has arrived
            "mtime": st.st_mtime,
            "age_s": max(0, int(time.time() - st.st_mtime)),
            "chunks": chunks,
        })
    out.sort(key=lambda x: -x["on_disk"])
    return out

def delete_partial(name):
    """Remove one partial and its chunk bookkeeping files. Jailed, twice."""
    if not PARTIAL_RE.match(name or "") or not name.endswith("-partial"):
        return False, "not a partial blob name"
    bd = _blobs_dir()
    if not bd:
        return False, "blobs directory is not mounted"
    root = os.path.realpath(bd)
    killed = 0
    try:
        names = os.listdir(bd)
    except Exception as e:
        return False, str(e)
    for m in names:
        if m != name and not m.startswith(name + "-"):
            continue
        if not PARTIAL_RE.match(m):
            continue
        p = os.path.realpath(os.path.join(bd, m))
        # Second half of the jail: after symlinks are resolved, is it still
        # inside the directory we were given? A name check alone is not enough.
        if os.path.dirname(p) != root:
            continue
        try:
            os.remove(p); killed += 1
        except Exception:
            pass
    return (killed > 0), ("removed %d file(s)" % killed if killed else "nothing removed")

# --- pull manager ----------------------------------------------------------
# One entry per tag. Threads write, HTTP handlers read; everything that touches
# the dict holds _pull_lock. Layers are kept in insertion order so the UI can
# draw them as a stack that grows downwards the way the CLI does.
PULLS = {}
_pull_lock = threading.Lock()
# Concurrency of one, on purpose. Three at once is what died: they share the
# same disk and the same uplink, so running them in parallel makes all three
# slower and makes any one of them take three times as long to become useful.
# Anything started while one is running is queued, not refused.
PULL_QUEUE = []

def _pull_new(tag):
    return {"tag": tag, "state": "queued", "started": None, "finished": None,
            "status": "queued", "layers": {}, "order": [], "error": None,
            "cancel": False, "completed": 0, "total": 0}

def _pull_worker(tag):
    with _pull_lock:
        p = PULLS.get(tag)
        if not p:
            return
        p["state"] = "running"; p["started"] = time.time(); p["status"] = "connecting"
    try:
        body = json.dumps({"model": tag, "stream": True}).encode()
        req = urllib.request.Request(OLLAMA_URL + "/api/pull", data=body,
                                     headers={"Content-Type": "application/json"})
        # No read timeout: the registry can go quiet for a while mid-blob on a
        # home connection and a timeout here would kill a healthy download. The
        # cancel flag below is the way out, and the UI shows how long it has
        # been since the last byte so a genuinely dead pull is visible.
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                with _pull_lock:
                    if p["cancel"]:
                        p["state"] = "cancelled"; p["status"] = "cancelled"
                        break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw.decode("utf-8", "replace"))
                except Exception:
                    continue
                if d.get("error"):
                    with _pull_lock:
                        p["state"] = "error"; p["error"] = str(d["error"])
                    break
                st = d.get("status") or ""
                dg = d.get("digest")
                with _pull_lock:
                    p["status"] = st
                    p["last"] = time.time()
                    if dg:
                        key = dg.split(":")[-1][:12]
                        if key not in p["layers"]:
                            p["order"].append(key)
                        p["layers"][key] = {
                            "digest": key,
                            "total": int(d.get("total") or 0),
                            "completed": int(d.get("completed") or 0),
                            "status": st,
                        }
                        p["total"] = sum(l["total"] for l in p["layers"].values())
                        p["completed"] = sum(l["completed"] for l in p["layers"].values())
                    if st == "success":
                        p["state"] = "done"
        with _pull_lock:
            if p["state"] == "running":
                # The stream ended without a "success" line. That is not a
                # completed pull and it must not be shown as one.
                p["state"] = "error"
                p["error"] = p.get("error") or "stream ended before success"
    except Exception as e:
        with _pull_lock:
            if p.get("cancel"):
                p["state"] = "cancelled"; p["status"] = "cancelled"
            else:
                p["state"] = "error"; p["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        with _pull_lock:
            p["finished"] = time.time()
        _pull_pump()

def _pull_pump():
    """Start the next queued pull if nothing is running. Caller must not hold the lock."""
    with _pull_lock:
        running = any(x["state"] == "running" for x in PULLS.values())
        if running:
            return
        nxt = None
        while PULL_QUEUE:
            cand = PULL_QUEUE.pop(0)
            pc = PULLS.get(cand)
            if pc and pc["state"] == "queued" and not pc["cancel"]:
                nxt = cand; break
        if not nxt:
            return
    # Two workers, one queue. Concurrency of one still applies across both:
    # they share the same disk and the same uplink, so a lane download racing
    # an Ollama pull would make both slower.
    worker = _lane_worker if nxt.startswith(LANE_TAG_PREFIX) else _pull_worker
    threading.Thread(target=worker, args=(nxt,), daemon=True).start()

def pull_start(tag):
    if not OLLAMA_ENABLED:
        return False, OLLAMA_OFF_MSG
    tag = (tag or "").strip()
    if not tag or len(tag) > 200 or any(c.isspace() for c in tag):
        return False, "bad model tag"
    with _pull_lock:
        p = PULLS.get(tag)
        if p and p["state"] in ("queued", "running"):
            return False, "already " + p["state"]
        PULLS[tag] = _pull_new(tag)
        PULL_QUEUE.append(tag)
    _pull_pump()
    return True, "queued"

def pull_cancel(tag):
    with _pull_lock:
        p = PULLS.get(tag)
        if not p:
            return False, "no such pull"
        if p["state"] not in ("queued", "running"):
            return False, "not running"
        p["cancel"] = True
        if p["state"] == "queued":
            p["state"] = "cancelled"; p["status"] = "cancelled"
            p["finished"] = time.time()
    _pull_pump()
    return True, "cancelling"

def pull_snapshot():
    now = time.time()
    with _pull_lock:
        out = []
        for tag, p in PULLS.items():
            layers = [p["layers"][k] for k in p["order"] if k in p["layers"]]
            el = None
            if p["started"]:
                el = int((p["finished"] or now) - p["started"])
            rate = None
            if p["state"] == "running" and el and p["completed"]:
                rate = p["completed"] / max(1, el)
            eta = None
            if rate and p["total"] > p["completed"]:
                eta = int((p["total"] - p["completed"]) / rate)
            out.append({
                "tag": tag, "state": p["state"], "status": p["status"],
                "error": p["error"], "completed": p["completed"], "total": p["total"],
                "layers": layers, "elapsed": el, "rate": rate, "eta": eta,
                "quiet_s": int(now - p["last"]) if p.get("last") else None,
            })
        out.sort(key=lambda x: (x["state"] != "running", x["tag"]))
        return out

# --- Hugging Face -> Ollama ------------------------------------------------
# The whole feature rests on one fact, and it is worth stating plainly because
# it is what kept this from becoming a second download engine: Ollama can pull
# a GGUF straight out of a Hugging Face repo.
#
#     ollama pull hf.co/{owner}/{repo}:{QUANT}
#
# So nothing below downloads anything. It resolves a Hugging Face URL into the
# list of quantisations that repo actually contains, tells you what each one
# would do to these two cards, and then hands the chosen tag to pull_start() -
# the same pull manager the Pulls tab uses. Progress, per-layer bars, rate,
# ETA, cancel, queueing, resume-from-partial: all of it already existed and all
# of it works here unchanged, because to Ollama this is just another pull.
#
# What this app deliberately does NOT do is hold a Hugging Face token. A gated
# repo is reported as gated and you are told how to authorise it at the Ollama
# end (its SSH key, added to your HF account). No credential ever passes
# through this container.
HF_HOST = os.environ.get("HF_HOST", "https://huggingface.co").rstrip("/")

# --- the llama lane as a second destination ---------------------------------
# Ollama is not the only inference backend on this box any more. llama-swap on
# :8090 reads plain .gguf files out of a directory, and it CANNOT read models
# that Ollama converted for its own engine — a Qwen MoE pulled from Ollama's
# library fails with "key qwen35moe.rope.dimension_sections has wrong array
# length; expected 4, got 3". Files pulled straight from Hugging Face are the
# ones llama.cpp's own converter produced, so this tab is the natural place to
# feed the lane.
#
# Set LANE_MODELS_DIR to the lane's models directory, bind-mounted into this
# container. If the path does not exist the destination picker stays hidden and
# nothing about the Ollama path changes.
LANE_MODELS_DIR = os.environ.get("LANE_MODELS_DIR", "/lane-models")
def lane_available():
    return bool(LANE_MODELS_DIR) and os.path.isdir(LANE_MODELS_DIR)
# Lane pulls live in the same PULLS dict as Ollama pulls so the Pulls tab, the
# queue, cancel and the progress rendering all work unchanged. This prefix is
# what tells them apart — and it also means pulling the same repo to both
# destinations is two separate entries rather than a collision.
LANE_TAG_PREFIX = "lane:"

# The lane's config file and its API, so this dashboard can show what
# llama-swap knows about and add to it. llama-swap has NO model
# auto-discovery — every model needs a hand-written entry, because a bare
# .gguf does not say what flags to run it with (Muse wants --top-k 64, Qwen
# wants 20; one needs --mmproj, the other has no such file). Guessing those
# is how you get a model that runs but quietly answers worse. So this panel
# does the finding and lets a human confirm the flags.
LANE_CONFIG = os.environ.get("LANE_CONFIG", "/lane-config/llama-swap.yaml")
LANE_URL    = os.environ.get("LANE_URL", "http://127.0.0.1:8090").rstrip("/")

# --- more than one lane ------------------------------------------------------
# Multiple llama-swap lanes are common: a stable one plus test lanes (e.g.
# fork) and llama-swap-flashnext on :8096 (the fork's Qwen4Exp branch). They
# all serve the SAME models directory (/data/models), so LANE_MODELS_DIR and
# the HF pulls are shared; what differs per lane is its CONFIG FILE and its
# URL. The registry below is what the Lane tab iterates.
#
# Override with the LANES env — semicolon-separated "name|config-path|url":
#   LANES="main|/lane-config/llama-swap.yaml|http://127.0.0.1:8090;test|/lane-config-test/llama-swap.yaml|http://127.0.0.1:8092"
# With LANES unset, three conventional lanes are offered, their URLs
# derived from LANE_URL by swapping the port. A lane whose config mount is
# missing still shows its running models (URL side) — the tab tells you the
# config is not mounted rather than hiding the lane.
def _lane_registry():
    env = os.environ.get("LANES", "").strip()
    reg = {}
    if env:
        for part in env.split(";"):
            bits = part.split("|")
            if len(bits) == 3 and bits[0].strip():
                reg[bits[0].strip()] = {"config": bits[1].strip(),
                                        "url": bits[2].strip().rstrip("/")}
        if reg:
            return reg
    # Default: ONE lane, from LANE_URL / LANE_CONFIG. Most people run one
    # llama-swap. Extra lanes are opt-in through LANES so nobody inherits
    # somebody else's port numbering.
    reg = {"main": {"config": LANE_CONFIG, "url": LANE_URL}}
    # Convenience: LANE_EXTRA_PORTS="8092,8096" adds lane2/lane3 on the same
    # host, each expecting /lane-config-lane2/llama-swap.yaml etc.
    extra = os.environ.get("LANE_EXTRA_PORTS", "").strip()
    if extra:
        for i, p in enumerate([x.strip() for x in extra.split(",") if x.strip()], 2):
            try:
                port_n = int(p)
            except ValueError:
                continue
            name = "lane%d" % i
            reg[name] = {"config": "/lane-config-%s/llama-swap.yaml" % name,
                         "url": re.sub(r":\d+$", ":%d" % port_n, LANE_URL)}
    return reg


def _lane_target(lane=None):
    """(config_path, url) for a lane name; unknown/None -> production."""
    reg = _lane_registry()
    if lane in reg:
        return reg[lane]["config"], reg[lane]["url"]
    first = next(iter(reg))
    return reg[first]["config"], reg[first]["url"]

# The owner half deliberately forbids a dot. Hugging Face usernames and orgs
# are letters, digits and hyphens only, and that one restriction is what stops
# a pasted link to some OTHER site from being accepted: "example.com/foo/bar"
# would otherwise parse as the repo "example.com/foo" and fail later as a
# confusing 404 instead of "that is not a Hugging Face link".
HF_REPO_RE  = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,94}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
HF_QUANT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# A split GGUF ends "-00001-of-00003". Strip that before reading the quant, or
# every shard of a big model shows up as its own bogus quantisation.
HF_SPLIT_RE = re.compile(r"-(\d{4,5})-of-(\d{4,5})$")
# The quant vocabulary llama.cpp actually emits. Anything not matching this is
# not treated as a quant name, which is what stops "Instruct" and "v0.2" from
# being offered as if they were quantisations.
HF_QTOK_RE = re.compile(
    r"^(?:IQ\d+(?:_[A-Za-z0-9]+)*|Q\d+(?:_[A-Za-z0-9]+)*"
    r"|BF16|F16|F32|FP16|FP32|MXFP4|TQ\d+(?:_\d+)?)$", re.I)
HF_HOSTS = ("huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co")

_hf_cache = {}
_hf_lock  = threading.Lock()
HF_TTL      = 900.0
HF_TTL_MISS = 60.0


def _hf_get(path, timeout=10.0):
    """GET a Hugging Face JSON API path. Returns (value, error_text).

    Cached the same way the Ollama registry lookups are, and for the same
    reason: this runs behind a UI that polls, and a repo that does not exist
    must not turn into a request storm against a service that has been nothing
    but polite. Failures are cached too, briefly.
    """
    now = time.time()
    with _hf_lock:
        hit = _hf_cache.get(path)
        if hit and now < hit[0]:
            return hit[1], hit[2]
    val, err = None, None
    try:
        req = urllib.request.Request(HF_HOST + path, headers={
            "Accept": "application/json",
            "User-Agent": "v620-dash/1",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            val = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        code = getattr(e, "code", None)
        if code in (401, 403):
            err = ("gated or private. Hugging Face wants you authenticated for this repo — "
                   "usually because you have to accept its licence on the model page first. "
                   "This app holds no Hugging Face credentials on purpose. Accept the licence, "
                   "then authorise Ollama itself by adding its SSH public key to your Hugging "
                   "Face account; after that the pull below works with no token here.")
        elif code == 404:
            err = "no such repo on Hugging Face (or it is private)."
        elif code == 429:
            err = "Hugging Face is rate-limiting this container. Wait a minute and retry."
        elif code:
            err = "Hugging Face returned HTTP %s" % code
        else:
            err = "%s: %s" % (type(e).__name__, e)
    with _hf_lock:
        _hf_cache[path] = (now + (HF_TTL if val is not None else HF_TTL_MISS), val, err)
    return val, err


def hf_parse(text):
    """Turn whatever got pasted into (repo, quant_or_None).

    These are the forms that actually get pasted, and all of them work:
        https://huggingface.co/bartowski/Qwen3-32B-GGUF
        https://huggingface.co/bartowski/Qwen3-32B-GGUF/tree/main
        huggingface.co/unsloth/gpt-oss-120b-GGUF/blob/main/gpt-oss-120b-Q4_K_M.gguf
        hf.co/bartowski/Qwen3-32B-GGUF:Q4_K_M
        bartowski/Qwen3-32B-GGUF

    A link to a specific .gguf file is the useful case: the quantisation is read
    off the filename, so pasting the download link for the exact file you were
    looking at selects that exact quantisation.
    """
    s = (text or "").strip()
    if not s:
        return None, None
    s = s.split("#", 1)[0].split("?", 1)[0].strip()
    low = s.lower()
    for pfx in ("https://", "http://", "git+https://"):
        if low.startswith(pfx):
            s = s[len(pfx):]
            low = s.lower()
            break
    for h in HF_HOSTS:
        if low == h or low.startswith(h + "/"):
            s = s[len(h):].lstrip("/")
            break
    s = s.strip("/")
    if not s:
        return None, None

    quant = None
    # /blob/<rev>/<path>, /resolve/<rev>/<path>, /tree/<rev>[/<path>]
    parts = s.split("/")
    cut = None
    for i, p in enumerate(parts):
        if p in ("blob", "resolve", "tree", "raw") and i >= 2:
            cut = i
            break
    if cut is not None:
        tail = parts[cut + 2:]          # everything after the revision
        repo = "/".join(parts[:cut])
        if tail and tail[-1].lower().endswith(".gguf"):
            quant = hf_quant_of(("/".join(tail)))
    else:
        # Possibly "owner/repo:QUANT". Only treat a colon as a tag separator
        # when what precedes it is exactly owner/repo — otherwise a stray colon
        # in a path would silently eat part of the name.
        if ":" in s:
            head, _, t = s.partition(":")
            if head.count("/") == 1 and t:
                s, quant = head, t
        repo = "/".join(s.split("/")[:2])

    repo = repo.strip("/")
    if not HF_REPO_RE.match(repo or ""):
        return None, None
    if quant and not HF_QUANT_RE.match(quant):
        quant = None
    return repo, quant


def hf_quant_of(path):
    """The quantisation a GGUF path represents, or None.

    Two layouts in the wild and both have to work:
      flat      Qwen3-32B-Q4_K_M.gguf                     -> quant in the filename
      foldered  Q4_K_M/Qwen3-235B-Q4_K_M-00001-of-00003.gguf -> quant is the folder
    """
    if not path or not path.lower().endswith(".gguf"):
        return None
    base = os.path.basename(path)[:-5]
    m = HF_SPLIT_RE.search(base)
    if m:
        base = base[:m.start()]
    tok = base.rsplit("-", 1)[-1]
    if HF_QTOK_RE.match(tok):
        return tok.upper()
    tok = base.rsplit(".", 1)[-1]
    if HF_QTOK_RE.match(tok):
        return tok.upper()
    d = os.path.dirname(path).split("/")[-1] if "/" in path else ""
    if d and HF_QTOK_RE.match(d):
        return d.upper()
    return None


def hf_split_parts(path):
    m = HF_SPLIT_RE.search(os.path.basename(path or "")[:-5])
    return (int(m.group(2)) if m else 1)


def hf_quants(repo):
    """Every GGUF quantisation in the repo, with its true total size.

    Sizes come from the tree API's lfs.size, which is the real object size. The
    plain `size` field on an LFS entry is the size of the pointer file — a
    couple of hundred bytes — and trusting it would report a 40 GB model as
    negligible, which is exactly the kind of confident-and-wrong number this
    dashboard exists to avoid.
    """
    tree, err = _hf_get("/api/models/%s/tree/main?recursive=true" % repo)
    if tree is None:
        return None, err, []
    if not isinstance(tree, list):
        return None, "unexpected answer from the Hugging Face tree API", []

    warn = []
    if len(tree) >= 1000:
        warn.append("This repo has 1000+ files and the listing is paginated; "
                    "quantisations beyond the first page will not appear here. "
                    "Pasting a direct link to the .gguf you want still works.")

    groups, unnamed = {}, 0
    mmprojs = []
    for e in tree:
        if not isinstance(e, dict):
            continue
        if (e.get("type") or "file") != "file":
            continue
        path = e.get("path") or ""
        if not path.lower().endswith(".gguf"):
            continue
        # Vision projectors (mmproj-*.gguf) are sidecars, not quantisations.
        # Left in, they pollute the F16/BF16 groups (mmproj-BF16.gguf reads as
        # a BF16 quant) and never ride along with the quant someone actually
        # pulls. Collected separately and attached to every group below, so
        # the lane worker can fetch the projector with any quant.
        _b = os.path.basename(path).lower()
        if _b.startswith("mmproj") or "-mmproj-" in _b or ".mmproj." in _b:
            mmprojs.append(path)
            continue
        # lfs.size is the real object size. The plain `size` field is used as a
        # fallback only. This ordering has NOT been verified against a live
        # response from this container - Hugging Face was unreachable from here
        # when this was written - so the sanity check below exists to catch the
        # case where both fields turn out to be pointer sizes rather than
        # letting a 40 GB model be reported as 134 bytes and recommended.
        lfs = e.get("lfs") or {}
        size = lfs.get("size") or e.get("size") or 0
        q = hf_quant_of(path)
        if not q:
            unnamed += 1
            q = os.path.basename(path)[:-5]
        g = groups.get(q)
        if not g:
            g = groups[q] = {"quant": q, "bytes": 0, "files": [], "parts": 1,
                             "named": bool(HF_QTOK_RE.match(q))}

        g["bytes"] += int(size or 0)
        g["files"].append(path)
        g["parts"] = max(g["parts"], hf_split_parts(path))

    if unnamed:
        warn.append("%d GGUF file(s) here do not follow the usual quantisation naming, "
                    "so they are listed under their filename. Ollama matches the tag "
                    "against the filename, so those still pull correctly." % unnamed)

    out = list(groups.values())
    if mmprojs:
        # Preference order: plain F16, then BF16, then whatever else exists.
        def _mmrank(pth):
            b = os.path.basename(pth).lower()
            if "bf16" in b:
                return 1
            if "f16" in b:
                return 0
            return 2
        mmprojs.sort(key=_mmrank)
        for g in out:
            g["mmproj_files"] = list(mmprojs)
    suspect = 0
    for g in out:
        g["files"].sort()
        g["nfiles"] = len(g["files"])
        # No GGUF worth pulling is under a megabyte. If one reads that small,
        # the size field is lying rather than the model being tiny, and the
        # right response is to say the size is unknown - not to draw a green
        # ONE CARD verdict off a number that is wrong by six orders of
        # magnitude.
        g["suspect"] = (g["bytes"] or 0) < 1048576
        if g["suspect"]:
            suspect += 1
    if suspect:
        warn.append("Hugging Face reported an implausibly small size for %d of these "
                    "quantisations, so their sizes and fit verdicts are shown as unknown. "
                    "The pull itself is unaffected - Ollama fetches the real file either "
                    "way, and the Pulls tab will show its true size once it starts." % suspect)
    out.sort(key=lambda g: g["bytes"])
    return out, None, warn


def hf_annotate(quants, ctx=None, reserve=None):
    """Attach a fit verdict to each quantisation, and mark the one to take.

    The verdict is the same fit_verdict() the Models tab uses, fed the real byte
    count from Hugging Face rather than an estimate from the parameter count.
    That is the point of this whole panel: knowing before a 40 GB download
    whether the thing lands on one card, spans both, or quietly spills into host
    RAM and runs at walking pace.
    """
    cards = detected_cards()
    for g in quants:
        gib = (g["bytes"] or 0) / GIB
        g["gib"] = round(gib, 2)
        if g.get("suspect"):
            g["fit"] = {"verdict": "unknown",
                        "why": "Hugging Face did not report a believable size for this file, "
                               "so there is nothing honest to compute a verdict from.",
                        "need_gib": None, "kv_gib": None}
            continue
        g["fit"] = fit_verdict(gib, cards, ctx=ctx, kv_bytes=1, reserve_gib=reserve)
    # Best = best verdict, then the LARGEST file at that verdict, because at
    # equal placement a bigger file is a less lossy quantisation. Same rule as
    # _best_tag() applies to the Ollama library, kept identical on purpose.
    best, best_key = None, None
    for g in quants:
        if g.get("suspect"):
            continue
        r = VERDICT_RANK.get(g["fit"].get("verdict"), 9)
        if r >= 3:                      # too_big / unknown is never a recommendation
            continue
        key = (r, -(g["bytes"] or 0))
        if best_key is None or key < best_key:
            best, best_key = g, key
    for g in quants:
        g["recommended"] = (g is best)
    return quants


def hf_resolve(text, ctx=None, reserve=None):
    repo, quant = hf_parse(text)
    if not repo:
        return None, ("Could not find a Hugging Face repo in that. Paste the model page URL, "
                      "a link to a specific .gguf file, or just owner/name.")
    meta, merr = _hf_get("/api/models/" + repo)
    quants, qerr, warn = hf_quants(repo)
    err = qerr or merr
    if quants is None and err:
        return {"repo": repo, "url": HF_HOST + "/" + repo, "error": err,
                "quants": [], "warnings": [], "meta": None, "picked": quant}, None
    quants = hf_annotate(quants or [], ctx=ctx, reserve=reserve)

    if not quants:
        warn.insert(0, "This repo contains no GGUF files. Ollama can only load GGUF, so it "
                       "cannot pull this one. Look for a community GGUF conversion — usually "
                       "the same model name with -GGUF on the end.")
    if any(g["parts"] > 1 for g in quants):
        warn.append("⚠ Some quantisations here are split across multiple .gguf shards. "
                    "Whether Ollama's Hugging Face path reassembles a split model has NOT "
                    "been verified on this box. If one of those pulls errors out, that is "
                    "the likely reason, and the single-file quantisations are unaffected.")

    m = meta or {}
    info = {
        "id": m.get("id") or repo,
        "gated": bool(m.get("gated")),
        "private": bool(m.get("private")),
        "downloads": m.get("downloads"),
        "likes": m.get("likes"),
        "modified": m.get("lastModified"),
        "pipeline": m.get("pipeline_tag"),
        "license": next((t.split(":", 1)[1] for t in (m.get("tags") or [])
                         if isinstance(t, str) and t.startswith("license:")), None),
    }
    # If a specific quant was pasted but the repo does not list it, say so
    # rather than silently starting a pull that will 404 twenty seconds later.
    if quant and not any(g["quant"].upper() == quant.upper() for g in quants):
        warn.append("The link named %r but no quantisation by that name is listed in this "
                    "repo. Pick one from the table instead." % quant)
        quant = None
    return {"repo": repo, "url": HF_HOST + "/" + repo, "error": None,
            "meta": info, "quants": quants, "warnings": warn,
            "picked": (quant.upper() if quant else None),
            "tagbase": "hf.co/" + repo}, None


def hf_tag(repo, quant):
    return "hf.co/" + repo + ((":" + quant) if quant else "")


# --- Hugging Face -> the llama lane -----------------------------------------
# Unlike the Ollama path, there is no existing downloader to hand this to:
# Ollama's puller writes into Ollama's blob store, which is the one place these
# files must NOT go. So this is a real downloader. It is deliberately small,
# and it reports into the same PULLS structure so the Pulls tab needs no
# changes at all — one synthetic "layer" per file, which is what the UI already
# knows how to draw.

def lane_tag(repo, quant):
    return LANE_TAG_PREFIX + hf_tag(repo, quant)


def _lane_parse(tag):
    """lane:hf.co/owner/repo:QUANT  ->  (owner/repo, QUANT)"""
    rest = tag[len(LANE_TAG_PREFIX):]
    if rest.startswith("hf.co/"):
        rest = rest[len("hf.co/"):]
    if ":" in rest:
        repo, quant = rest.rsplit(":", 1)
    else:
        repo, quant = rest, ""
    return repo, quant


def _lane_worker(tag):
    with _pull_lock:
        p = PULLS.get(tag)
        if not p:
            return
        p["state"] = "running"; p["started"] = time.time()
        p["status"] = "listing files"; p["last"] = time.time()

    tmpnames = []
    try:
        repo, quant = _lane_parse(tag)
        if not lane_available():
            raise RuntimeError("LANE_MODELS_DIR (%s) is not mounted in this container" % LANE_MODELS_DIR)

        groups, err, _warn = hf_quants(repo)
        if groups is None:
            raise RuntimeError(err or "could not read the repo file list")
        want = None
        for g in groups:
            if str(g["quant"]).upper() == quant.upper():
                want = g; break
        if want is None:
            raise RuntimeError("no quantisation named %r in %s" % (quant, repo))

        files = list(want["files"])
        jobs = [(path, os.path.basename(path)) for path in files]

        # Vision models ship their projector as a separate mmproj sidecar that
        # belongs to no quant group. Pull it alongside whatever quant was
        # chosen — renamed with the model's name (mmproj-F16.gguf is generic;
        # two repos' projectors must be able to share this folder, and the
        # Muse-mmproj cross-pairing incident is why the name says which model
        # it belongs to). Skipped if already on disk from an earlier pull.
        mm_note = None
        mm_dest = None
        mm = (want.get("mmproj_files") or [])
        if mm:
            src = mm[0]
            model_base = repo.split("/")[-1]
            if model_base.lower().endswith("-gguf"):
                model_base = model_base[:-5]
            mm_dest = model_base + "-" + os.path.basename(src)
            if os.path.exists(os.path.join(LANE_MODELS_DIR, mm_dest)):
                mm_note = mm_dest + " already present"
            else:
                jobs.append((src, mm_dest))

        with _pull_lock:
            for _path, key in jobs:
                if key not in p["layers"]:
                    p["order"].append(key)
                p["layers"][key] = {"digest": key, "total": 0, "completed": 0,
                                    "status": "queued"}
            p["status"] = ("%d file(s)" % len(jobs)) + (" · " + mm_note if mm_note else "")

        for path, key in jobs:
            dest = os.path.join(LANE_MODELS_DIR, key)
            # Download to .part and rename only on success. A half-written file
            # sitting under its real name is indistinguishable from a good one,
            # and llama-server would fail on it in a way that looks like a
            # metadata problem rather than a truncated download.
            tmp = dest + ".part"
            tmpnames.append(tmp)
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0

            req = urllib.request.Request(
                HF_HOST + "/" + repo + "/resolve/main/" + urllib.parse.quote(path),
                headers={"User-Agent": "v620-dash/1.0"})
            if have:
                # Resume. An interrupted 38 GB pull should not start over.
                req.add_header("Range", "bytes=%d-" % have)

            with urllib.request.urlopen(req, timeout=60) as r:
                clen = int(r.headers.get("Content-Length") or 0)
                total = have + clen if r.status == 206 else (clen or 0)
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with _pull_lock:
                    p["layers"][key].update({"total": total, "completed": have,
                                             "status": "downloading"})
                    p["total"] = sum(l["total"] for l in p["layers"].values())
                    p["completed"] = sum(l["completed"] for l in p["layers"].values())
                    p["status"] = "downloading " + key
                with open(tmp, mode) as fh:
                    while True:
                        with _pull_lock:
                            if p["cancel"]:
                                p["state"] = "cancelled"; p["status"] = "cancelled"
                                return
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        with _pull_lock:
                            p["layers"][key]["completed"] = have
                            p["completed"] = sum(l["completed"] for l in p["layers"].values())
                            p["last"] = time.time()

            # Verify before it is allowed to take its real name. A GGUF starts
            # with the four bytes "GGUF"; an HTML error page saved as .gguf
            # does not, and that is the failure this catches.
            with open(tmp, "rb") as fh:
                magic = fh.read(4)
            if magic != b"GGUF":
                raise RuntimeError("%s is not a GGUF (starts with %r) — download failed or "
                                   "the repo served an error page" % (key, magic))
            os.replace(tmp, dest)
            tmpnames.remove(tmp)
            with _pull_lock:
                p["layers"][key]["status"] = "done"

        with _pull_lock:
            p["state"] = "done"
            p["status"] = "success"
            first = os.path.basename(files[0]) if files else ""
            p["lane_file"] = first
            p["lane_dir"] = LANE_MODELS_DIR
            if mm_dest:
                # Surfaced so the Add-entry form (and anyone reading the pull
                # log) knows which projector pairs with this model.
                p["lane_mmproj"] = mm_dest
    except Exception as e:
        with _pull_lock:
            if p.get("cancel"):
                p["state"] = "cancelled"; p["status"] = "cancelled"
            else:
                p["state"] = "error"
                p["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        # A cancelled or failed download leaves its .part behind on purpose so
        # a retry resumes instead of starting over. Nothing else reads .part.
        with _pull_lock:
            p["finished"] = time.time()
        _pull_pump()


def _lane_cfg_read(path=None):
    try:
        with open(path or LANE_CONFIG, "r", encoding="utf-8") as fh:
            return fh.read(), None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)


def _lane_api_models(url=None):
    """What llama-swap currently offers, and what is resident right now."""
    try:
        req = urllib.request.Request((url or LANE_URL) + "/v1/models",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
        out = []
        for m in d.get("data", []):
            st = (m.get("status") or {})
            out.append({"id": m.get("id"),
                        "state": st.get("value") if isinstance(st, dict) else None,
                        "aliases": ((m.get("meta") or {}).get("llamaswap") or {}).get("aliases") or []})
        return out, None
    except Exception as e:
        return [], "%s: %s" % (type(e).__name__, e)


def lane_overview(lane=None):
    cfg_path, url = _lane_target(lane)
    files, cfg_err = [], None
    cfg, cfg_err = _lane_cfg_read(cfg_path)
    # Test filenames against the config with FULL-LINE COMMENTS REMOVED.
    # A commented-out block is not a working entry, and counting it would show
    # a model as configured while llama-swap cannot serve it — which is worse
    # than showing nothing, because it stops you adding the real entry.
    cfg_code = None
    if cfg is not None:
        cfg_code = "\n".join(ln for ln in cfg.splitlines()
                              if not ln.lstrip().startswith("#"))
    if lane_available():
        try:
            # First pass: collect every gguf with its size and shard identity.
            # A split GGUF ships as name-00001-of-00003.gguf etc. llama.cpp
            # loads the WHOLE model from shard 1 and discovers the siblings
            # itself, so shard 1 is the only file that belongs in a cmd's -m.
            # The 08-25 version of this listed every shard as its own addable
            # file, which offered the user two options that cannot load.
            raw = []
            for n in sorted(os.listdir(LANE_MODELS_DIR)):
                if not n.lower().endswith(".gguf"):
                    continue
                try:
                    sz = os.path.getsize(os.path.join(LANE_MODELS_DIR, n))
                except Exception:
                    sz = 0
                m = HF_SPLIT_RE.search(n[:-5])
                raw.append({
                    "name": n, "bytes": sz,
                    "part": int(m.group(1)) if m else None,
                    "parts": int(m.group(2)) if m else None,
                    "stem": (n[:-5][:m.start()] if m else None),
                })

            shard_groups = {}
            for r in raw:
                if r["parts"]:
                    shard_groups.setdefault(r["stem"], []).append(r)

            for r in raw:
                n = r["name"]
                is_mm = "mmproj" in n.lower()
                # "Configured" is decided by whether the filename appears in
                # the config text at all. Deliberately a substring test and not
                # a YAML parse: the config is full of load-bearing comments and
                # commented-out blocks, and a parser would disagree with what a
                # human reading the file would say.
                entry = {
                    "name": n, "bytes": r["bytes"], "mmproj": is_mm,
                    "configured": bool(cfg_code and n in cfg_code),
                }
                if r["parts"]:
                    sibs = shard_groups[r["stem"]]
                    present = sorted(s["part"] for s in sibs)
                    complete = present == list(range(1, r["parts"] + 1))
                    if r["part"] == 1:
                        # The addable one. Size shown is the WHOLE model, not
                        # the first file — the fit question is about the total.
                        entry["parts"] = r["parts"]
                        entry["parts_present"] = len(present)
                        entry["group_bytes"] = sum(s["bytes"] for s in sibs)
                        entry["shards_incomplete"] = not complete
                    else:
                        # A continuation shard. Loads only via shard 1; must
                        # never be offered as its own model.
                        first = "%s-%05d-of-%05d.gguf" % (
                            r["stem"], 1, r["parts"])
                        entry["shard_of"] = first
                        # Configured-by-proxy: if shard 1 is in the config the
                        # whole set is in use, and "unused" here would invite
                        # someone to delete a file the model needs.
                        entry["configured"] = bool(cfg_code and first in cfg_code)
                files.append(entry)
        except Exception as e:
            cfg_err = cfg_err or ("%s: %s" % (type(e).__name__, e))
    models, api_err = _lane_api_models(url)
    # A part-file means a download that never finished. Worth surfacing: it is
    # 20 GB of nothing and llama-server cannot open it.
    partials = []
    if lane_available():
        try:
            partials = sorted(n for n in os.listdir(LANE_MODELS_DIR) if n.endswith(".part"))
        except Exception:
            pass
    reg = _lane_registry()
    cur = lane if lane in reg else next(iter(reg))
    return {"available": lane_available(), "dir": LANE_MODELS_DIR,
            "config": cfg_path, "config_readable": cfg is not None,
            "config_error": cfg_err, "url": url, "api_error": api_err,
            "lane": cur,
            "lanes": [{"name": n, "url": v["url"]} for n, v in reg.items()],
            "files": files, "partials": partials, "models": models}


_LANE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# --- where a new lane entry goes -------------------------------------------
# 2026-08-25. The previous version of lane_config_add() APPENDED at end of
# file, reasoning that "comments do not terminate a YAML block mapping, so
# keys added at the end of the file are still inside models:". That was true
# when models: was the last block. It stopped being true the day groups: was
# added after it, and nothing re-checked - so three models added from the Lane
# tab were written INSIDE groups:, where llama-swap does not look for models,
# and simply never appeared. Same class of bug: a write that lands in the wrong YAML block.
#
# The fix is to stop inferring position from "end of file" and name the target
# block explicitly. These helpers are text-based on purpose: a YAML round-trip
# would strip the comments that are most of this file's value.

_LANE_TOP_RE = re.compile(r'^(?![#\s])([A-Za-z_][\w.\-]*|"[^"]+"|\'[^\']+\')\s*:')
_LANE_CHILD_RE = re.compile(r'^  ([A-Za-z_][\w.\- ]*|"[^"]+"|\'[^\']+\')\s*:\s*$')


def _lane_top_blocks(lines):
    """[(name, line_index)] for every top-level key, in file order."""
    out = []
    for i, ln in enumerate(lines):
        m = _LANE_TOP_RE.match(ln)
        if m:
            out.append((m.group(1).strip('"\''), i))
    return out


def _lane_block_of(lines, idx):
    """Name of the top-level block that contains line idx, or None."""
    name = None
    for n, i in _lane_top_blocks(lines):
        if i > idx:
            break
        name = n
    return name


def _lane_models_insert_at(lines):
    """Line index at which to insert a new child of models:, or None.

    Returns the position just after the last real content of the models:
    block - stepping back over trailing blank lines and column-0 comments,
    which introduce the NEXT block rather than closing this one.
    """
    blocks = _lane_top_blocks(lines)
    start = nxt = None
    for j, (name, i) in enumerate(blocks):
        if name == "models":
            start = i
            nxt = blocks[j + 1][1] if j + 1 < len(blocks) else len(lines)
            break
    if start is None:
        return None
    k = nxt
    while k - 1 > start:
        prev = lines[k - 1]
        if not prev.strip() or prev.startswith("#"):
            k -= 1
            continue
        break
    return k


def _lane_model_names(text):
    """Names defined directly under models:. Text-based, no PyYAML needed."""
    lines = text.splitlines()
    names = set()
    for i, ln in enumerate(lines):
        m = _LANE_CHILD_RE.match(ln)
        if m and _lane_block_of(lines, i) == "models":
            names.add(m.group(1).strip('"\''))
    return names


def _lane_stranded_in_groups(text):
    """Names under groups: that carry a cmd: — i.e. model entries in the wrong
    block. llama-swap does not look for models there, so they simply never
    appear; this is the classic wrong-block failure.

    Detecting this is worth more than preventing it, because the file can
    already be in this state before we touch it.
    """
    lines = text.splitlines()
    bad = set()
    for i, ln in enumerate(lines):
        m = _LANE_CHILD_RE.match(ln)
        if not m or _lane_block_of(lines, i) != "groups":
            continue
        for j in range(i + 1, len(lines)):
            nxt = lines[j]
            if not nxt.strip() or nxt.lstrip().startswith("#"):
                continue
            indent = len(nxt) - len(nxt.lstrip())
            if indent <= 2:
                break
            if re.match(r"^\s{3,}cmd\s*:", nxt):
                bad.add(m.group(1).strip('"\''))
                break
    return bad


def lane_config_add(model_id, filename, ctx=131072, top_k=64, temp=1.0,
                    top_p=0.95, mmproj=None, ttl=1800, lane=None):
    """Insert one model entry into llama-swap.yaml's models: block.

    `lane` selects WHICH llama-swap.yaml (see _lane_registry) — the lanes
    share one models directory, so only the config file differs.

    INSERT BY NAME, never append-at-EOF, and never rewrite the whole file.
    That file is mostly comments explaining why each flag is there, and
    regenerating it from a parsed structure would delete all of them - but
    "append at the end" put entries wherever the last block happened to be.
    _lane_models_insert_at() finds the models: block explicitly, and the
    write is refused unless the new name is verifiably a child of it.
    """
    cfg_path, _url = _lane_target(lane)
    if not _LANE_ID_RE.match(model_id or ""):
        return False, "a model name may use letters, digits, . _ - only"
    if not filename or "/" in filename or not filename.lower().endswith(".gguf"):
        return False, "pick a .gguf file"
    if not lane_available():
        return False, "the lane's model directory is not mounted (%s)" % LANE_MODELS_DIR
    # A split GGUF loads ONLY from its first shard: llama.cpp reads
    # name-00001-of-0000N.gguf and discovers the siblings in the same
    # directory itself. Pointing -m at shard 2+ produces a model that cannot
    # load, and the failure surfaces later as a llama-server startup error
    # that looks nothing like "wrong file picked". Refuse it here, by name.
    _sm = HF_SPLIT_RE.search(filename[:-5])
    if _sm and int(_sm.group(1)) != 1:
        first = "%s-%05d-of-%05d.gguf" % (
            filename[:-5][:_sm.start()], 1, int(_sm.group(2)))
        return False, ("%s is shard %d of a %d-part model. llama.cpp loads "
                       "the whole model from shard 1 and finds the rest "
                       "itself — add %s instead."
                       % (filename, int(_sm.group(1)), int(_sm.group(2)), first))
    if _sm:
        # Shard 1 of a set: make sure the siblings are actually all here.
        _stem = filename[:-5][:_sm.start()]
        _total = int(_sm.group(2))
        _missing = [i for i in range(1, _total + 1)
                    if not os.path.exists(os.path.join(
                        LANE_MODELS_DIR,
                        "%s-%05d-of-%05d.gguf" % (_stem, i, _total)))]
        if _missing:
            return False, ("this model is split into %d shards and shard(s) "
                           "%s are not in %s — llama-server would fail at "
                           "load. Finish the pull first."
                           % (_total, ", ".join(map(str, _missing)),
                              LANE_MODELS_DIR))
    if not lane_available() or not os.path.exists(os.path.join(LANE_MODELS_DIR, filename)):
        return False, "that file is not in " + LANE_MODELS_DIR
    if mmproj and (("/" in mmproj) or not mmproj.lower().endswith(".gguf")
                   or not os.path.exists(os.path.join(LANE_MODELS_DIR, mmproj))):
        return False, "that mmproj file is not in " + LANE_MODELS_DIR
    try:
        ctx = max(512, min(int(ctx), 1048576))
        top_k = max(1, min(int(top_k), 200))
        ttl = max(0, min(int(ttl), 86400))
        temp = max(0.0, min(float(temp), 2.0))
        top_p = max(0.01, min(float(top_p), 1.0))
    except Exception:
        return False, "context, top-k and ttl must be numbers"

    cfg, err = _lane_cfg_read(cfg_path)
    if cfg is None:
        return False, "cannot read %s (%s)" % (cfg_path, err)
    if re.search(r'^\s*"?%s"?\s*:' % re.escape(model_id), cfg, re.M):
        return False, "%r is already in the config" % model_id

    # PRE-FLIGHT: is the file ALREADY broken? Refuse to add to a config whose
    # existing entries are stranded — otherwise the user keeps clicking Add,
    # keeps being told it worked, and keeps seeing nothing in the lane.
    stranded = _lane_stranded_in_groups(cfg)
    if stranded:
        return False, (
            "%s is malformed and nothing was added. These are model entries "
            "sitting under groups:, where llama-swap does not look for "
            "models, so they never load: %s. Move them into the models: "
            "block, then add again."
            % (cfg_path, ", ".join(sorted(stranded))))

    block = '\n  # Added from the telemetry dashboard. Flags are DEFAULTS, not measured:\n'
    block += '  # check --top-k and the context against what this model actually wants.\n'
    block += '  "%s":\n    cmd: |\n      /app/llama-server\n' % model_id
    block += '      --port ${PORT} --host 127.0.0.1\n'
    block += '      -ngl 99 -c %d --parallel 1\n' % ctx
    # A split is NOT optional on this chassis. With no -ts, llama.cpp spreads
    # the model over every visible card - which silently puts compute on index
    # 0 (airflow under suspicion). 0,1,1 keeps index 0 empty and guarantees
    # index 2 gets work, which is the only card whose exhaust the BMC's fan
    # curve can actually feel. See the header of llama-swap.yaml.
    block += '      -ts 0,1,1 --main-gpu 1\n'
    block += '      --jinja\n      -ctk q8_0 -ctv q8_0\n'
    block += '      --temp %s --top-p %s --top-k %d\n' % (temp, top_p, top_k)
    block += '      --no-webui\n'
    if mmproj:
        block += '      --mmproj /models/%s\n' % mmproj
    block += '      -m /models/%s\n' % filename
    block += '    ttl: %d\n' % ttl

    lines = cfg.splitlines()
    at = _lane_models_insert_at(lines)
    if at is None:
        return False, ("no top-level `models:` block in %s — refusing to guess "
                       "where the entry goes" % cfg_path)
    newtext = "\n".join(lines[:at] + block.rstrip("\n").splitlines()
                        + lines[at:]) + "\n"

    # STRUCTURAL CHECK, AND IT IS NOT OPTIONAL.
    #
    # The previous version validated only `if PyYAML happens to be importable`,
    # with `except ImportError: pass`. PyYAML is not in this container, so that
    # check silently did nothing while three models were written into groups:
    # and vanished. A safety net that can quietly not run is not a safety net.
    # This one is pure stdlib and always executes.
    before = _lane_model_names(cfg)
    after = _lane_model_names(newtext)
    if model_id not in after:
        return False, ("the new entry did not land under models: — nothing "
                       "written. Check that %s still has a top-level models: "
                       "block." % cfg_path)
    missing = before - after
    if missing:
        return False, ("writing would have lost %s — nothing written"
                       % ", ".join(sorted(missing)))
    # Only fire on newly INTRODUCED breakage. The pre-flight above already
    # refuses a file that arrived broken; blaming this write for a condition
    # it inherited would send the user hunting the wrong thing.
    introduced = _lane_stranded_in_groups(newtext) - _lane_stranded_in_groups(cfg)
    if introduced:
        return False, ("this write would have put %s under groups: — that is "
                       "a wrong-block write. Nothing written."
                       % ", ".join(sorted(introduced)))

    # PyYAML, when present, is a SECOND opinion — never the only one.
    try:
        import yaml as _yaml
        parsed = _yaml.safe_load(newtext)
        if not isinstance(parsed, dict) or model_id not in (parsed.get("models") or {}):
            return False, "the new entry did not parse into models — nothing written"
        old = _yaml.safe_load(cfg) or {}
        lost = set((old.get("models") or {})) - set((parsed.get("models") or {}))
        if lost:
            return False, "writing would have lost %s — nothing written" % ", ".join(sorted(lost))
        old_bad = {g for g, v in (old.get("groups") or {}).items()
                   if isinstance(v, dict) and "cmd" in v}
        new_bad = {g for g, v in (parsed.get("groups") or {}).items()
                   if isinstance(v, dict) and "cmd" in v}
        if new_bad - old_bad:
            return False, ("this write would have put %s under groups: — that "
                           "is a wrong-block write. Nothing written."
                           % ", ".join(sorted(new_bad - old_bad)))
    except ImportError:
        pass
    except Exception as e:
        return False, "the result is not valid YAML (%s) — nothing written" % e

    # Atomic: llama-swap polls this file every 2 s and must never read a
    # half-written config.
    tmp = cfg_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(newtext)
        os.replace(tmp, cfg_path)
    except Exception as e:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return False, "could not write %s: %s" % (cfg_path, e)
    return True, "added — llama-swap picks it up within 2 seconds"


def lane_unload(lane=None):
    """Free the lane's VRAM. The Ollama unload button cannot touch these:
    llama-server does not implement Ollama's keep_alive API."""
    _cfg, url = _lane_target(lane)
    tried = []
    for ep in ("/unload", "/api/unload", "/v1/unload", "/models/unload"):
        for meth in ("POST", "GET"):
            try:
                req = urllib.request.Request(url + ep, method=meth)
                if meth == "POST":
                    req.data = b""
                with urllib.request.urlopen(req, timeout=12) as r:
                    if r.status in (200, 204):
                        return True, "unloaded via %s %s" % (meth, ep)
                    tried.append("%s %s -> %s" % (meth, ep, r.status))
            except Exception as e:
                tried.append("%s %s -> %s" % (meth, ep, type(e).__name__))
    return False, ("no unload endpoint answered on this llama-swap build. "
                   "Use the unload button at " + url + "/ui, or run "
                   "llama-lane-unload.sh force. Tried: " + "; ".join(tried[:4]))


def lane_pull_start(repo, quant):
    if not lane_available():
        return False, ("the lane's model directory is not mounted in this container "
                       "(LANE_MODELS_DIR=%s)" % LANE_MODELS_DIR)
    if not quant:
        return False, "pick a quantisation — the lane needs a specific file, not a repo"
    tag = lane_tag(repo, quant)
    with _pull_lock:
        p = PULLS.get(tag)
        if p and p["state"] in ("queued", "running"):
            return False, "already " + p["state"]
        PULLS[tag] = _pull_new(tag)
        PULL_QUEUE.append(tag)
    _pull_pump()
    return True, "queued"


def ollama_copy(source, destination):
    """ollama cp. Manifest-only: both names point at the same blobs, so giving
    a 55-character hf.co tag a short name costs zero disk."""
    try:
        body = json.dumps({"source": source, "destination": destination}).encode()
        req = urllib.request.Request(OLLAMA_URL + "/api/copy", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return True, "copied"
    except Exception as e:
        code = getattr(e, "code", None)
        if code == 404:
            return False, "Ollama does not have that model. Has the pull finished?"
        return False, "%s: %s" % (type(e).__name__, e)


def installed_models():
    """What Ollama already has, with the details the browser can show."""
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/tags",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4.0) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)
    out = []
    for m in (d.get("models") or []):
        det = m.get("details") or {}
        out.append({
            "name": m.get("name"),
            "size": m.get("size"),
            "family": det.get("family"),
            "params": det.get("parameter_size"),
            "quant": det.get("quantization_level"),
            "modified": m.get("modified_at"),
        })
    out.sort(key=lambda x: (x["name"] or ""))
    return out, None


# --- unloading -------------------------------------------------------------
# Ollama has no "unload" endpoint. What it has is keep_alive: send any request
# with keep_alive 0 and the model is dropped as soon as that request finishes.
# With an empty prompt there is nothing to generate, so it is load-check-drop
# and returns in well under a second.
#
# The one wrinkle is that a generate call against an embedding model is refused
# by Ollama - and the embedding model behind Open WebUI's RAG is exactly the
# kind of thing sitting on a card unnoticed. So a refusal is retried against
# /api/embed rather than reported as a failure.
_EMBED_HINTS = ("does not support generate", "embedding", "not support generate")

def _keepalive_zero(path, payload):
    req = urllib.request.Request(OLLAMA_URL + path,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", "replace")
    # /api/generate streams by default; keep_alive-0 with an empty prompt sends
    # one object, but parse defensively rather than assuming that stays true.
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if isinstance(o, dict) and o.get("error"):
            return str(o["error"])
    return None

def unload_model(name):
    """(ok, message). Drops one model out of VRAM. Weights stay on disk."""
    if not name:
        return False, "no model"
    try:
        err = _keepalive_zero("/api/generate",
                              {"model": name, "prompt": "", "keep_alive": 0})
        if err and any(h in err.lower() for h in _EMBED_HINTS):
            err = _keepalive_zero("/api/embed",
                                  {"model": name, "input": "", "keep_alive": 0})
        if err:
            return False, err
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        if any(h in body.lower() for h in _EMBED_HINTS):
            try:
                err = _keepalive_zero("/api/embed",
                                      {"model": name, "input": "", "keep_alive": 0})
                if err:
                    return False, err
            except Exception as e2:
                return False, "%s: %s" % (type(e2).__name__, e2)
        else:
            return False, "HTTP %s: %s" % (e.code, body or e.reason)
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)
    # The card does not free instantly and /api/ps is cached for a few seconds.
    # Expire both so the panel shows the truth on its next poll instead of
    # redrawing the model it was just asked to remove.
    _ollama.update(next=0.0)
    _ollama_seen.pop(name, None)
    return True, "unloaded"

def _ago(sec):
    if sec is None:
        return "moments"
    sec = float(sec)
    return "%ds" % round(sec) if sec < 60 else "%dm" % round(sec / 60)

def _resident(name):
    """The /api/ps entry for a model, or None. Used to refuse politely."""
    models, _err = read_ollama_ps()
    for m in (models or []):
        if m["name"] == name or m["name"].split(":")[0] == name:
            return m
    return None


# ---------------- API ----------------
@app.route("/api/power")
def api_power():
    with lock:
        s, stats = state["stats"], None
        if state["logging"] and s and s["n"]:
            stats = {"min": s["min"], "max": s["max"], "avg": round(s["sum"]/s["n"],1),
                     "n": s["n"], "elapsed": int(time.time()-s["start"]),
                     "umax": s.get("umax"), "c1max": s.get("c1max"),
                     "c2max": s.get("c2max"), "gjmax": s.get("gjmax"),
                     "fmax": s.get("fmax"), "fmin": s.get("fmin"),
                     "vmax": s.get("vmax"), "gpwmax": s.get("gpwmax")}
        return jsonify({"version": APP_VERSION,
                        "current": state["current"], "logging": state["logging"],
                        "logfile": os.path.basename(state["logfile"]) if state["logfile"] else None,
                        "stats": stats, "poll": POLL, "error": state["error"],
                        "cpu": state["cpu"], "util": state["util"], "fans": state["fans"],
                        "fan": state["fan"], "gpus": state["gpus"],
                        "test": test_public(),
                        "gpumem": state["gpumem"], "gpumem_meta": state["gpumem_meta"],
                        "ram": state["ram"],
                        "ollama_enabled": OLLAMA_ENABLED,
                        "samples": state["samples"][-MAXPOINTS:]})

# ---------------- BMC fan mode: the "oh crap" switch ----------------
# Deliberately INDEPENDENT of gpu-fan-control. This is the control you want
# when the daemon is stopped, wedged, or was never installed - which is
# precisely when you need to cool the box in a hurry.
#
# Fan control on this chassis is BINARY (measured 2026-08-23):
#   0x00 Standard - the BMC's own curve, ~4,900 RPM at idle
#   0x01 Full     - ~23,400 RPM
#   0x02 Optimal  - rejected, rsp=0xcc
# The manual duty command is write-dead here and any duty <=20 LATCHES every
# fan at maximum until `ipmitool mc reset cold`, so nothing below touches it.
FAN_MODE_BYTE = {"standard": "0x00", "full": "0x01"}
FAN_MODE_NAME = {0: "standard", 1: "full"}
FANMODE_HOT_C = float(os.environ.get("FANMODE_HOT_C", "85"))
_fanmode_cache = {"t": 0.0, "mode": None, "err": None}

def read_fan_mode(max_age=2.0):
    """Current BMC fan mode, cached briefly so the UI poll can't hammer IPMI."""
    now = time.time()
    if now - _fanmode_cache["t"] < max_age:
        return _fanmode_cache["mode"], _fanmode_cache["err"]
    mode = err = None
    try:
        out = subprocess.run(["ipmitool", "raw", "0x30", "0x45", "0x00"],
                             capture_output=True, text=True, timeout=12)
        if out.returncode != 0:
            err = (out.stderr or out.stdout or "").strip()[:160]
        else:
            tok = (out.stdout or "").split()
            if not tok:
                err = "empty response from the BMC"
            else:
                mode = FAN_MODE_NAME.get(int(tok[-1], 16), "mode-0x%s" % tok[-1])
    except Exception as e:
        err = ("%s: %s" % (type(e).__name__, e))[:160]
    _fanmode_cache.update({"t": now, "mode": mode, "err": err})
    return mode, err

def set_fan_mode(mode):
    """Set the mode and READ IT BACK.

    A command that exits 0 has not necessarily done anything on this hardware -
    that has been true three separate times (the duty register, the power cap,
    and a threshold write that landed in the wrong field). Never report success
    from an exit code alone.
    """
    try:
        out = subprocess.run(["ipmitool", "raw", "0x30", "0x45", "0x01",
                              FAN_MODE_BYTE[mode]],
                             capture_output=True, text=True, timeout=15)
    except Exception as e:
        return False, None, ("%s: %s" % (type(e).__name__, e))[:160]
    if out.returncode != 0:
        return False, None, (out.stderr or out.stdout or "").strip()[:160]
    time.sleep(1.0)
    _fanmode_cache["t"] = 0.0          # force a fresh read
    got, err = read_fan_mode(max_age=0)
    return (got == mode), got, err

def _fan_rpm_max():
    with lock:
        fans = dict(state["fans"] or {})
    vals = [v for v in fans.values() if isinstance(v, (int, float)) and v > 0]
    return max(vals) if vals else None

@app.route("/api/fanmode", methods=["GET"])
def api_fanmode_get():
    mode, err = read_fan_mode()
    with lock:
        gpus = list(state["gpus"] or [])
    rpm = _fan_rpm_max()
    # Mode says quiet but the wall is pinned: that is the fan-fault latch, and
    # the BMC will keep accepting mode commands while ignoring every one.
    latched = bool(mode == "standard" and rpm is not None and rpm >= 22000)
    return jsonify({"mode": mode, "error": err, "rpm": rpm,
                    "hottest_c": hottest_gpu(gpus),
                    "hot_threshold_c": FANMODE_HOT_C,
                    "latched_suspect": latched})

@app.route("/api/fanmode", methods=["POST"])
def api_fanmode_post():
    body = request.get_json(silent=True) or {}
    mode = str(body.get("mode", "")).strip().lower()
    if mode not in FAN_MODE_BYTE:
        return jsonify({"ok": False,
                        "error": "mode must be 'full' or 'standard'"}), 400

    # Going LOUD never needs a confirmation - it is the panic button.
    # Going QUIET while the cards are hot does, because it hands cooling back
    # to a BMC that cannot see the GPUs at all.
    if mode == "standard" and not body.get("force"):
        with lock:
            gpus = list(state["gpus"] or [])
        hot = hottest_gpu(gpus)
        if hot is not None and hot >= FANMODE_HOT_C:
            return jsonify({
                "ok": False, "needs_force": True, "hottest_c": hot,
                "error": "Hottest junction is %.0f C. Going quiet hands cooling "
                         "back to the BMC, which cannot see GPU temperature at "
                         "all - during one measured run its own sensors moved 1 "
                         "C while a card went from 36 C to 109 C. Quiet anyway?"
                         % hot}), 409

    ok, got, err = set_fan_mode(mode)
    return jsonify({
        "ok": ok, "asked": mode, "mode": got, "error": err,
        "rpm": _fan_rpm_max(),
        "note": "spin-up from quiet to full takes about 40 s, so the RPM "
                "reading will lag the mode change"}), (200 if ok else 500)

@app.route("/api/fan", methods=["GET"])
def api_fan_get():
    with lock:
        st = state["fan"]
    return jsonify({"status": st,
                    "conf_present": os.path.isfile(FAN_CONF),
                    "writable": os.access(FAN_CONF, os.W_OK) if os.path.isfile(FAN_CONF)
                                else os.access(FAN_DIR, os.W_OK),
                    "dir": FAN_DIR, "presets": list(PRESETS)})

@app.route("/api/fan", methods=["POST"])
def api_fan_post():
    body = request.get_json(silent=True) or {}
    updates, err = validate_fan_update(body)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    if not os.path.isdir(FAN_DIR):
        return jsonify({"ok": False,
                        "error": "%s is not mounted - add the gpu-fan-control folder "
                                 "to this container's volumes" % FAN_DIR}), 409
    # A preset sent on its own means "give me this whole curve" - so drop any
    # leftover explicit overrides, which would otherwise outrank it.
    clear = OVERRIDE_KEYS if list(updates) == ["PRESET"] else ()
    try:
        write_fan_conf(updates, clear)
    except Exception as e:
        return jsonify({"ok": False, "error": "write failed: %s" % e}), 500
    with lock:
        iv = int((state["fan"] or {}).get("interval", 8))
    return jsonify({"ok": True, "applied": updates,
                    "cleared": sorted(clear), "takes_effect_in_s": iv})

@app.route("/api/log/start", methods=["POST"])
def api_start():
    fn = _start_log()
    with lock:
        cols = list(state["cols"])
    return jsonify({"ok": True, "logfile": os.path.basename(fn), "cols": cols})

@app.route("/api/log/stop", methods=["POST"])
def api_stop():
    # Stopping the log by hand during a run would leave the test blind, so it
    # ends the run too rather than pretending the run is still supervised.
    with lock:
        running = test["phase"] not in ("idle", "done")
    if running:
        stop_test("logging stopped by hand, so the run stopped too")
    _stop_log()
    return jsonify({"ok": True, "ended_test": running})

@app.route("/api/test", methods=["GET"])
def api_test_get():
    with lock:
        pub = test_public()
    pub["limits"] = {k: {"default": d, "lo": lo, "hi": hi}
                     for k, (d, lo, hi) in TEST_LIMITS.items()}
    pub["models"] = ollama_tags()
    pub["fan_dir"] = os.path.isdir(FAN_DIR)
    return jsonify(pub)

@app.route("/api/test/start", methods=["POST"])
def api_test_start():
    body = request.get_json(silent=True) or {}
    cfg, err = validate_test_config(body)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    ok, err = start_test(cfg)
    if not ok:
        return jsonify({"ok": False, "error": err}), 409
    with lock:
        return jsonify({"ok": True, "test": test_public()})

@app.route("/api/test/stop", methods=["POST"])
def api_test_stop():
    stop_test()
    _stop_log()
    with lock:
        return jsonify({"ok": True, "test": test_public()})

def _safe_log(name):
    # only allow our own telemetry-*.csv files, no path traversal
    # dashes optional so logs written by earlier versions still open
    if not re.fullmatch(r"telemetry-?[0-9]{8}-?[0-9]{6}\.csv", name or ""):
        return None
    p = os.path.join(DATA_DIR, name)
    return p if os.path.isfile(p) else None

def server_tz():
    """How the container is currently keeping time.

    python:3.12-slim ships no /usr/share/zoneinfo, so setting TZ without
    also installing tzdata silently leaves the clock on UTC. Reporting the
    live offset here is the cheapest way to see which of those happened.
    """
    now = datetime.datetime.now().astimezone()
    off = now.strftime("%z")            # e.g. "-0400"
    label = ("UTC%s:%s" % (off[:3], off[3:])) if len(off) == 5 else "UTC"
    return {"name": now.tzname() or "", "label": label,
            "zone": os.environ.get("TZ") or ""}

@app.route("/api/logs")
def api_logs():
    items = []
    try:
        for p in glob.glob(os.path.join(DATA_DIR, "telemetry*.csv")):
            st = os.stat(p)
            items.append({"name": os.path.basename(p),
                          "size": st.st_size, "mtime": int(st.st_mtime)})
    except Exception:
        pass
    items.sort(key=lambda x: x["mtime"], reverse=True)
    active = os.path.basename(state["logfile"]) if state["logfile"] else None
    return jsonify({"logs": items, "active": active, "tz": server_tz()})

@app.route("/api/logs/<name>")
def api_log_data(name):
    p = _safe_log(name)
    if not p:
        return jsonify({"error": "not found"}), 404
    cols, rows = [], []
    try:
        with open(p, newline="") as f:
            rd = csv.reader(f)
            cols = next(rd, [])
            for r in rd:
                rows.append(r)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"name": name, "cols": cols, "rows": rows})


# ---------------- Models + pulls API ----------------
# Anything that reaches _reg_get becomes part of a URL path, so it is checked
# against a character set rather than merely being non-empty. A repo of "../.."
# would otherwise walk out of /v2/library and ask the registry for something
# else entirely.
REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*(/[A-Za-z0-9][A-Za-z0-9._\-]*)?$")

def _safe_ref(s):
    return bool(s) and len(s) <= 128 and ".." not in s and bool(REF_RE.match(s))


@app.route("/api/hw")
def api_hw():
    """What this machine actually has. The Models tab asks this first."""
    cards = detected_cards()
    return jsonify({
        "cards": [{"slot": c["slot"], "vram_total": c["vram_total"],
                   "vram_free": c["vram_free"], "vram_used": c["vram_used"]} for c in cards],
        "total": sum(c["vram_total"] or 0 for c in cards),
        "free":  sum(c["vram_free"] or 0 for c in cards),
        "biggest": (cards[0]["vram_total"] if cards else 0),
        "kv_anchor": {"gib": KV_ANCHOR_GIB, "ctx": KV_ANCHOR_CTX,
                      "weights_gib": KV_ANCHOR_WEIGHTS},
        "ctx_env": os.environ.get("OLLAMA_CONTEXT_LENGTH"),
        "disk": models_dir_state(),
    })

@app.route("/api/catalog")
def api_catalog():
    """The static half: what these models are. No network, always fast."""
    return jsonify({"models": [
        {k: v for k, v in c.items()} for c in CATALOG
    ]})

@app.route("/api/catalog/size")
def api_catalog_size():
    """Authoritative size for one tag, if the registry will tell us.

    `src` is the whole point of this endpoint and the UI colours on it:
      registry  the manifest said so. A fact.
      catalog   the published figure, hardcoded here. An estimate.
      none      neither. Do not pretend to know.
    """
    repo = (request.args.get("repo") or "").strip()
    tag  = (request.args.get("tag") or "latest").strip()
    if not _safe_ref(repo) or not _safe_ref(tag):
        return jsonify({"error": "bad repo or tag"}), 400
    ctx = request.args.get("ctx", type=int)
    reserve = request.args.get("reserve", type=float)

    b, src = catalog_size_bytes(repo, tag)
    if not b:
        return jsonify({"repo": repo, "tag": tag, "src": "none",
                        "error": "no size from the registry and none in the catalog"}), 200

    gib = b / GIB
    sev, why = tag_warning(tag)
    fit = fit_verdict(gib, detected_cards(), ctx=ctx, kv_bytes=1, reserve_gib=reserve)
    return jsonify({"repo": repo, "tag": tag, "src": src, "bytes": b,
                    "gib": round(gib, 2), "warn": sev, "warn_why": why, "fit": fit})

@app.route("/api/catalog/summary")
def api_catalog_summary():
    """The whole grid in one request.

    The old Models tab fired one size request per tag - roughly thirty on first
    paint - and rendered a row for each, which is why it read as a wall. This
    returns one object per FAMILY with the best-fitting tag already chosen, so
    the browser paints a grid immediately and can sort and filter on fit
    without asking again. Per-tag detail is still one call away, but it is now
    something you ask for about a model you have already picked out.
    """
    ctx = request.args.get("ctx", type=int)
    reserve = request.args.get("reserve", type=float)
    cards = detected_cards()
    _prefetch_sizes([(c["repo"], t["tag"]) for c in CATALOG
                     for t in (c.get("tags") or [])])
    fams = [family_summary(c, cards, ctx, reserve) for c in CATALOG]
    tasks = sorted({t for f in fams for t in f["tasks"]})
    return jsonify({"families": fams, "tasks": tasks, "cards": len(cards),
                    "ctx": ctx,
                    "stale": any((f.get("best") or {}).get("src") == "catalog"
                                 for f in fams)})

@app.route("/api/catalog/family")
def api_catalog_family():
    """Every tag of one repo, sized and judged. What the detail sheet opens.

    This used to serve only the 17 hand-written families and 404 on anything
    else. It now serves any of the 234 repos in the library index, because the
    registry is what actually knows the tags and it always did:

      curated repo    the hand-written tag list first (it carries per-tag NOTES
                      the registry cannot know), with `all=1` to widen to every
                      published tag.
      any other repo  straight to the registry, no `all=1` needed -- there is no
                      hand-written list to prefer, so asking for one would just
                      be a 404 with extra steps.

    `meta` merges the same way: curated blurb/vendor/licence/ctx where we have
    them, otherwise the library index's own description. Neither is invented,
    and the sheet renders whichever exists.
    """
    repo = (request.args.get("repo") or "").strip()
    if not _safe_ref(repo):
        return jsonify({"error": "bad repo"}), 400
    c = CATALOG_BY_REPO.get(repo)
    ctx = request.args.get("ctx", type=int)
    reserve = request.args.get("reserve", type=float)
    # An uncurated repo has nothing to widen FROM, so it starts wide.
    want_all = request.args.get("all") in ("1", "true", "yes") or not c
    src = "catalog"
    if want_all:
        names = registry_tags(repo)
        if names:
            src = "registry"
            known = {t["tag"]: t for t in ((c or {}).get("tags") or [])}
            tags = [known.get(n) or {"tag": n} for n in names]
        else:
            tags = list((c or {}).get("tags") or [])
    else:
        tags = list((c or {}).get("tags") or [])

    lib_rec = _library_rec(repo)
    if not tags:
        if lib_rec:
            # The repo is real -- it is in the library index -- but the registry
            # did not answer. Say that, rather than "nothing known about it":
            # the two have completely different fixes.
            return jsonify({"error": "registry.ollama.ai did not answer for %s. The model "
                                     "exists; the size lookup is what failed. Try again in "
                                     "a moment." % repo}), 502
        return jsonify({"error": "nothing known about %s" % repo}), 404

    # A repo can publish a hundred tags, and sizing all of them means a hundred
    # manifest fetches. The prefetch is capped; the rest fall back to whatever
    # the hour-long cache already holds and self-correct on the next open. The
    # sheet's own filter is what makes a hundred rows usable -- this cap is only
    # about not hanging on the way there.
    _prefetch_sizes([(repo, t["tag"]) for t in tags[:60]])
    cards = detected_cards()
    out = [_tag_detail(repo, t, cards, ctx, reserve) for t in tags]
    # Best first, so the answer is the first row rather than somewhere in
    # thirty. Blocked formats sink to the bottom regardless of size.
    out.sort(key=lambda t: (1 if t["warn"] == "bad" else 0,
                            VERDICT_RANK.get(t["fit"].get("verdict"), 9),
                            -(t["gib"] or 0)))
    meta = {k: (c or {}).get(k) for k in
            ("name", "vendor", "license", "ctx", "moe", "blurb", "tasks")}
    if lib_rec:
        meta["name"] = meta["name"] or lib_rec.get("name") or repo
        meta["blurb"] = meta["blurb"] or lib_rec.get("description") or ""
        meta["library"] = {k: lib_rec.get(k) for k in
                           ("pulls", "tag_count", "updated", "updated_rel",
                            "capabilities", "sizes", "url")}
    meta["curated"] = bool(c)
    return jsonify({"repo": repo, "src": src, "tags": out, "meta": meta,
                    # Whether this response IS the full registry list. The
                    # client asks with all=0 for an uncurated repo and still
                    # gets everything, so without this it would offer a
                    # "show every tag" button that changes nothing.
                    "all": bool(want_all),
                    "blocked": sum(1 for t in out if t["warn"] == "bad")})

@app.route("/api/catalog/tags")
def api_catalog_tags():
    """Every tag the registry knows for a repo, so the browser is not limited
    to the handful hardcoded in the catalog."""
    repo = (request.args.get("repo") or "").strip()
    if not _safe_ref(repo):
        return jsonify({"error": "bad repo"}), 400
    tags = registry_tags(repo)
    known = {t["tag"]: t for t in ((CATALOG_BY_REPO.get(repo) or {}).get("tags") or [])}
    out = []
    for t in (tags or list(known)):
        sev, why = tag_warning(t)
        out.append({"tag": t, "warn": sev, "warn_why": why,
                    "note": (known.get(t) or {}).get("note"),
                    "gb": (known.get(t) or {}).get("gb")})
    return jsonify({"repo": repo, "src": "registry" if tags else "catalog", "tags": out})

def _installed_repos():
    """Repo names of everything Ollama already has, for the Installed badge.

    A model's Ollama name is `repo:tag`, sometimes with a `namespace/` in front
    for anything that did not come from the official library. The store's rows
    are keyed on the bare library repo, so both are stripped. This never fails
    the request: if Ollama is unreachable the store still paints, just without
    the badge, which is the right trade for a decoration.
    """
    models, err = installed_models()
    repos, full = set(), []
    for m in (models or []):
        n = (m.get("name") or "").strip()
        if not n:
            continue
        full.append(n)
        base = n.split(":")[0]
        repos.add(base.split("/")[-1])
    return sorted(repos), sorted(full), err


@app.route("/api/library")
def api_library():
    """The whole ollama.com library, from disk. No network, always fast.

    This is the store's first paint and it deliberately carries NO byte sizes
    and NO tag lists: 234 repos x ~15 tags is thousands of manifest requests
    and it would turn opening a tab into a minute of blank page. What it does
    carry is what the index page publishes -- description, capability badges,
    the parameter-size tokens, pull count, tag count, last update -- which is
    enough to search, filter and rank on. Exact sizes and real fit verdicts are
    one click away in the detail sheet, fetched per repo, which is the only
    shape of this that scales.
    """
    with LIB_LOCK:
        lib = LIBRARY
    repos, full, ierr = _installed_repos()
    caps = {}
    for m in lib.get("models") or []:
        for c in m.get("capabilities") or []:
            caps[c] = caps.get(c, 0) + 1
    cards = detected_cards()
    return jsonify({
        "count": lib.get("count", 0),
        "fetched_at": lib.get("fetched_at"),
        "age_hours": lib_age_hours(),
        "source": lib.get("source"),
        "sort": lib.get("sort"),
        "problems": (lib.get("problems") or [])[:25],
        "problems_n": len(lib.get("problems") or []),
        "models": lib.get("models") or [],
        "curated": sorted(CATALOG_BY_REPO.keys()),
        "installed": repos,
        "installed_full": full,
        "installed_error": ierr,
        "caps": caps,
        "cards": [{"slot": c["slot"], "vram_total": c["vram_total"]} for c in cards],
        "biggest_gib": round((cards[0]["vram_total"] or 0) / GIB, 2) if cards else 0,
        "last_sync": {k: LIB_LAST.get(k) for k in ("ok", "msg", "at", "running")},
        "auto_hours": LIB_AUTO_HOURS,
    })


@app.route("/api/library/sync", methods=["POST"])
def api_library_sync():
    """Re-scrape ollama.com/library and replace the cache -- if the parse is sound.

    `force=1` bypasses the guards in lib_write() and lib_health(). It exists
    because the guards are heuristics: if ollama really does prune the library
    to 80 models one day, the refusal is wrong and there has to be a way to say
    so. It is not the default because the far more likely cause of a sudden
    drop is a restyle that broke the regexes, and silently replacing 234 good
    records with 3 bad ones is the failure this whole design is built around.
    """
    force = (request.args.get("force") in ("1", "true", "yes")
             or bool((request.get_json(silent=True) or {}).get("force")))
    ok, msg = lib_sync(force=force)
    LIB_LAST.update({"ok": ok, "msg": msg, "at": time.time()})
    with LIB_LOCK:
        lib = LIBRARY
    return jsonify({"ok": ok, "message": msg, "count": lib.get("count", 0),
                    "fetched_at": lib.get("fetched_at"),
                    "problems_n": len(lib.get("problems") or []),
                    "problems": (lib.get("problems") or [])[:25]}), (200 if ok else 409)

@app.route("/favicon.ico")
def favicon():
    """A tab icon, so the browser stops logging a 404 on every page load.

    Served inline rather than as a file because app.py is bind-mounted on its
    own -- there is no static directory next to it to put one in.
    """
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
           '<rect width="32" height="32" rx="6" fill="#0f1216"/>'
           '<rect x="6" y="9" width="20" height="5" rx="2" fill="#43d17c"/>'
           '<rect x="6" y="18" width="20" height="5" rx="2" fill="#3a7bd5"/>'
           '</svg>')
    return app.response_class(svg, mimetype="image/svg+xml",
                              headers={"Cache-Control": "public, max-age=86400"})

@app.route("/api/models/installed")
def api_models_installed():
    if not OLLAMA_ENABLED:
        return jsonify({"models": [], "error": None, "disabled": True,
                        "can_delete": False, "can_unload": False})
    models, err = installed_models()
    return jsonify({"models": models or [], "error": err,
                    "can_delete": ALLOW_MODEL_DELETE,
                    "can_unload": ALLOW_MODEL_UNLOAD})

@app.route("/api/model/delete", methods=["POST"])
def api_model_delete():
    if not OLLAMA_ENABLED:
        return jsonify({"error": OLLAMA_OFF_MSG}), 400
    if not ALLOW_MODEL_DELETE:
        return jsonify({"error": "model deletion is disabled. Set ALLOW_MODEL_DELETE=1 "
                                 "in the compose if you want it."}), 403
    name = ((request.get_json(silent=True) or {}).get("model") or "").strip()
    if not name:
        return jsonify({"error": "no model"}), 400
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/delete",
                                     data=json.dumps({"model": name}).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="DELETE")
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": "%s: %s" % (type(e).__name__, e)}), 502

@app.route("/api/model/unload", methods=["POST"])
def api_model_unload():
    """Free one model's VRAM now instead of waiting out OLLAMA_KEEP_ALIVE.

    Refuses a model that served a request in the last few seconds unless the
    caller explicitly says to do it anyway. Unloading mid-generation does not
    corrupt anything - Ollama finishes the in-flight request first - but it
    does mean the next message reloads 20 GiB, so the person on the other end
    waits. Worth one confirmation.
    """
    if not OLLAMA_ENABLED:
        return jsonify({"error": OLLAMA_OFF_MSG + " Lane models unload from "
                                 "the Lane tab."}), 400
    if not ALLOW_MODEL_UNLOAD:
        return jsonify({"error": "unloading is disabled. Set ALLOW_MODEL_UNLOAD=1 "
                                 "in the compose if you want it."}), 403
    body = request.get_json(silent=True) or {}
    name = (body.get("model") or "").strip()
    if not name:
        return jsonify({"error": "no model"}), 400
    m = _resident(name)
    if m is None:
        # Not an error worth a red box: the goal state is already true.
        return jsonify({"ok": True, "msg": "not loaded", "noop": True})
    if m.get("active") and not body.get("force"):
        return jsonify({"error": "%s served a request %s ago — it is working right now. "
                                 "Unload anyway?" % (name, _ago(m.get("used_ago"))),
                        "confirm": True, "model": m["name"]}), 409
    ok, msg = unload_model(m["name"])
    return jsonify({"ok": ok, "msg": msg, "freed": m.get("size_vram") or 0,
                    "model": m["name"]}), (200 if ok else 502)

@app.route("/api/model/unload_all", methods=["POST"])
def api_model_unload_all():
    """Every resident model, skipping whatever is mid-request unless forced.

    Reports per model rather than one aggregate ok/failed, because "3 of 4"
    is the interesting answer and an aggregate hides which one held on.
    """
    if not OLLAMA_ENABLED:
        return jsonify({"error": OLLAMA_OFF_MSG + " Lane models unload from "
                                 "the Lane tab."}), 400
    if not ALLOW_MODEL_UNLOAD:
        return jsonify({"error": "unloading is disabled. Set ALLOW_MODEL_UNLOAD=1 "
                                 "in the compose if you want it."}), 403
    force = bool((request.get_json(silent=True) or {}).get("force"))
    models, err = read_ollama_ps()
    if err:
        return jsonify({"error": "Ollama is not answering: %s" % err}), 502
    out, freed, skipped = [], 0, 0
    for m in (models or []):
        if m.get("active") and not force:
            out.append({"model": m["name"], "ok": False, "skipped": True,
                        "msg": "working — served a request %s ago" % _ago(m.get("used_ago"))})
            skipped += 1
            continue
        ok, msg = unload_model(m["name"])
        if ok:
            freed += m.get("size_vram") or 0
        out.append({"model": m["name"], "ok": ok, "msg": msg,
                    "freed": (m.get("size_vram") or 0) if ok else 0})
    return jsonify({"ok": True, "results": out, "freed": freed, "skipped": skipped})

@app.route("/api/pull")
def api_pull_list():
    return jsonify({"pulls": pull_snapshot(),
                    "partials": list_partials(),
                    "disk": models_dir_state(),
                    "ollama": OLLAMA_URL})

@app.route("/api/pull/start", methods=["POST"])
def api_pull_start():
    tag = ((request.get_json(silent=True) or {}).get("model") or "").strip()
    ok, msg = pull_start(tag)
    return jsonify({"ok": ok, "msg": msg, "pulls": pull_snapshot()}), (200 if ok else 400)

@app.route("/api/pull/cancel", methods=["POST"])
def api_pull_cancel():
    tag = ((request.get_json(silent=True) or {}).get("model") or "").strip()
    ok, msg = pull_cancel(tag)
    return jsonify({"ok": ok, "msg": msg, "pulls": pull_snapshot()}), (200 if ok else 400)

@app.route("/api/pull/forget", methods=["POST"])
def api_pull_forget():
    """Drop a finished entry from the list. Touches no files."""
    tag = ((request.get_json(silent=True) or {}).get("model") or "").strip()
    with _pull_lock:
        p = PULLS.get(tag)
        if p and p["state"] in ("done", "error", "cancelled"):
            PULLS.pop(tag, None)
    return jsonify({"ok": True, "pulls": pull_snapshot()})

@app.route("/api/pull/orphans/delete", methods=["POST"])
def api_pull_orphan_delete():
    name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
    with _pull_lock:
        if any(x["state"] == "running" for x in PULLS.values()):
            return jsonify({"error": "a pull is running. Stop it first — the partial you "
                                     "are about to delete may be the one it is writing."}), 409
    ok, msg = delete_partial(name)
    return jsonify({"ok": ok, "msg": msg, "partials": list_partials()}), (200 if ok else 400)


# --- Hugging Face import ---------------------------------------------------
@app.route("/api/hf/resolve", methods=["POST"])
def api_hf_resolve():
    """Paste in -> what this repo actually contains, and what it would do here.

    Read-only. Nothing is downloaded and nothing is written by this call; it is
    safe to hit it as often as you like while you make up your mind.
    """
    b = request.get_json(silent=True) or {}
    raw = (b.get("url") or "").strip()
    if len(raw) > 500:
        return jsonify({"error": "that is too long to be a Hugging Face link"}), 400
    try:
        ctx = int(b.get("ctx") or 0) or None
    except Exception:
        ctx = None
    try:
        reserve = float(b.get("reserve")) if b.get("reserve") not in (None, "") else None
    except Exception:
        reserve = None
    res, err = hf_resolve(raw, ctx=ctx, reserve=reserve)
    if err:
        return jsonify({"error": err}), 400
    res["lane"] = {"available": lane_available(), "dir": LANE_MODELS_DIR}
    return jsonify(res)


@app.route("/api/hf/start", methods=["POST"])
def api_hf_start():
    """Hand the chosen tag to the existing pull manager. That is the entire job.

    There is no separate Hugging Face downloader in this app and there should
    never be one: the pull below is queued, cancellable, resumable and visible
    on the Pulls tab because it IS a normal pull.
    """
    b = request.get_json(silent=True) or {}
    repo  = (b.get("repo") or "").strip()
    quant = (b.get("quant") or "").strip()
    if not HF_REPO_RE.match(repo):
        return jsonify({"ok": False, "msg": "bad repo name"}), 400
    if quant and not HF_QUANT_RE.match(quant):
        return jsonify({"ok": False, "msg": "bad quantisation name"}), 400
    dest = (b.get("dest") or "ollama").strip().lower()
    if dest not in ("ollama", "lane"):
        return jsonify({"ok": False, "msg": "destination must be ollama or lane"}), 400
    if dest == "lane":
        tag = lane_tag(repo, quant)
        ok, msg = lane_pull_start(repo, quant)
    else:
        tag = hf_tag(repo, quant)
        ok, msg = pull_start(tag)
    return jsonify({"ok": ok, "msg": msg, "tag": tag, "dest": dest,
                    "pulls": pull_snapshot()}), (200 if ok else 400)



# ---------------------------------------------------------------------------
# BENCH — concurrency testing as a dashboard module (replaces the CLI script's
# awkward model/level juggling). One run at a time; results land in
# /data/bench-history.jsonl so models are comparable across days.
# Measurement logic is the proven llama-lane-concurrency.py core: barrier-
# synced threads, warmup excluded, distinct prompts, usage-based token counts.
# ---------------------------------------------------------------------------
BENCH_HISTORY = os.path.join(DATA_DIR, "bench-history.jsonl")
BENCH_PROMPTS = [
    "Explain how a refrigerator works in about 150 words.",
    "Write a short story opening about a lighthouse keeper, ~150 words.",
    "Summarize the causes of the French Revolution in ~150 words.",
    "Explain TCP vs UDP to a new programmer in about 150 words.",
    "Describe how photosynthesis works in about 150 words.",
    "Explain compound interest with a concrete example, ~150 words.",
    "Write a product description for a stainless steel water bottle, ~150 words.",
    "Explain why the sky is blue in about 150 words.",
]
BENCH = {"running": False, "model": None, "levels": [], "results": [],
         "note": "", "error": None, "started": None}
_BENCH_LOCK = threading.Lock()


def _bench_one(model, max_tokens, prompt, out, idx):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(LANE_URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; chunks = 0; usage_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except ValueError:
                    continue
                if d.get("usage"):
                    usage_tokens = d["usage"].get("completion_tokens")
                cs = d.get("choices") or []
                if cs and (cs[0].get("delta") or {}).get("content"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.time() - t0
    except Exception as e:
        out[idx] = {"error": str(e)}
        return
    total = time.time() - t0
    tokens = usage_tokens if usage_tokens else chunks
    gen = total - (ttft or 0)
    out[idx] = {"ttft": ttft, "tokens": tokens,
                "tps": tokens / gen if gen > 0 and tokens else 0.0}


def _bench_level(model, n, max_tokens):
    out = [None] * n
    barrier = threading.Barrier(n + 1)

    def worker(i):
        barrier.wait()
        _bench_one(model, max_tokens, BENCH_PROMPTS[i % len(BENCH_PROMPTS)], out, i)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(n)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.time()
    for t in threads:
        t.join()
    wall = time.time() - t0
    good = [r for r in out if r and "error" not in r]
    if not good:
        return None
    tok = sum(r["tokens"] for r in good)
    return {"n": n, "agg": round(tok / wall, 1),
            "user_avg": round(sum(r["tps"] for r in good) / len(good), 1),
            "user_min": round(min(r["tps"] for r in good), 1),
            "ttft_max": round(max(r["ttft"] or 0 for r in good), 2),
            "ok": len(good), "fail": n - len(good)}


def _bench_slots(model):
    for u in ("/upstream/%s/props" % model, "/props"):
        try:
            req = urllib.request.Request(LANE_URL + u,
                                         headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=6) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            if "total_slots" in d:
                return d["total_slots"]
        except Exception:
            continue
    return None


def _bench_worker(model, levels, max_tokens):
    try:
        # warmup: also what swaps the model in (deliberate and visible)
        warm = [None]
        _bench_one(model, 16, "Say OK.", warm, 0)
        if warm[0] and "error" in warm[0]:
            with _BENCH_LOCK:
                BENCH["error"] = "warmup failed: " + warm[0]["error"]
                BENCH["running"] = False
            return
        slots = _bench_slots(model)
        with _BENCH_LOCK:
            if slots:
                capped = [n for n in levels if n <= slots] or [slots]
                if capped != levels:
                    BENCH["note"] = ("capped at server's %d slots — raise "
                                     "--parallel in the lane config for more") % slots
                    BENCH["levels"] = capped
                    levels = capped
                else:
                    BENCH["note"] = "server reports %d slots" % slots
            else:
                BENCH["note"] = "slot count unreadable - trusting your levels"
        for n in levels:
            r = _bench_level(model, n, max_tokens)
            if r is None:
                with _BENCH_LOCK:
                    BENCH["error"] = "all requests failed at N=%d" % n
                break
            with _BENCH_LOCK:
                BENCH["results"].append(r)
            time.sleep(3)
        with _BENCH_LOCK:
            if BENCH["results"]:
                rec = {"ts": time.strftime("%Y-%m-%d %H:%M"), "model": model,
                       "slots": slots,
                       "levels": {str(r["n"]): r["agg"] for r in BENCH["results"]},
                       "user": {str(r["n"]): r["user_avg"] for r in BENCH["results"]}}
                try:
                    with open(BENCH_HISTORY, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except OSError:
                    pass
    finally:
        with _BENCH_LOCK:
            BENCH["running"] = False


def _bench_history(limit=25):
    try:
        with open(BENCH_HISTORY) as f:
            recs = [json.loads(l) for l in f.read().splitlines() if l]
        return recs[-limit:]
    except (OSError, ValueError):
        return []


@app.route("/api/bench")
def api_bench():
    models, err = _lane_api_models()
    with _BENCH_LOCK:
        state = dict(BENCH)
    return jsonify({"bench": state, "models": models, "models_error": err,
                    "history": _bench_history()})


@app.route("/api/bench/start", methods=["POST"])
def api_bench_start():
    b = request.get_json(silent=True) or {}
    model = (b.get("model") or "").strip()
    levels = sorted({int(x) for x in (b.get("levels") or []) if int(x) > 0})
    max_tokens = int(b.get("max_tokens") or 200)
    if not model or not levels:
        return jsonify({"ok": False, "msg": "model and levels required"}), 400
    with _BENCH_LOCK:
        if BENCH["running"]:
            return jsonify({"ok": False, "msg": "a bench is already running"}), 409
        BENCH.update({"running": True, "model": model, "levels": levels,
                      "results": [], "note": "", "error": None,
                      "started": time.strftime("%H:%M:%S")})
    threading.Thread(target=_bench_worker, args=(model, levels, max_tokens),
                     daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/lane")
def api_lane():
    return jsonify(lane_overview(request.args.get("lane")))


@app.route("/api/lane/add", methods=["POST"])
def api_lane_add():
    b = request.get_json(silent=True) or {}
    which = (b.get("lane") or "").strip() or None
    ok, msg = lane_config_add(
        (b.get("id") or "").strip(),
        (b.get("file") or "").strip(),
        ctx=b.get("ctx") or 131072,
        top_k=b.get("top_k") or 64,
        temp=b.get("temp") or 1.0,
        top_p=b.get("top_p") or 0.95,
        mmproj=(b.get("mmproj") or "").strip() or None,
        lane=which,
    )
    return jsonify({"ok": ok, "msg": msg, "lane": lane_overview(which)}), (200 if ok else 400)


@app.route("/api/lane/unload", methods=["POST"])
def api_lane_unload():
    b = request.get_json(silent=True) or {}
    which = (b.get("lane") or "").strip() or None
    ok, msg = lane_unload(which)
    return jsonify({"ok": ok, "msg": msg, "lane": lane_overview(which)}), (200 if ok else 400)


@app.route("/api/hf/alias", methods=["POST"])
def api_hf_alias():
    if not OLLAMA_ENABLED:
        return jsonify({"ok": False, "msg": OLLAMA_OFF_MSG}), 400
    """Give a downloaded hf.co/... tag a short name.

    `hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` is correct and
    unusable. This is `ollama cp`, which writes a second manifest pointing at
    the same blobs — no copy of the weights, no extra disk. The long tag stays
    where it is; delete it from the Models tab if you want it gone.
    """
    b = request.get_json(silent=True) or {}
    src = (b.get("source") or "").strip()
    dst = (b.get("name") or "").strip()
    if not src or any(c.isspace() for c in src) or len(src) > 200:
        return jsonify({"ok": False, "msg": "bad source tag"}), 400
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}$", dst or ""):
        return jsonify({"ok": False, "msg": "a name may only use letters, digits, . _ - : /"}), 400
    if ":" not in dst:
        dst += ":latest"
    ok, msg = ollama_copy(src, dst)
    return jsonify({"ok": ok, "msg": msg, "name": dst}), (200 if ok else 400)



INDEX = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SITE_NAME__ — Telemetry</title>
<style>
:root{--bg:#0f1216;--panel:#171b21;--line:#232a33;--txt:#e6edf3;--mut:#8b98a5;--accent:#3fb6ff;--grn:#2ec26a;--amb:#e0a63b;--red:#e5484d;}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
.wrap{max-width:900px;margin:0 auto;padding:24px}
h1{font-size:18px;font-weight:600;margin:0 0 2px}.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.big{font-size:60px;font-weight:700;letter-spacing:-2px;line-height:1}.big span{font-size:23px;color:var(--mut);font-weight:500}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:14px}
.ttl{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}
.row{display:flex;gap:26px;flex-wrap:wrap;align-items:baseline}
.item{display:flex;flex-direction:column}.item .k{color:var(--mut);font-size:12px}
.item .v{font-size:28px;font-weight:700;line-height:1.15}.item .u{font-size:14px;color:var(--mut);font-weight:500}
.cardname{font-size:12px;color:var(--mut);margin-bottom:5px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{color:var(--mut);font-size:12px}.tile .v{font-size:20px;font-weight:600;margin-top:2px}
button{font:inherit;font-weight:600;border:0;border-radius:9px;padding:11px 18px;cursor:pointer;color:#fff}
.start{background:var(--grn)}.stop{background:#e5484d}.btns{display:flex;gap:10px;align-items:center;margin-top:14px}
.status{color:var(--mut);font-size:13px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#556;margin-right:6px;vertical-align:middle}
.dot.on{background:var(--grn);box-shadow:0 0 8px var(--grn)}
.dot.warn{background:var(--amb);box-shadow:0 0 8px var(--amb)}
#c{width:100%;height:140px;display:block}
.note{color:var(--mut);font-size:12px;margin-top:8px}.err{color:#e5484d;font-size:12px;margin-top:6px;min-height:16px}
.muted{color:var(--mut)}
.hbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select,input[type=number],input[type=text]{font:inherit;background:#0f1216;color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
input[type=number]{width:88px}input[type=text]{min-width:230px}
.ghost{background:#0f1216;color:var(--txt);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font:inherit;font-weight:600;cursor:pointer}
.ghost:hover{border-color:var(--accent)}
.ghost.sel{border-color:var(--accent);background:#12212c;color:var(--accent)}
a.lnk{text-decoration:none;display:inline-block}
.chartwrap{position:relative}
#hc{width:100%;height:280px;display:block;cursor:crosshair}
.tip{position:absolute;pointer-events:none;background:#0b0e12;border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.5;color:var(--txt);display:none;white-space:nowrap;box-shadow:0 4px 14px #0008;z-index:5}
.tip b{color:var(--mut);font-weight:600}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.lg{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--mut);cursor:pointer;user-select:none}
.lg .sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.lg.off{opacity:.35;text-decoration:line-through}
.warn{background:#2a2113;border:1px solid #5a4520;color:#e0a63b;border-radius:8px;padding:9px 12px;font-size:13px;margin-top:10px}
.bad{background:#2a1416;border:1px solid #5a2327;color:#e5484d;border-radius:8px;padding:9px 12px;font-size:13px;margin-top:10px}
.fanrow{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}
.lbl{color:var(--mut);font-size:12px;margin-right:2px}
details summary{cursor:pointer;color:var(--mut);font-size:12px;margin-top:12px}
.duty{font-size:34px;font-weight:700;line-height:1}
.slow{color:var(--red)!important}
/* --- VRAM occupancy bars --- */
.vcard{margin-bottom:18px}.vcard:last-child{margin-bottom:0}
.vhead{display:flex;justify-content:space-between;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.vhead .l{font-size:12px;color:var(--mut)}
.vhead .r{font-size:12px;color:var(--mut)}
.vhead .r b{font-size:19px;font-weight:700;color:var(--txt)}
.vbar{display:flex;height:30px;border-radius:8px;overflow:hidden;background:#0b0e12;border:1px solid var(--line)}
.vseg{display:flex;align-items:center;justify-content:center;overflow:hidden;white-space:nowrap;
      font-size:11px;font-weight:700;color:#08121a;min-width:0;transition:flex-basis .45s ease}
.vseg.oth{background:#3c4653;color:#cfd8e3}
.vfree{flex:1 1 auto;display:flex;align-items:center;justify-content:flex-end;padding-right:10px;
       font-size:11px;color:var(--mut);white-space:nowrap;min-width:0;overflow:hidden}
.vlg{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--mut)}
.vlg .e{display:flex;align-items:center;gap:6px}
.vlg .sw{width:10px;height:10px;border-radius:3px;display:inline-block;flex:0 0 auto}
.vlg b{color:var(--txt);font-weight:600}
.vsub{font-size:11px;color:var(--mut);margin-top:5px}
/* Unload controls. Deliberately quiet - this is a thing you reach for, not a
   thing that should catch your eye every time you look at the panel. */
.ub{background:transparent;border:1px solid var(--line);color:var(--mut);
    border-radius:5px;font-size:10px;padding:1px 6px;cursor:pointer;line-height:1.5}
.ub:hover{border-color:var(--amb);color:var(--amb)}
.ub:disabled{opacity:.45;cursor:default;border-color:var(--line);color:var(--mut)}
.ub.big{font-size:12px;padding:5px 12px}
.ub.arm{border-color:var(--amb);color:var(--amb)}
.vact{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:14px}
.vactmsg{font-size:12px;color:var(--mut)}

.tabs{display:flex;gap:6px;align-items:center;margin:0 0 16px}
.tab{background:var(--panel);border:1px solid var(--line);color:var(--mut);
     padding:8px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
.tab.on{color:var(--txt);border-bottom-color:var(--panel);background:#1c222a}
.tabflag{margin-left:10px;font-size:12px}
.trow{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0}
.fld{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--mut)}
.fld input,.fld select{background:#0f1216;border:1px solid var(--line);color:var(--txt);
     border-radius:6px;padding:6px 8px;font-size:14px;min-width:120px}
.chk{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--mut)}
.ph{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13px}
.ph .nm{width:90px;color:var(--mut)}
.ph .tr{flex:1;height:12px;background:#0f1216;border:1px solid var(--line);
     border-radius:6px;overflow:hidden}
.ph .fl{height:100%;background:var(--accent);transition:width .5s linear}
.ph.done .fl{background:var(--grn)}
.ph.now .nm{color:var(--txt);font-weight:600}
.ph .tm{width:74px;text-align:right;color:var(--mut);font-variant-numeric:tabular-nums}
.vd{padding:14px;border-radius:8px;font-size:14px;line-height:1.55}
.vd.pass{background:rgba(46,194,106,.10);border:1px solid rgba(46,194,106,.45)}
.vd.fail{background:rgba(229,72,77,.10);border:1px solid rgba(229,72,77,.45)}
.vd h4{margin:0 0 6px;font-size:16px}
.nlog{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.7;color:var(--mut)}
.nlog b{color:var(--txt)}
.pill{display:inline-block;border:1px solid var(--line);border-radius:20px;padding:1px 9px;
      font-size:11px;color:var(--mut);margin-left:6px}
.pill.ok{border-color:#2a5f42;color:var(--grn)}
.pill.guess{border-color:#5a4520;color:var(--amb)}
.pill.live{border-color:#1d5c38;color:var(--grn)}
.pill.live::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;
      background:var(--grn);margin-right:5px;vertical-align:middle;animation:lp 1.6s infinite}
@keyframes lp{0%,100%{opacity:1}50%{opacity:.25}}
.mt{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px}
.mt th{text-align:left;color:var(--mut);font-size:11px;text-transform:uppercase;
     letter-spacing:.5px;font-weight:600;padding:0 10px 6px 0;border-bottom:1px solid var(--line)}
.mt td{padding:8px 10px 8px 0;border-bottom:1px solid var(--line);vertical-align:top}
.mt tr:last-child td{border-bottom:none}
.mt td.tg{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--txt);word-break:break-all}
.mt td.tg .note{word-break:normal;overflow-wrap:anywhere}
.mt td.nw{white-space:nowrap}
.mt button{padding:4px 10px;font-size:12px}
.pillrow{display:flex;gap:6px;flex-wrap:wrap}
/* --- model browser --- */
.mtools{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
.mtools input[type=search],.mtools select{background:#0f1216;border:1px solid var(--line);
  color:var(--txt);border-radius:8px;padding:8px 10px;font:inherit}
.mtools input[type=search]{min-width:240px;flex:1 1 240px}
.chip{background:#0f1216;border:1px solid var(--line);color:var(--mut);border-radius:20px;
  padding:3px 11px;font-size:12px;cursor:pointer;user-select:none;font:inherit;line-height:1.6}
.chip:hover{border-color:var(--accent)}
.chip.on{border-color:var(--accent);color:var(--accent);background:#12212c}
.chip.flat{cursor:default}
.chip.flat:hover{border-color:var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.mcard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;
  display:flex;flex-direction:column;gap:8px;cursor:pointer;transition:border-color .12s}
.mcard:hover{border-color:var(--accent)}
.mcard .hd{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}
.mcard .nm{font-size:16px;font-weight:700;line-height:1.2}
.mcard .vn{font-size:12px;color:var(--mut)}
.mcard .bl{font-size:12px;color:var(--mut);line-height:1.5;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.mcard .bt{background:#0f1216;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.mcard .bt .tg{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:var(--txt);
  word-break:break-all}
.mcard .bt .sz{font-size:19px;font-weight:700;margin-top:2px}
.mcard .bt .sz span{font-size:12px;color:var(--mut);font-weight:500}
.mcard .ft{display:flex;justify-content:space-between;align-items:center;gap:10px;
  font-size:11px;color:var(--mut);margin-top:auto;padding-top:2px}
.fitb{border-radius:20px;padding:3px 10px;font-size:10.5px;font-weight:700;letter-spacing:.4px;
  white-space:nowrap;border:1px solid currentColor;flex:0 0 auto}
.fitb.v0{color:var(--grn)}.fitb.v1{color:var(--amb)}.fitb.v2{color:var(--amb)}
.fitb.v3{color:var(--red)}.fitb.v4{color:var(--mut)}
.mcard.v3{opacity:.62}
.mempty{color:var(--mut);font-size:13px;padding:18px 2px}
.chip.cap{border-color:#2b3d55;color:#7fb2e5}
.chip.sz{font-family:ui-monospace,Menlo,Consolas,monospace}
.chip.more{color:var(--mut);cursor:default}
.mcard .hd .rt{display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex:0 0 auto}
.mcard .hd .rt2{display:flex;gap:4px}
.instb{border-radius:20px;padding:2px 9px;font-size:10px;font-weight:700;letter-spacing:.4px;
  white-space:nowrap;border:1px solid var(--grn);color:var(--grn)}
.curb{border-radius:20px;padding:2px 9px;font-size:10px;font-weight:700;letter-spacing:.4px;
  white-space:nowrap;border:1px solid var(--line);color:var(--mut)}
.mcard .ft .stat{display:flex;gap:10px;flex-wrap:wrap}
.lk{color:var(--accent);text-decoration:none}
.lk:hover{text-decoration:underline}
.sheet .ftools{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:14px}
.sheet .ftools input[type=search]{background:#0f1216;border:1px solid var(--line);
  color:var(--txt);border-radius:8px;padding:6px 10px;font:inherit;font-size:13px;
  min-width:180px;flex:1 1 180px}
/* --- detail sheet --- */
.ovl{position:fixed;inset:0;background:#000a;backdrop-filter:blur(2px);z-index:40;
  display:flex;align-items:flex-start;justify-content:center;padding:40px 16px;overflow:auto}
.ovl[hidden]{display:none}
.sheet{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  max-width:860px;width:100%;padding:20px 22px;box-shadow:0 18px 50px #000b}
.sheet .sh{display:flex;justify-content:space-between;align-items:flex-start;gap:14px}
.sheet .sh .nm{font-size:22px;font-weight:700;line-height:1.15}
.sheet .x{background:transparent;border:1px solid var(--line);color:var(--mut);border-radius:8px;
  padding:4px 11px;font:inherit;font-size:16px;cursor:pointer;line-height:1.2}
.sheet .x:hover{border-color:var(--red);color:var(--red)}
</style></head><body><div class="wrap">
<h1>__SITE_NAME__ — Telemetry</h1><div class="sub">Power + CPU load/temps + per-card GPU temps, VRAM and board power + per-fan RPM · redundant PSU ceiling on 120V = 800&nbsp;W
<span class="pill" id="ver" title="Which build of app.py this container is actually running. If this isn't the version in the file you just copied, the container is running a different file.">__APP_VERSION__</span></div>

<div class="tabs">
  <button class="tab on" id="tab-live"   onclick="showTab('live')">Live</button>
  <button class="tab"    id="tab-models" onclick="showTab('models')">Models</button>
  <button class="tab"    id="tab-pulls"  onclick="showTab('pulls')">Pulls</button>
  <button class="tab"    id="tab-hf"     onclick="showTab('hf')">HF import</button>
  <button class="tab"    id="tab-lane"   onclick="showTab('lane')">Lane</button>
  <button class="tab"    id="tab-bench"  onclick="showTab('bench')">Bench</button>
  <button class="tab"    id="tab-scripts" onclick="showTab('scripts')">Scripts</button>
__TEST_TAB__
  <span class="tabflag" id="tabflag"></span>
  <span class="tabflag" id="pullflag" style="color:var(--accent)"></span>
</div>

<div id="pane-live">

<div class="panel">
  <div class="ttl">System power (IPMI DCMI)</div>
  <div class="big"><span id="cur">—</span><span> W</span></div>
  <div class="btns">
    <button class="start" id="startBtn" onclick="startLog()">▶ Start logging</button>
    <button class="stop"  id="stopBtn"  onclick="stopLog()" style="display:none">■ Stop logging</button>
    <span class="status"><span class="dot" id="dot"></span><span id="stat">not logging</span></span>
  </div>
  <div class="err" id="err"></div>
</div>

<div class="panel">
  <div class="ttl">CPU — load &amp; temperature</div>
  <div class="row" id="cpubox"><span class="muted">reading…</span></div>
</div>

<div class="panel">
  <div class="ttl">GPU temperature (AMD / amdgpu)</div>
  <div id="gpubox"><span class="muted">No AMD GPU detected yet (V620 not installed / not bound to amdgpu).</span></div>
  <div class="note">Passive V620: watch <b>junction</b> (hotspot). Green &lt;85&deg;C · amber 85–100 · red &gt;100 (throttles ~110). Cards are labelled by PCI slot, which stays put across reboots.</div>
</div>

<div class="panel">
  <div class="ttl">GPU memory — what's resident right now</div>
  <div id="vrambox"><span class="muted">reading…</span></div>
  <div class="vact" id="vramact" style="display:none">
    <button class="ub big" id="unloadAllBtn" onclick="unloadAll(false)">Free all VRAM</button>
    <span class="vactmsg" id="vramactmsg"></span>
  </div>
  <div class="note" id="vramnote"></div>
</div>

<!-- System RAM: ONE bar, one line. Deliberately not the VRAM panel's density —
     RAM matters now (Flash-Next parks its ~50 GB n-gram table there) but it is
     one pool, not three cards with attribution problems. Keep it quiet. -->
<div class="panel">
  <div class="ttl">System RAM</div>
  <div id="rambox"><span class="muted">reading…</span></div>
</div>

<div class="panel">
  <div class="ttl">GPU ECC — trade error correction for 2 GiB/card</div>
  <div id="eccbox"><span class="muted">reading…</span></div>
  <div style="margin-top:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <button class="ub" id="eccOffBtn" onclick="eccStage('off')">Stage ECC OFF (30→32 GiB)</button>
    <button class="ub" id="eccOnBtn" onclick="eccStage('on')">Stage ECC ON (default)</button>
    <span id="eccmsg" class="vactmsg"></span>
  </div>
  <details style="margin-top:8px"><summary class="muted" style="cursor:pointer;font-size:12px">the guide — read once before first use</summary>
  <div class="note" style="margin-top:6px">
    <b>What this is.</b> Each V620 reserves ~2 GiB of VRAM for ECC (error
    correction). Disabling it via the kernel parameter
    <code>amdgpu.ras_enable=0</code> reclaims the reserve: <b>30 → 32 GiB per
    card, +4 GiB total</b> on this box.<br><br>
    <b>What the buttons do — and don't.</b> They only <i>stage</i> the kernel
    option. The mechanism is detected per host: <b>TrueNAS</b> boxes go through
    middleware (<code>midclt</code>), <b>Ubuntu/Debian</b> boxes get
    <code>GRUB_CMDLINE_LINUX_DEFAULT</code> in <code>/etc/default/grub</code>
    rewritten and <code>update-grub</code> run. Either way only this one token
    is touched, every other kernel option is preserved, the file is backed up
    to <code>.bak.ecc</code> first, and the change is verified by reading it
    back — a zero exit code is a claim, the file is the evidence.
    <b>Nothing reboots automatically.</b> You reboot deliberately, with the
    GPUs idle — check the VRAM panel above first.<br><br>
    <b>If the panel says it cannot stage,</b> do it by hand on the host:<br>
    <code>sudo sed -i 's/^\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 amdgpu.ras_enable=0"/' /etc/default/grub</code><br>
    <code>sudo update-grub &amp;&amp; sudo reboot</code><br><br>
    <b>The dance:</b> stage OFF → reboot → (historically the reserve releases
    on the <i>second</i> boot; the phase line above tells you if another cycle
    is needed) → reboot again if asked → this panel shows 32.0 GiB per card
    and "full VRAM reclaimed". Re-enabling is the same in reverse.<br><br>
    <b>Should ECC be off?</b> For inference serving: reasonable — a rare
    flipped bit in a weight changes one activation in one token, and 4 GiB is
    real capacity. <b>For training runs: turn ECC back ON</b> — a bit flip
    that lands in an optimizer state or gradient silently corrupts every step
    after it, and you will not find out until the loss curve does something
    unexplainable. That asymmetry is the whole policy: serving box OFF,
    training days ON.<br><br>
    <b>Falsifier for "it worked":</b> the cards read ~32.0 GiB in
    <code>mem_info_vram_total</code>. Anything else = not applied yet,
    whatever the config says.
  </div></details>
</div>

<div class="panel">
  <div class="ttl">Fan control (gpu-fan-control)</div>
  <div id="fanmode"><span class="muted">reading…</span></div>
  <div id="fanctl"><span class="muted">reading…</span></div>
</div>

<div class="panel">
  <div class="ttl">Chassis fans (IPMI · RPM)</div>
  <div class="row" id="fanbox"><span class="muted">reading…</span></div>
  <div class="note">The <b>slowest</b> fan is the diagnostic one — it's the one that trips the BMC's Lower Critical threshold and triggers a 100% fan-failure ramp. It's shown in red.</div>
</div>

<div class="tiles" id="tiles" style="display:none">
  <div class="tile"><div class="k">Power peak</div><div class="v" id="mx">—</div></div>
  <div class="tile"><div class="k">CPU util peak</div><div class="v" id="um">—</div></div>
  <div class="tile"><div class="k">CPU temp peak</div><div class="v" id="cm">—</div></div>
  <div class="tile"><div class="k">GPU junction peak</div><div class="v" id="gm">—</div></div>
  <div class="tile"><div class="k">GPU VRAM peak</div><div class="v" id="vm">—</div></div>
  <div class="tile"><div class="k">GPU board power peak</div><div class="v" id="gw">—</div></div>
  <div class="tile"><div class="k">Fan peak</div><div class="v" id="fm">—</div></div>
  <div class="tile"><div class="k">Fan floor (min)</div><div class="v" id="fn">—</div></div>
  <div class="tile"><div class="k">Power min</div><div class="v" id="mn">—</div></div>
  <div class="tile"><div class="k">Power avg</div><div class="v" id="av">—</div></div>
  <div class="tile"><div class="k">Samples</div><div class="v" id="ns">—</div></div>
  <div class="tile"><div class="k">Elapsed</div><div class="v" id="el">—</div></div>
</div>

<div class="panel"><canvas id="c" width="820" height="140"></canvas>
  <div class="note">Live power sparkline (DCMI — under-reads vs true PSU AC; use the BMC <b>PMBus</b> tab for absolute watts).</div>
</div>

<div class="panel">
  <div class="ttl">History — past logs</div>
  <div class="hbar">
    <select id="logsel"></select>
    <select id="metric"></select>
    <button class="ghost" onclick="loadLogs()">↻ Refresh</button>
    <a id="dl" class="ghost lnk" href="#" download>⬇ CSV</a>
  </div>
  <div class="chartwrap">
    <canvas id="hc" width="820" height="280"></canvas>
    <div id="tip" class="tip"></div>
  </div>
  <div id="legend" class="legend"></div>
  <div id="hsum" class="note"></div>
</div>
</div>

</div><!-- /pane-live -->

<div id="pane-models" style="display:none">

<div class="panel">
  <div class="ttl">This machine</div>
  <div class="note" style="margin-top:0">
    Read from amdgpu, not from a config file — the same counters the VRAM panel on the Live
    tab uses. Pull a card out and every verdict below changes with it.
  </div>
  <div class="row" id="hwrow" style="margin-top:10px"></div>
  <div class="trow" style="margin-top:12px">
    <label class="fld">Context to plan for
      <select id="mctx" onchange="mctxChanged()">
        <option value="8192">8K</option>
        <option value="32768" selected>32K — what Ollama is set to now</option>
        <option value="65536">64K</option>
        <option value="131072">128K</option>
        <option value="262144">256K</option>
      </select>
    </label>
    <label class="fld">KV reserve override (GiB)
      <input id="mreserve" type="number" step="0.1" min="0" placeholder="auto" oninput="mctxChanged()">
    </label>
    <label class="fld">&nbsp;
      <button class="ghost" onclick="loadLibrary()">Reload list</button>
    </label>
  </div>
  <div class="note" id="kvnote"></div>
</div>

<div class="panel">
  <div class="ttl">Size any tag</div>
  <div class="trow" style="margin-top:0">
    <label class="fld">model:tag
      <input id="anytag" placeholder="qwen3.6:35b-a3b-q8_0" style="min-width:280px"
             onkeydown="if(event.key==='Enter')sizeAny()">
    </label>
    <label class="fld">&nbsp;<button class="ghost" onclick="sizeAny()">Check</button></label>
  </div>
  <div id="anyout"></div>
</div>

<div class="panel">
  <div class="ttl">Installed locally</div>
  <div id="instout" class="note" style="margin-top:0">loading…</div>
</div>

<div class="panel">
  <div class="ttl">The Ollama library <span class="pill" id="libcount">…</span></div>
  <div class="note" id="libnote" style="margin-top:0">Loading the library…</div>
  <div class="mtools">
    <input type="search" id="mq" placeholder="Search every model on ollama.com — name, or what it's for…"
           oninput="renderCatalog()" autocomplete="off">
    <select id="msort" onchange="renderCatalog()">
      <option value="popular">Sort: most pulled</option>
      <option value="newest">Sort: recently updated</option>
      <option value="name">Sort: name</option>
      <option value="small">Sort: smallest first</option>
      <option value="big">Sort: largest first</option>
    </select>
    <button class="ghost" id="libsync" onclick="libSync(0)">↻ Sync from ollama.com</button>
  </div>
  <div class="pillrow" id="mflags" style="margin-bottom:8px"></div>
  <div class="pillrow" id="mcaps" style="margin-bottom:12px"></div>
  <div id="mgrid" class="grid"><div class="mempty">Loading the library…</div></div>
  <div class="note" id="mfoot"></div>
</div>

<div class="ovl" id="msheet" hidden onclick="if(event.target===this)closeFamily()">
  <div class="sheet" id="msheetbody"></div>
</div>

<div class="panel">
  <div class="ttl">How the verdicts are worked out</div>
  <div class="note" style="margin-top:0">
    <b>The list</b> is discovered, not written. This app fetches
    <code>ollama.com/library</code>, parses every model card off it and caches the result in
    <code>/data/catalog.json</code>; the Sync button repeats that. Nothing is hardcoded, so a
    model released this morning appears the next time you press it. A handful of models also
    carry hand-written notes in this app — licence, native context, MoE layout, per-tag
    warnings — things the registry does not publish. Those are marked
    <span class="curb">NOTES</span> and the notes are an <i>overlay</i>: they add to the
    discovered entry, they no longer decide what is in the list.<br><br>
    <b>The badge on a card is an estimate.</b> The library index publishes parameter counts,
    not byte sizes, and turning 234 models into real byte sizes would be thousands of manifest
    requests every time you open this tab. So the card converts the published parameter size to
    GiB at q4 using a ratio measured on this machine, rounded up so it errs toward pessimism.
    Open a model and the sheet asks registry.ollama.ai for the actual layer sizes of every tag.
    <b>When the card and the sheet disagree, the sheet is right.</b><br><br>
    <b>Size</b> comes from the model's OCI manifest at registry.ollama.ai — the sum of the
    weights layer, which is very close to what lands in VRAM. A size marked
    <span class="pill ok">registry</span> was fetched. One marked
    <span class="pill guess">catalog</span> is the published figure hardcoded in this app
    because the registry could not be reached, and it may be stale.<br><br>
    <b>KV cache</b> is an estimate, always. The anchor is a real measurement on this box:
    qwen3.6:35b-a3b at Q4 is 20.6&nbsp;GiB of weights and the two cards held 21.9&nbsp;GiB
    resident at 32K with <code>OLLAMA_KV_CACHE_TYPE=q8_0</code> and flash attention on,
    so everything that is not weights came to about 1.3&nbsp;GiB. Context scales that
    linearly, which is safe. Scaling it to a <i>different</i> model is not safe — the
    manifest does not say how many layers or KV heads a model has — so this uses a square
    root of the size ratio as a hedge and labels the result an estimate. If you know
    better, type the number into the override field.<br><br>
    <b>Overhead</b> is a flat 0.8&nbsp;GiB per card for the framebuffer and the driver's own
    allocations, which do not appear in <code>mem_info_vram_total</code>'s arithmetic.<br><br>
    <b>Spans both</b> is not a failure, but it is not free either. Ollama splits a model by
    layer, so with two cards every token crosses PCIe at every boundary. There is no xGMI
    bridge in this machine, so that crossing is PCIe 3.0 and nothing else. Prefer a model
    that fits one card when the choice is close.
  </div>
</div>

</div><!-- /pane-models -->

<div id="pane-scripts" style="display:none">
<div class="panel">
  <div class="ttl">Host scripts</div>
  <div class="sub">Everything in the scripts folder on the host, run ON THE HOST as root.
  One click, output captured; the gpu-guard also runs itself on a timer while this
  container is up, so reboots recover the GPUs with no clicking at all.</div>
  <div id="scr_msg" style="color:var(--red)"></div>
  <div id="scr_list" style="margin-top:.5em">loading…</div>
  <div id="scr_logwrap" style="display:none;margin-top:.8em">
    <div class="ttl" id="scr_logttl"></div>
    <pre id="scr_log" style="white-space:pre-wrap;font-size:.8em;max-height:22em;overflow:auto"></pre>
  </div>
</div>
</div>

<div id="pane-pulls" style="display:none">

<div class="panel" id="pullformpanel">
  <div class="ttl">Download a model</div>
  <div class="trow" style="margin-top:0">
    <label class="fld">model:tag
      <input id="pulltag" placeholder="qwen3.6:35b-a3b-q8_0" style="min-width:300px"
             onkeydown="if(event.key==='Enter')startPull()">
    </label>
    <label class="fld">&nbsp;<button class="start" onclick="startPull()">▼ Pull</button></label>
  </div>
  <div class="err" id="pullerr"></div>
  <div class="note">
    The download runs in a thread <b>inside this container</b>, not in your browser. That is
    the whole reason this tab exists: Ollama cancels a pull when the client that asked for it
    disconnects, which is why a pull started from a browser tab dies with the browser tab, and
    why three of them died at once when TrueNAS signed the web shell out. Close this page, put
    the laptop to sleep, leave the house — the download carries on.<br><br>
    The honest limit is the other half of the same fact: if <b>this</b> container restarts, its
    threads go with it and the pull stops. It will resume from the partial next time, but do
    not redeploy this project mid-download, and remember watchtower can recreate a container
    without asking. One pull runs at a time; anything else you start is queued. That is
    deliberate — three concurrent downloads share one disk and one uplink, so they all finish
    later than they would have one after another.
  </div>
</div>

<div class="panel">
  <div class="ttl">Downloads</div>
  <div id="pullout">none yet</div>
</div>

<div class="panel">
  <div class="ttl">Half-finished downloads on disk</div>
  <div id="partout" class="note" style="margin-top:0">loading…</div>
  <div class="note">
    <b>On disk</b> is the number that tells the truth. Ollama creates a <code>-partial</code>
    file at its full final length up front and writes chunks into it at offsets, so
    <code>ls -l</code> shows a download that is 11% done as a complete 36&nbsp;GB file. What
    this column reports is blocks actually allocated — the same thing <code>du</code> counts.<br><br>
    A partial is not waste as long as you still want the model: starting the same pull again
    resumes from it. It is waste when you have changed your mind, and then it is worth
    reclaiming, because these run to tens of gigabytes each.<br><br>
    One trap worth knowing: <b>Ollama prunes incomplete blobs when it starts.</b> Restarting or
    redeploying the ollama-rocm container destroys every partial here, and resume goes back to
    zero. <code>OLLAMA_NOPRUNE=true</code> prevents that, at the cost of orphans accumulating
    forever with nothing to clean them up — which is what this list is for.
  </div>
</div>

</div><!-- /pane-pulls -->

<div id="pane-hf" style="display:none">

<div class="panel">
  <div class="ttl">Import from Hugging Face</div>
  <div class="trow" style="margin-top:0">
    <label class="fld">Model page URL, .gguf link, or owner/name
      <input id="hfurl" placeholder="https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
             style="min-width:520px" onkeydown="if(event.key==='Enter')hfResolve()">
    </label>
    <label class="fld">&nbsp;<button class="start" onclick="hfResolve()">Look it up</button></label>
    <label class="fld">&nbsp;<button class="ghost" onclick="hfPaste()">Paste</button></label>
  </div>
  <div class="trow" id="hfdestrow" style="display:none;margin-top:2px">
    <label class="fld">Download it to
      <select id="hfdest" style="min-width:300px">
        <option value="ollama">Ollama — chat, vision, aux models</option>
        <option value="lane">llama lane — agentic, llama.cpp on :8090</option>
      </select>
    </label>
    <div class="note" id="hfdestnote" style="margin:0;max-width:420px"></div>
  </div>
  <div class="err" id="hferr"></div>
  <div class="note">
    Paste the address bar from any Hugging Face model page. If you paste a link to one specific
    <code>.gguf</code> file, that exact quantisation is pre-selected for you.<br><br>
    Nothing downloads at this step. This reads the repo's file list and tells you, for every
    quantisation in it, the real byte count and what it would do to <b>these</b> cards — before
    you spend forty gigabytes finding out. Then the download runs through the same manager the
    Pulls tab uses: queued, cancellable, resumable, and it keeps going with this page closed.
  </div>
</div>

<div class="panel" id="hfrepopanel" style="display:none">
  <div class="ttl" id="hfrepottl">Repo</div>
  <div id="hfmeta" class="note" style="margin-top:0"></div>
  <div id="hfwarn"></div>
  <div id="hfquants" style="margin-top:12px"></div>
  <div class="trow" id="hfmanualrow" style="display:none">
    <label class="fld">Not listed? Pull a quantisation by name
      <input id="hfmanual" placeholder="Q4_K_M" style="min-width:180px"
             onkeydown="if(event.key==='Enter')hfManual()">
    </label>
    <label class="fld">&nbsp;<button class="ghost" onclick="hfManual()">Pull that one</button></label>
  </div>
</div>

<div class="panel" id="hfprogpanel" style="display:none">
  <div class="ttl">Downloading from Hugging Face</div>
  <div id="hfout"></div>
  <div class="note">
    This is the same download the Pulls tab shows, because it is a normal Ollama pull — the
    Hugging Face repo is just where the layers come from. Closing this page does not stop it.
    Restarting <b>this container</b> does.
  </div>
</div>

<div class="panel" id="hfaliaspanel" style="display:none">
  <div class="ttl">Give it a name you will actually type</div>
  <div class="trow" style="margin-top:0">
    <label class="fld">Downloaded tag
      <select id="hfaliassrc" style="min-width:420px"></select>
    </label>
    <label class="fld">Short name
      <input id="hfaliasdst" placeholder="qwen3-coder" style="min-width:180px"
             onkeydown="if(event.key==='Enter')hfAlias()">
    </label>
    <label class="fld">&nbsp;<button class="ghost" onclick="hfAlias()">Name it</button></label>
  </div>
  <div class="err" id="hfaliaserr"></div>
  <div class="note">
    <code>hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M</code> is a correct name and an
    unusable one. This runs <code>ollama cp</code>, which writes a second manifest pointing at the
    same blobs — no second copy of the weights, no extra disk, and both names work from then on.
    The long tag stays until you delete it from the Models tab.
  </div>
</div>

<div class="panel">
  <div class="ttl">What the verdicts mean, and what they don't</div>
  <div class="note" style="margin-top:0">
    The sizes here are the actual bytes Hugging Face reports for the files, not an estimate from
    the parameter count, and the verdict is the same arithmetic the Models tab uses against the
    cards this container can currently see. The reserve on top of the weights is the KV cache at
    the context length set on the Models tab — raise the context and a model that fit stops
    fitting, which is why <b>ONE CARD, TIGHT</b> is called out separately from <b>ONE CARD</b>.<br><br>
    What it cannot tell you is whether the model is any good, whether its chat template is right,
    or whether the person who quantised it did so competently. A repo with a broken template
    downloads and loads perfectly and then answers as if it has never seen a conversation.<br><br>
    <b>Gated repos.</b> If Hugging Face wants a licence accepted, accept it on the model page,
    then authorise Ollama by adding <i>its</i> SSH public key to your Hugging Face account. No
    token is stored in this app and none should be — this dashboard is on the LAN with no auth
    in front of it.
  </div>
</div>

</div><!-- /pane-hf -->

<div id="pane-lane" style="display:none">

<div class="panel">
  <div class="ttl" id="lanettl">llama lane</div>
  <!-- One dashboard, several llama-swap lanes (production :8090, rdna2 test
       :8092, flash-next test :8096). Same models directory; different config
       file and port per lane. The buttons pick which lane the whole tab
       (models, files, add, unload) talks to. -->
  <div class="btns" id="lanesel" style="margin:4px 0 8px"></div>
  <div id="lanehdr" class="note" style="margin-top:0">reading…</div>
  <div class="btns">
    <button class="ghost" onclick="laneLoad()">Refresh</button>
    <button class="stop" onclick="laneUnload()" id="laneunloadbtn">Unload model (free VRAM)</button>
  </div>
  <div class="err" id="laneerr"></div>
  <div class="note">
    The Ollama unload button on the Live tab cannot touch these — llama-server does not
    speak Ollama's <code>keep_alive</code> API, so lane models are invisible to it.
    Unloading here frees the VRAM immediately; the next request reloads the model,
    costing one slow first message.
  </div>
</div>

<div class="panel">
  <div class="ttl">Models llama-swap is serving</div>
  <div id="lanemodels">—</div>
</div>

<div class="panel">
  <div class="ttl">Files on disk</div>
  <div id="lanefiles">—</div>
  <div class="note">
    llama-swap has no auto-discovery, and that is deliberate: a bare <code>.gguf</code> does
    not say what flags to run it with. One model wants <code>--top-k 64</code>, another wants
    20; one needs <code>--mmproj</code>, another has no such file. A guessed entry produces a
    model that runs and quietly answers worse. So this lists what is there and lets you
    confirm the flags — the config is re-read within 2 seconds of saving, no restart.
  </div>
</div>

<div class="panel">
  <div class="ttl">Sampler cheat sheet</div>
  <div class="note" style="margin-top:0">
    What each model family actually asks for. Getting these wrong does not break
    anything visibly — it makes answers quietly worse, which is harder to notice
    than a crash. Click <b>use</b> to push a row into the add form below.
  </div>
  <table class="tbl">
    <thead><tr><th>Family</th><th>temp</th><th>top-p</th><th>top-k</th>
      <th>context</th><th>Where this came from</th><th></th></tr></thead>
    <tbody>
      <tr>
        <td class="tg"><b>Qwen3.6</b><div class="note" style="margin:0">35B-A3B, 27B</div></td>
        <td class="nw">1.0</td><td class="nw">0.95</td><td class="nw"><b>20</b></td>
        <td class="nw">131072</td>
        <td class="note" style="margin:0">Qwen thinking-mode numbers. <b>repeat-penalty must be 1.0</b> —
          1.1 punishes the tokens that close a thinking block and end a turn.</td>
        <td class="nw"><button class="ghost" onclick="laneCheat(20,131072)">use</button></td>
      </tr>
      <tr>
        <td class="tg"><b>Muse Glimmer</b><div class="note" style="margin:0">30B, Meta</div></td>
        <td class="nw">1.0</td><td class="nw">0.95</td><td class="nw"><b>64</b></td>
        <td class="nw">131072<div class="note" style="margin:0">max 262144</div></td>
        <td class="note" style="margin:0">Unsloth's own llama-server command for this model.
          No repeat penalty specified — leave it at 1.0 if you set one.</td>
        <td class="nw"><button class="ghost" onclick="laneCheat(64,131072)">use</button></td>
      </tr>
      <tr>
        <td class="tg"><b>Gemma 3 / 4</b><div class="note" style="margin:0">E4B, 4B, 12B, 31B</div></td>
        <td class="nw">1.0</td><td class="nw">0.95</td><td class="nw"><b>64</b></td>
        <td class="nw">131072</td>
        <td class="note" style="margin:0">Unsloth model card. Vision needs an
          <code>mmproj</code> file alongside — without it these are text-only.</td>
        <td class="nw"><button class="ghost" onclick="laneCheat(64,131072)">use</button></td>
      </tr>
      <tr>
        <td class="tg"><b>Any helper role</b><div class="note" style="margin:0">titles, summaries, compaction</div></td>
        <td class="nw">—</td><td class="nw">—</td><td class="nw">as family</td>
        <td class="nw"><b>8192</b></td>
        <td class="note" style="margin:0">Override the model's native context deliberately.
          KV cache sized for 128k to summarise a page title is pure waste —
          about 6 GB of VRAM for nothing.</td>
        <td class="nw"><button class="ghost" onclick="laneCheat(null,8192)">use</button></td>
      </tr>
    </tbody>
  </table>
  <div class="note">
    <b>True for every entry on this lane, regardless of family:</b>
    <code>--jinja</code> (the tool-calling fix — the whole reason this lane exists),
    <code>--parallel 1</code> (concurrency measured 1.1x not 3-4x on MoE, and one slot
    means <code>-c</code> means what it says),
    <code>-ctk q8_0 -ctv q8_0</code> (a trade to afford 128k context — measured ~6%
    slower than f16 elsewhere, so not a free win), and
    <b>no <code>-fa</code></b> (measured slower, but that was on ROCm — unverified on
    the Vulkan build this lane now runs).
  </div>
</div>

<div class="panel" id="laneaddpanel" style="display:none">
  <div class="ttl" id="laneaddttl">Add to the lane</div>
  <div class="trow">
    <label class="fld">Name it<input id="laneaddid" style="min-width:200px"
      placeholder="my-model-q6"></label>
    <label class="fld">Context<input id="laneaddctx" style="min-width:110px" value="131072"></label>
    <label class="fld">top-k<input id="laneaddtopk" style="min-width:80px" value="64"></label>
    <label class="fld">Vision (mmproj)
      <select id="laneaddmm" style="min-width:200px"></select></label>
  </div>
  <div class="btns">
    <button class="start" onclick="laneAdd()">Add it</button>
    <button class="ghost" onclick="document.getElementById('laneaddpanel').style.display='none'">Cancel</button>
  </div>
  <div class="note" id="laneaddnote"></div>
</div>

</div><!-- /pane-lane -->

<div id="pane-bench" style="display:none">
<div class="panel">
  <div class="ttl">concurrency bench — the lane under load</div>
  <div class="note">Heads up: benching a model <b>swaps it in</b> (that is how
    llama-swap works) — whatever is loaded now gets evicted. Currently loaded:
    <span id="benchloaded" class="muted">…</span></div>
  <div class="btns" style="margin-top:8px">
    <label class="fld">Model<select id="benchmodel" style="min-width:220px"></select></label>
    <label class="fld">Levels
      <span id="benchlevels">
        <label><input type="checkbox" value="1" checked>1</label>
        <label><input type="checkbox" value="2" checked>2</label>
        <label><input type="checkbox" value="4" checked>4</label>
        <label><input type="checkbox" value="8">8</label>
        <label><input type="checkbox" value="16">16</label>
      </span></label>
    <label class="fld">Max tokens<input id="benchmaxtok" style="min-width:70px" value="200"></label>
    <button class="start" id="benchgo" onclick="benchStart()">Run bench</button>
  </div>
  <div class="note" id="benchnote"></div>
  <table style="margin-top:10px"><thead><tr>
    <th>N</th><th>agg t/s</th><th>user avg</th><th>user min</th><th>TTFT max</th><th>ok</th>
  </tr></thead><tbody id="benchrows"></tbody></table>
</div>
<div class="panel">
  <div class="ttl">history — every model, same ruler</div>
  <div class="note">aggregate t/s by concurrency, per-user average in parens</div>
  <table style="margin-top:8px"><thead><tr id="benchhisthead"></tr></thead>
  <tbody id="benchhistrows"></tbody></table>
</div>
</div><!-- /pane-bench -->


__TEST_PANE__

<script>
let samples=[];
function fmtEl(s){const m=Math.floor(s/60),ss=s%60;return m+"m "+String(ss).padStart(2,'0')+"s";}
function gcol(t){if(t==null)return 'var(--txt)';if(t>=100)return 'var(--red)';if(t>=85)return 'var(--amb)';return 'var(--grn)';}
function ccol(t){if(t==null)return 'var(--txt)';if(t>=83)return 'var(--red)';if(t>=70)return 'var(--amb)';return 'var(--grn)';}
function ucol(u){if(u==null)return 'var(--txt)';if(u>=85)return 'var(--red)';if(u>=60)return 'var(--amb)';return 'var(--grn)';}
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function itm(k,v,u,col){return v==null?'':`<div class="item"><span class="k">${k}</span><span class="v" style="color:${col}">${v}<span class="u"> ${u}</span></span></div>`;}
function renderCpu(cpu,util){
  const b=document.getElementById('cpubox'); cpu=cpu||{};
  let h='';
  h+=itm('utilization', util, '%', ucol(util));
  h+=itm('CPU1', cpu.CPU1, '°C', ccol(cpu.CPU1));
  h+=itm('CPU2', cpu.CPU2, '°C', ccol(cpu.CPU2));
  if(cpu.System!=null) h+=itm('ambient', cpu.System, '°C', 'var(--txt)');
  else if(cpu.Peripheral!=null) h+=itm('ambient', cpu.Peripheral, '°C', 'var(--txt)');
  b.innerHTML = h || '<span class="muted">reading…</span>';
}
function renderFans(fans){
  const b=document.getElementById('fanbox');
  const keys=fans?Object.keys(fans).sort():[];
  if(!keys.length){b.innerHTML='<span class="muted">No fan sensors reported.</span>';return;}
  const vals=keys.map(k=>fans[k]); const lo=Math.min(...vals);
  b.innerHTML = keys.map(k=>{
    const v=fans[k]; const isLo = (v===lo && keys.length>1);
    const col = isLo?'var(--red)':(v>=8000?'var(--amb)':'var(--txt)');
    return `<div class="item"><span class="k">${esc(k)}${isLo?' · slowest':''}</span><span class="v" style="color:${col}">${v}<span class="u"> RPM</span></span></div>`;
  }).join('');
}
function renderGpus(gpus){
  const box=document.getElementById('gpubox');
  if(!gpus||!gpus.length){box.innerHTML='<span class="muted">No AMD GPU detected yet (V620 not installed / not bound to amdgpu).</span>';return;}
  const hot=Math.max(...gpus.map(g=>g.junction==null?-99:g.junction));
  box.innerHTML = gpus.map((g,i)=>{
    const it=(k,v)=> v==null?'':`<div class="item"><span class="k">${k}</span><span class="v" style="color:${k=='junction'?gcol(v):'var(--txt)'}">${v}<span class="u"> °C</span></span></div>`;
    const isHot = gpus.length>1 && g.junction===hot;
    const name = `GPU ${i} · ${esc(g.slot||g.card||'')}${g.card&&g.slot?' · '+esc(g.card):''}${isHot?' · hottest (drives the curve)':''}`;
    return `<div style="margin-bottom:${i<gpus.length-1?14:0}px"><div class="cardname">${name}</div><div class="row">${it('edge',g.edge)}${it('junction',g.junction)}${it('mem',g.mem)}</div></div>`;
  }).join('');
}

/* ---------------- VRAM occupancy ----------------
   The bar's LENGTH is always amdgpu's mem_info_vram_used. The coloured pieces
   inside it are only labels for that length, and anything we can't name stays
   visible as "other" rather than being folded into free space — a card that is
   80% full for reasons nothing can explain is exactly the thing worth seeing. */
const VPAL=['#3fb6ff','#a06bff','#2ec26a','#e0a63b','#3fd9d0','#f472b6','#8be04e','#fb7185'];
const GIB=1073741824;
function gib(b){return b==null?'—':(b/GIB).toFixed(1);}
function vcol(p){if(p==null)return 'var(--txt)';if(p>=92)return 'var(--red)';if(p>=75)return 'var(--amb)';return 'var(--grn)';}
function srcPill(src,over){
  if(over) return '<span class="pill guess">attribution over-counts — see note</span>';
  if(src==='fdinfo')      return '<span class="pill ok">per-process (/proc)</span>';
  if(src==='pin')         return '<span class="pill ok">Ollama pinned here</span>';
  if(src==='single-card') return '<span class="pill ok">only card</span>';
  if(src==='inferred')    return '<span class="pill guess">placement inferred</span>';
  return '';
}
function renderVram(cards,meta){
  const box=document.getElementById('vrambox'), noteEl=document.getElementById('vramnote');
  cards=cards||[]; meta=meta||{};
  if(!cards.length){
    box.innerHTML='<span class="muted">No AMD GPU reporting memory yet.</span>';
    noteEl.textContent=''; return;
  }
  let h='', ci=0;
  cards.forEach((c,idx)=>{
    const tot=c.vram_total, used=c.vram_used;
    const name=`GPU ${idx} · ${esc(c.slot||c.pci||'')}`;
    if(!tot){
      h+=`<div class="vcard"><div class="vhead"><span class="l">${name}</span></div>`+
         `<div class="muted" style="font-size:12px">This card isn't exposing mem_info_vram_total. `+
         `That's an old amdgpu, or the card has dropped off the bus — check the temperature panel above.</div></div>`;
      return;
    }
    const pct = used==null?null:Math.round(100*used/tot);
    /* Right-hand summary: fullness first because that's the question, then the
       two numbers that explain a full card that isn't doing anything. */
    let right=`<b style="color:${vcol(pct)}">${gib(used)}</b> / ${gib(tot)} GiB`+
              (pct==null?'':` · <b style="color:${vcol(pct)}">${pct}%</b>`);
    if(c.busy!=null)    right+=` · ${c.busy}% busy`;
    if(c.power_w!=null) right+=` · ${c.power_w} W`+(c.cap_w?` / ${c.cap_w} cap`:'');
    if(c.junction!=null)right+=` · <span style="color:${gcol(c.junction)}">${c.junction}°C</span>`;
    h+=`<div class="vcard"><div class="vhead"><span class="l">${name}${srcPill(c.src,c.over)}</span>`+
       `<span class="r">${right}</span></div>`;

    /* Segments are sized against TOTAL, not against used, so two cards side by
       side are directly comparable and a half-empty card looks half empty. */
    const segs=(c.segments||[]).map(s=>({...s,color:VPAL[(ci++)%VPAL.length]}));
    let bar='';
    segs.forEach(s=>{
      const p=100*s.bytes/tot;
      const sp=s.split?` · split across ${s.split.cards} cards, ${gib(s.split.total)} GiB total`:'';
      bar+=`<div class="vseg" style="flex:0 0 ${p.toFixed(3)}%;background:${s.color}" `+
           `title="${esc(s.label)} — ${gib(s.bytes)} GiB${sp}${s.pid?(' · pid '+s.pid):''}">`+
           `${p>=9?esc(s.label):''}</div>`;
    });
    if(c.other>0){
      const p=100*c.other/tot;
      bar+=`<div class="vseg oth" style="flex:0 0 ${p.toFixed(3)}%" `+
           `title="allocated on the card, but nothing here can say by what — ${gib(c.other)} GiB">`+
           `${p>=9?'other':''}</div>`;
    }
    const freeb=Math.max(0,tot-(used||0)), freep=100*freeb/tot;
    bar+=`<div class="vfree" title="${gib(freeb)} GiB free">${freep>=14?(gib(freeb)+' GiB free'):''}</div>`;
    h+=`<div class="vbar">${bar}</div>`;

    /* Legend carries the exact numbers, so a 2% sliver is still readable. */
    let lg=segs.map(s=>{
      let extra='';
      if(s.model && s.model.size && s.model.size_vram && s.model.size_vram < s.model.size*0.98){
        extra=` <span style="color:var(--amb)">· only ${Math.round(100*s.model.size_vram/s.model.size)}% on the GPU</span>`;
      }
      /* One model spread over both cards is two segments with the same name.
         Without this the legend just prints that name twice and you cannot
         tell one 24 GiB model from two 12 GiB ones. */
      if(s.split && s.split.parts){
        const other=s.split.parts.filter(p=>!p.self)
          .map(p=>`${gib(p.bytes)} GiB on ${esc(p.slot)}`).join(', ');
        if(other) extra+=` <span style="opacity:.75">· here; ${other} — ${gib(s.split.total)} GiB in one model</span>`;
      }
      /* A name that came from a size match is a guess, and the panel says so
         rather than letting it read like the digest-based one next to it. */
      if(s.name_src==='size'||s.name_src==='pid')
        extra+=` <span class="pill guess" title="matched by size against Ollama's /api/ps, not by blob digest">by size</span>`;
      /* Who is actually working. This is the whole point of the two activity
         signals: with two models resident, "the card is at 99%" doesn't say
         whose request it is. */
      if(s.gpu_pct!=null&&s.gpu_pct>=1)
        extra+=` <span style="color:var(--grn)">· <b>${s.gpu_pct}%</b> of this card</span>`;
      if(s.active)
        extra+=` <span class="pill live" title="Ollama's keep-alive timer moved, so this model served a request in the last ${Math.round(s.used_ago||0)}s">working</span>`;
      else if(s.used_ago!=null&&s.used_ago<600)
        extra+=` <span style="opacity:.7">· idle ${s.used_ago<60?Math.round(s.used_ago)+'s':Math.round(s.used_ago/60)+'m'}</span>`;
      /* Unload, per model. Only on segments we could actually name — an
         unnamed allocation has no name to send, and ComfyUI does not answer to
         Ollama. On a split model both halves carry the button and both do the
         same thing; unloading twice is a no-op, so that costs nothing. */
      if(meta.can_unload && s.kind==='model')
        extra+=` <button class="ub" data-unload="${esc(s.label)}" title="Drop this model out of VRAM now instead of waiting for its keep-alive to expire. The weights stay on disk; the next request reloads it.">unload</button>`;
      return `<span class="e"><span class="sw" style="background:${s.color}"></span>`+
             `${esc(s.label)} <b>${gib(s.bytes)}</b> GiB${extra}</span>`;
    });
    if(c.other>0) lg.push(`<span class="e"><span class="sw" style="background:#3c4653"></span>other <b>${gib(c.other)}</b> GiB</span>`);
    lg.push(`<span class="e"><span class="sw" style="background:#0b0e12;border:1px solid var(--line)"></span>free <b>${gib(freeb)}</b> GiB</span>`);
    h+=`<div class="vlg">${lg.join('')}</div>`;

    if(c.over>0){
      h+=`<div class="warn">Named allocations add up to ${gib(c.named)} GiB but the driver only reports `+
         `${gib(used)} GiB in use. The attribution is wrong — treat the labels as a hint and the `+
         `${gib(used)} GiB as the fact.</div>`;
    }
    if(c.gtt_used>2*GIB){
      h+=`<div class="warn">${gib(c.gtt_used)} GiB is in GTT — host RAM reached over PCIe, not VRAM. `+
         `A model spilling into GTT hasn't failed to load, it has quietly become several times slower.</div>`;
    }
    h+=`</div>`;
  });
  box.innerHTML=h;

  /* The note explains where the labels came from, because a bar that can't say
     how it knows is a bar you can't act on. */
  let n=[];
  if(meta.fdinfo) n.push('Segments come from <b>/proc/&lt;pid&gt;/fdinfo</b> — the actual per-process, per-card allocation'+(meta.ollama_disabled?'.':', so ComfyUI and anything else show up alongside Ollama.'));
  else n.push('Per-process attribution is off: this container can only see its own PIDs. Add <code>pid: host</code> to the compose file to name every consumer. It grants nothing new — the container is already <code>privileged: true</code>.');
  if(meta.ollama_disabled){
    /* lane-only box: every Ollama-flavoured footnote below is noise here.
       The bars and process names are complete on their own. */
    n.push('This box has no Ollama — lane models are managed from the Lane tab.');
    noteEl.innerHTML=n.join(' ');
    const act0=document.getElementById('vramact');
    if(act0) act0.style.display='none';
    return;
  }
  const nb=meta.named_by||{};
  if(nb.manifest) n.push(`Model names come from each runner’s <code>--model .../blobs/sha256-…</code> argument joined against the manifests on disk — an exact match, so a model <b>split across both cards by <code>OLLAMA_SCHED_SPREAD</code></b> is still named on each card, and two models of similar size can’t be swapped.`);
  if(nb.pid||nb.size) n.push(`Names tagged <b>by size</b> were matched against Ollama’s <code>/api/ps</code> byte count instead${meta.models_dir?', because the runner’s command line carried no blob path':' — mount Ollama’s models directory and set <code>OLLAMA_MODELS_DIR</code> to get the exact join'}. Treat those as strong hints rather than facts.`);
  if(meta.engine_moving) n.push('The per-model percentages are that process’s own share of the card, from <code>drm-engine-*</code> in fdinfo — so with two models resident you can see which one the card is actually busy with, not just that it’s busy.');
  else if(meta.engine_time) n.push('Per-process GPU time is published by the kernel but <b>not moving</b>, which is expected on ROCm: HIP work goes through KFD user-mode queues that bypass amdgpu’s DRM scheduler accounting. The <b>working</b> marker below falls back to Ollama’s keep-alive timer, which advances whenever a model serves a request.');
  n.push('<b>working</b> means Ollama’s keep-alive on that model moved within the last 15 seconds — it served a request just now. It says which <i>model</i> is busy, not which person asked: everything arrives through Open WebUI, so Ollama sees one client for the whole house.');
  if(meta.unnamed_ollama) n.push(`${meta.unnamed_ollama} Ollama allocation${meta.unnamed_ollama>1?'s':''} could not be tied to a model and ${meta.unnamed_ollama>1?'are':'is'} left generic. That is deliberate — the alternative is printing whichever name happened to be nearest in size.`);
  if(meta.ollama_error) n.push(`Ollama at <code>${esc(meta.ollama_url||'')}</code> isn’t answering (<code>${esc(String(meta.ollama_error).slice(0,120))}</code>), so model names are missing. The bars above are still correct — they come from the driver.`);
  else if(!meta.ollama_models) n.push('Ollama is up and holding no models resident.');
  if(meta.pin_unmatched)
    n.push(`<b>OLLAMA_GPU_PCI is set to <code>${esc(meta.pin||'')}</code>, which matches no card here</b>, so no model has been placed — that memory is showing as “other” instead. Fix the value in the compose file (use the slot exactly as it appears in the headings above) rather than removing it.`);
  if(!meta.fdinfo && meta.ollama_models && !meta.pin)
    n.push('With two cards and no <code>OLLAMA_GPU_PCI</code> set, which card a model sits on is <b>inferred</b> from how full each card is. Set that variable to whichever card Ollama’s <code>HIP_VISIBLE_DEVICES</code> selects and the guess goes away.');
  n.push('Bars are drawn against each card’s full VRAM, so two cards are directly comparable.');
  const act=document.getElementById('vramact');
  if(act){
    const any=(cards||[]).some(c=>(c.segments||[]).some(s=>s.kind==='model'));
    act.style.display = (meta.can_unload && any) ? '' : 'none';
  }
  if(meta.can_unload)
    n.push('<b>unload</b> frees a model’s VRAM immediately by sending Ollama a request with <code>keep_alive: 0</code> — the same thing the keep-alive timer does when it expires, just now. Nothing is deleted: the weights stay on disk and the next request that needs the model loads it again, which costs one slow first message. A model that is <b>working</b> asks for confirmation first, because unloading it means whoever is waiting on that reply waits for a reload too.');
  noteEl.innerHTML=n.join(' ');
}

/* ---------------- system RAM — one bar, one line ----------------
   Deliberately NOT the VRAM panel: no per-segment buttons, no footnote essay.
   Segments are each process's PRIVATE memory (RssAnon), page cache is its own
   reclaimable band (mmap'd GGUFs live there — big mappers get a "+N mapped"
   note instead of a double-counted segment), free closes the bar. */
function renderRam(r){
  const box=document.getElementById('rambox');
  if(!box) return;
  if(!r||r.error||!r.total){
    box.innerHTML='<span class="muted">'+(r&&r.error?esc(r.error):'no /proc/meminfo yet')+'</span>';
    return;
  }
  const tot=r.total, pctUsed=Math.round(100*(tot-r.avail)/tot);
  let right=`<b style="color:${vcol(pctUsed)}">${gib(tot-r.avail)}</b> / ${gib(tot)} GiB`+
            ` · <b style="color:${vcol(pctUsed)}">${gib(r.avail)} GiB avail</b>`;
  if(r.swap_total>0) right+=` · swap ${gib(r.swap_total-r.swap_free)}/${gib(r.swap_total)}`;
  let h=`<div class="vcard"><div class="vhead"><span class="l">whole box · private per process; mmap'd model files sit in cache</span><span class="r">${right}</span></div>`;
  let bar='', lg=[], ci=0, shown=0;
  (r.procs||[]).forEach(p=>{
    if(p.anon<1*GIB) return;
    const col=VPAL[(ci++)%VPAL.length], pc=100*p.anon/tot; shown+=p.anon;
    bar+=`<div class="vseg" style="flex:0 0 ${pc.toFixed(3)}%;background:${col}" `+
         `title="${esc(p.comm)} — ${gib(p.anon)} GiB private · pid ${p.pid}">${pc>=9?esc(p.comm):''}</div>`;
    const mapped=p.mapped>=2*GIB?` <span style="opacity:.75">+${gib(p.mapped)} mapped</span>`:'';
    lg.push(`<span class="e"><span class="sw" style="background:${col}"></span>${esc(p.comm)} <b>${gib(p.anon)}</b>${mapped}</span>`);
  });
  const other=Math.max(0,r.used-shown);
  if(other>0.01*tot){
    bar+=`<div class="vseg oth" style="flex:0 0 ${(100*other/tot).toFixed(3)}%" `+
         `title="everything else + kernel — ${gib(other)} GiB">${100*other/tot>=9?'other':''}</div>`;
    lg.push(`<span class="e"><span class="sw" style="background:#3c4653"></span>other <b>${gib(other)}</b></span>`);
  }
  const cp=100*r.cache/tot;
  bar+=`<div class="vseg" style="flex:0 0 ${cp.toFixed(3)}%;background:#26313d" `+
       `title="page cache (reclaimable; mmap'd model files live here) — ${gib(r.cache)} GiB">${cp>=9?'cache':''}</div>`;
  lg.push(`<span class="e"><span class="sw" style="background:#26313d"></span>cache <b>${gib(r.cache)}</b></span>`);
  bar+=`<div class="vfree" title="${gib(r.free)} GiB free">${100*r.free/tot>=14?(gib(r.free)+' GiB free'):''}</div>`;
  lg.push(`<span class="e"><span class="sw" style="background:#0b0e12;border:1px solid var(--line)"></span>free <b>${gib(r.free)}</b></span>`);
  if((r.nodes||[]).length>1)
    lg.push(`<span class="e" style="opacity:.7">${r.nodes.map(n=>`${n.node} ${gib(n.free)} free`).join(' · ')}</span>`);
  h+=`<div class="vbar">${bar}</div><div class="vlg">${lg.join('')}</div>`;
  if(r.swap_total===0 && r.avail<8*GIB)
    h+=`<div class="warn">Under ${gib(r.avail)} GiB available and this box has no swap — the next big allocation fails hard, not gracefully.</div>`;
  h+='</div>';
  box.innerHTML=h;
}

/* ---- unloading ----
   Event delegation rather than an onclick per button: the legend is rebuilt
   from scratch every poll, and a name with a quote in it would break an inline
   handler. The button carries the model name in a data attribute instead. */
document.addEventListener('click', ev=>{
  const b = ev.target.closest && ev.target.closest('[data-unload]');
  if(b) unloadModel(b.getAttribute('data-unload'), b);
});

function vmsg(t,col){
  const el=document.getElementById('vramactmsg');
  if(el){ el.innerHTML=t; el.style.color = col||'var(--mut)'; }
}

async function unloadModel(name, btn, force){
  if(btn){ btn.disabled=true; btn.textContent='unloading…'; }
  try{
    const r=await fetch('/api/model/unload',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:name, force:!!force})});
    const d=await r.json();
    if(r.status===409 && d.confirm){
      /* Arm rather than confirm(): a modal dialog blocks the poll loop, and
         the thing you most want to see while deciding is the panel behind it. */
      vmsg(esc(d.error)+' <button class="ub arm" id="armed">unload anyway</button>','var(--amb)');
      const a=document.getElementById('armed');
      if(a) a.onclick=()=>{ vmsg(''); unloadModel(name,null,true); };
      if(btn){ btn.disabled=false; btn.textContent='unload'; }
      return;
    }
    if(d.error){ vmsg(esc(d.error),'var(--red)'); if(btn){btn.disabled=false;btn.textContent='unload';} return; }
    vmsg(d.noop ? esc(name)+' was not loaded.'
               : esc(d.model||name)+' unloaded — '+gib(d.freed)+' GiB freed.','var(--grn)');
    tick();
  }catch(e){ vmsg(String(e),'var(--red)'); if(btn){btn.disabled=false;btn.textContent='unload';} }
}

async function unloadAll(force){
  const btn=document.getElementById('unloadAllBtn');
  if(btn){ btn.disabled=true; btn.textContent='freeing…'; }
  try{
    const r=await fetch('/api/model/unload_all',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({force:!!force})});
    const d=await r.json();
    if(d.error){ vmsg(esc(d.error),'var(--red)'); return; }
    const done=(d.results||[]).filter(x=>x.ok).length;
    const failed=(d.results||[]).filter(x=>!x.ok && !x.skipped);
    let m = done? `${done} model${done>1?'s':''} unloaded — ${gib(d.freed)} GiB freed.`
                : 'Nothing was unloaded.';
    if(d.skipped) m += ` ${d.skipped} still working, left alone. `
      + `<button class="ub arm" id="armedall">free those too</button>`;
    if(failed.length) m += ' Failed: ' + failed.map(x=>esc(x.model)+' ('+esc(x.msg)+')').join(', ');
    vmsg(m, failed.length? 'var(--amb)' : (d.skipped? 'var(--amb)' : 'var(--grn)'));
    const a=document.getElementById('armedall');
    if(a) a.onclick=()=>{ vmsg(''); unloadAll(true); };
    tick();
  }catch(e){ vmsg(String(e),'var(--red)'); }
  finally{ if(btn){ btn.disabled=false; btn.textContent='Free all VRAM'; } }
}

/* ---------------- GPU ECC panel ---------------- */
function eccMsg(t,c){ const e=document.getElementById('eccmsg');
  if(e){ e.textContent=t; e.style.color=c||'var(--muted)'; } }
async function eccLoad(){
  const box=document.getElementById('eccbox'); if(!box) return;
  try{
    const d=await (await fetch('/api/ecc')).json();
    const pill=(txt,color)=>`<span style="display:inline-block;border:1px solid ${color};color:${color};border-radius:999px;padding:1px 10px;font-size:12px;margin-right:6px">${txt}</span>`;
    let h='';
    if(d.backend_error){
      h+=pill('cannot stage','var(--red)')+
        `<div class="note" style="margin-top:6px">${esc(d.backend_error)} — the status below is from the running kernel only. Stage it by hand: see the guide.</div>`;
    }else{
      h+=pill('staged: ECC '+(d.staged_off?'OFF':'ON'), d.staged_off?'var(--amber,#c90)':'var(--green,#0a0)');
      if(d.backend) h+=pill('via '+d.backend,'var(--muted)');
      if(d.staged_options!==undefined && d.staged_options!=='')
        h+=`<div class="note" style="margin-top:6px">staged kernel options: <code>${esc(d.staged_options)}</code></div>`;
    }
    if(d.running_off!==undefined)
      h+=pill('this boot: ECC '+(d.running_off?'OFF':'ON'),'var(--muted)');
    (d.cards||[]).forEach(c=>{
      h+=pill(`${c.card}: ${c.vram_gib} GiB${c.reclaimed?' ✓':''}`,
              c.reclaimed?'var(--green,#0a0)':'var(--muted)');
    });
    h+=`<div style="margin-top:8px;font-weight:600">${d.phase||''}</div>`;
    box.innerHTML=h;
  }catch(e){ box.innerHTML='<span class="muted">ecc status unavailable: '+String(e)+'</span>'; }
}
async function eccStage(mode){
  const warn = mode==='off'
    ? 'Stage ECC OFF?\n\nThis only stages the kernel option — nothing reboots.\nRemember: turn ECC back ON before training runs.'
    : 'Stage ECC ON (default)?\n\nThis only stages the kernel option — nothing reboots.';
  if(!confirm(warn)) return;
  eccMsg('staging…');
  try{
    const r=await fetch('/api/ecc/stage',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});
    const d=await r.json();
    eccMsg(d.msg, d.ok?'var(--green,#0a0)':'var(--red)');
    eccLoad();
  }catch(e){ eccMsg(String(e),'var(--red)'); }
}
eccLoad(); setInterval(eccLoad, 60000);

/* ---------------- scripts tab ---------------- */
/* The list re-renders on a timer, so the args boxes would be wiped mid-typing;
   scrArgs holds what you typed, keyed by script, and re-rendering restores it. */
let scrTimer=null, scrArgs={}, scrViewName=null;
function scrPoll(on){
  if(scrTimer){ clearInterval(scrTimer); scrTimer=null; }
  if(on){ loadScripts(); scrTimer=setInterval(loadScripts, 4000); }
}
async function loadScripts(){
  try{
    const d = await (await fetch('/api/scripts')).json();
    document.querySelectorAll('#scr_list input').forEach(i=>{ scrArgs[i.dataset.n]=i.value; });
    const rows = d.scripts.map(x=>{
      let st;
      if(x.state==='running') st='<span style="color:var(--accent)">running…</span>';
      else if(x.state==='done') st = x.exit===0 ? '<span style="color:var(--ok,#3a9)">exit 0</span>'
                                                : `<span style="color:var(--red)">exit ${x.exit}</span>`;
      else st='<span style="opacity:.55">idle</span>';
      const g = x.is_guard ? ' <span style="opacity:.55">(auto every '+Math.round(d.guard_every/60)+' min)</span>' : '';
      return `<div class="trow" style="display:flex;gap:.6em;align-items:center;margin:.25em 0">
        <span style="min-width:14em"><b>${esc(x.name)}</b>${g}</span>
        <span style="min-width:6em">${st}</span>
        <input data-n="${esc(x.name)}" value="${esc(scrArgs[x.name]||'')}" placeholder="args (optional)"
               style="width:11em">
        <button onclick="runScript('${esc(x.name)}')">Run</button>
        <button onclick="scrLog('${esc(x.name)}')">Log</button></div>`;
    });
    document.getElementById('scr_list').innerHTML =
      rows.length ? rows.join('')
      : '<div class="note" style="margin:0">Nothing to run: <code>'+esc(d.dir)+'</code> on this host '
        + 'is empty or does not exist. This tab lists <code>.sh</code>/<code>.py</code> files from '
        + 'that folder and runs them on the host as root. Create the folder on the host, point '
        + '<code>SCRIPTS_DIR</code> at it in the compose, and mount it in.</div>';
    if(scrViewName) scrLog(scrViewName, true);
  }catch(e){ document.getElementById('scr_msg').textContent=String(e); }
}
async function runScript(n){
  const inp=document.querySelector(`#scr_list input[data-n="${n}"]`);
  const r = await (await fetch('/api/scripts/run',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:n,args:inp?inp.value:''})})).json();
  document.getElementById('scr_msg').textContent = r.ok ? '' : (r.error||'failed');
  scrViewName=n; loadScripts();
}
async function scrLog(n, quiet){
  scrViewName=n;
  const d = await (await fetch('/api/scripts/log?name='+encodeURIComponent(n))).json();
  document.getElementById('scr_logwrap').style.display='';
  document.getElementById('scr_logttl').textContent='log — '+n;
  const pre=document.getElementById('scr_log');
  const stick = pre.scrollTop+pre.clientHeight >= pre.scrollHeight-8;
  pre.textContent=d.log;
  if(!quiet || stick) pre.scrollTop=pre.scrollHeight;
}

/* ---- Hugging Face import ----
   This tab resolves a repo and then gets out of the way: the download itself is
   handed to pull_start() on the server, so everything below stops mattering the
   moment a pull is queued. That is deliberate. A second download engine living
   in this file would be a second thing that can silently disagree with the
   Pulls tab about whether a model finished. */
let HFRES=null;

/* Plain English for what Ollama is doing right now. The raw status strings are
   accurate and unreadable — "verifying sha256 digest" sits there for four
   minutes on a 40 GB model with no progress of any kind, which looks exactly
   like a hang unless something says otherwise. */
function stageText(p){
  const s=String(p.status||'').toLowerCase();
  if(p.state==='queued')    return 'Waiting — one download runs at a time';
  if(p.state==='error')     return 'Failed';
  if(p.state==='cancelled') return 'Stopped';
  if(s==='connecting')                return 'Connecting to Ollama';
  if(s.startsWith('pulling manifest'))return 'Reading the file list';
  if(s.startsWith('writing manifest'))return 'Registering the model with Ollama';
  if(s.startsWith('removing'))        return 'Cleaning up unused layers';
  if(s.startsWith('verifying'))       return 'Verifying the checksum — this has no progress bar and takes minutes on a large model';
  if(s==='success'||p.state==='done') return 'Done — installed';
  if(s.startsWith('pulling'))         return 'Downloading weights';
  return p.status||'';
}

/* What each quantisation costs you, in one line. Ordered longest-prefix first
   so Q4_K_M is not caught by the Q4 rule. */
const QNOTE=[
  ['Q8_0',  'Near-lossless. Twice the size of Q4 for a difference you will struggle to measure.'],
  ['Q6_K',  'Very close to the original. The one to take when it still fits a single card.'],
  ['Q5_K_M','A real step up from Q4 in quality, and a real step up in size.'],
  ['Q5',    'Between Q4 and Q6 on both quality and size.'],
  ['Q4_K_M','The default balance and what most people run. Start here.'],
  ['Q4_K_S','A smaller Q4. Slightly worse than Q4_K_M to save a little space.'],
  ['Q4_0',  'The old Q4 format. Q4_K_M is better at the same size — prefer it if present.'],
  ['IQ4',   'Newer 4-bit scheme, a little smaller than Q4_K_M at similar quality. Slower on some backends.'],
  ['Q3',    'Squeezed. Noticeably degraded — worth it only to fit a card you otherwise cannot.'],
  ['IQ3',   'Squeezed hard. Degraded, but usually better than Q3 at the same size.'],
  ['Q2',    'Heavily squeezed. Expect real quality loss, not a subtle one.'],
  ['IQ2',   'Heavily squeezed. Expect real quality loss, not a subtle one.'],
  ['MXFP4', '4-bit float, the format gpt-oss ships in natively. Not a downconversion.'],
  ['BF16',  'Unquantised weights. Largest, and rarely worth the VRAM for inference.'],
  ['F16',   'Unquantised weights. Largest, and rarely worth the VRAM for inference.'],
  ['F32',   'Full precision. Enormous, and pointless for inference here.'],
];
function qnote(q){
  const u=String(q||'').toUpperCase();
  for(const [k,v] of QNOTE) if(u===k) return v;
  for(const [k,v] of QNOTE) if(u.startsWith(k)) return v;
  return '';
}

/* The KV reserve override lives on the Models tab. Read it if that pane has
   been rendered, and fall back to the server's own estimate if it has not —
   this tab must work on a cold page load where the Models tab was never
   opened, which is exactly how it will be used. */
function hfReserve(){
  const el=document.getElementById('mreserve');
  if(!el) return null;
  const v=parseFloat(el.value);
  return (isFinite(v) && v>=0)? v : null;
}

async function hfPaste(){
  try{
    const t=await navigator.clipboard.readText();
    if(t){ document.getElementById('hfurl').value=t.trim(); hfResolve(); }
  }catch(e){
    document.getElementById('hferr').textContent =
      'The browser would not hand over the clipboard. Paste into the box with Ctrl+V instead.';
  }
}

async function hfResolve(){
  const v=(document.getElementById('hfurl').value||'').trim();
  const err=document.getElementById('hferr');
  err.textContent='';
  if(!v){ err.textContent='Paste a Hugging Face link first.'; return; }
  document.getElementById('hfrepottl').textContent='Looking it up…';
  document.getElementById('hfrepopanel').style.display='';
  document.getElementById('hfmeta').textContent='';
  document.getElementById('hfwarn').innerHTML='';
  document.getElementById('hfquants').innerHTML='<div class="note" style="margin:0">asking Hugging Face…</div>';
  try{
    const r=await fetch('/api/hf/resolve',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url:v, ctx:catCtx, reserve:hfReserve()})});
    const d=await r.json();
    if(d.error && !d.repo){ err.textContent=d.error; document.getElementById('hfrepopanel').style.display='none'; return; }
    HFRES=d; hfDestSetup(d.lane); hfRender(d);
  }catch(e){
    err.textContent='Could not reach this app’s own API: '+e;
    document.getElementById('hfrepopanel').style.display='none';
  }
}

function hfRender(d){
  document.getElementById('hfrepottl').innerHTML =
    `<a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(d.repo)}</a>`;

  const m=d.meta||{}, bits=[];
  if(m.pipeline)  bits.push(esc(m.pipeline));
  if(m.license)   bits.push('licence '+esc(m.license));
  if(m.downloads!=null) bits.push(Number(m.downloads).toLocaleString()+' downloads/month');
  if(m.likes!=null)     bits.push(m.likes+' likes');
  if(m.modified)  bits.push('updated '+esc(String(m.modified).slice(0,10)));
  if(m.gated)     bits.push('<b style="color:var(--amb)">GATED</b>');
  if(m.private)   bits.push('<b style="color:var(--amb)">PRIVATE</b>');
  document.getElementById('hfmeta').innerHTML = bits.join(' · ') || '';

  const w=document.getElementById('hfwarn');
  let wh = (d.warnings||[]).map(x=>
    `<div class="note" style="color:var(--amb);margin:6px 0 0">${esc(x)}</div>`).join('');
  if(d.error) wh = `<div class="note" style="color:var(--red);margin:6px 0 0">${esc(d.error)}</div>` + wh;
  w.innerHTML = wh;

  const qs=d.quants||[];
  const el=document.getElementById('hfquants');
  document.getElementById('hfmanualrow').style.display = qs.length? '' : 'none';
  if(!qs.length){ el.innerHTML=''; return; }

  el.innerHTML='<table class="mt"><thead><tr><th>Quantisation</th><th>Size</th><th>Files</th>'
    + '<th>On these cards</th><th>What it is</th><th></th></tr></thead><tbody>'
    + qs.map(g=>{
        const [vl,vc]=VERD[(g.fit||{}).verdict]||VERD.unknown;
        const rec = g.recommended
          ? ' <span class="pill" style="border-color:var(--grn);color:var(--grn)">PICK</span>' : '';
        const pre = (d.picked && d.picked===String(g.quant).toUpperCase())
          ? ' <span class="pill" title="This is the quantisation your link pointed at">FROM LINK</span>' : '';
        const split = g.parts>1
          ? `<span title="split into ${g.parts} shards" style="color:var(--amb)">${g.parts}&nbsp;shards</span>`
          : (g.nfiles>1? g.nfiles+' files' : '1 file');
        const btn = `<button class="${g.recommended?'start':'ghost'}" `
          + `onclick="hfStart('${esc(d.repo)}','${esc(g.quant)}')">▼ Pull</button>`;
        return `<tr><td class="tg"><b>${esc(g.quant)}</b>${rec}${pre}</td>`
          + `<td class="nw"><b>${g.suspect? '<span style="color:var(--amb)" title="Hugging Face reported an implausible size for this file">unknown</span>' : fmtB(g.bytes)}</b></td>`
          + `<td class="nw">${split}</td>`
          + `<td class="nw"><span style="color:${vc}">${vl}</span>`
          + `<div class="note" style="margin:0;max-width:260px">${esc((g.fit||{}).why||'')}</div></td>`
          + `<td class="note" style="max-width:300px">${esc(qnote(g.quant))}</td>`
          + `<td class="nw">${btn}</td></tr>`;
      }).join('')
    + '</tbody></table>'
    + `<div class="note" style="margin:8px 0 0">Sizes are the real object sizes from Hugging Face. `
    + `Verdicts assume a ${(catCtx/1024).toFixed(0)}k context — change it on the Models tab and look again.</div>`;
}

/* The destination picker only appears when the lane's model directory is
   actually mounted in this container. If it is not, there is no second
   destination to offer and the tab behaves exactly as it always did. */
function hfDestSetup(lane){
  const row=document.getElementById('hfdestrow');
  const sel=document.getElementById('hfdest');
  const note=document.getElementById('hfdestnote');
  if(!lane || !lane.available){ row.style.display='none'; return; }
  if(!OLLAMA_ON){
    /* lane-only box: there is no second destination to pick. Every import
       goes to the lane's models dir; hfStart() forces dest='lane' too. */
    row.style.display='none';
    if(sel) sel.value='lane';
    return;
  }
  row.style.display='';
  const explain=()=>{
    note.innerHTML = sel.value==='lane'
      ? 'Writes the .gguf straight into <code>'+esc(lane.dir)+'</code>. Ollama never sees it. '
        + 'Add it to <code>llama-swap.yaml</code> as <code>-m /models/&lt;file&gt;</code> afterwards.'
      : 'Normal <code>ollama pull</code>. Note that models Ollama converts for its own engine '
        + 'may not load in llama.cpp — that is why the lane destination exists.';
  };
  sel.onchange=explain; explain();
}

/* ---- the lane panel ------------------------------------------------------
   Reads /api/lane, which merges three things the user otherwise has to hold in
   their head: what files exist on disk, what llama-swap's config knows about,
   and what is resident in VRAM right now. A file that is present but not in
   the config is the interesting case — 20 GB sitting there doing nothing. */
let LANE=null;
/* Which lane the tab is looking at. Survives refreshes of the tab but not of
   the page — deliberate: after a reload you see production first, which is
   the one that matters when something is wrong. */
let LANECUR='production';

function laneFmt(b){ return fmtB(b); }

function laneSel(name){ LANECUR=name; laneLoad(); }

async function laneLoad(){
  const err=document.getElementById('laneerr'); err.textContent='';
  try{
    const r=await fetch('/api/lane?lane='+encodeURIComponent(LANECUR));
    LANE=await r.json();
  }catch(e){ err.textContent=String(e); return; }
  const L=LANE;
  if(L.lane) LANECUR=L.lane;

  /* Lane picker + title. Buttons, not a select — one click, state visible. */
  document.getElementById('lanettl').textContent =
    'llama lane — '+LANECUR+' ('+(L.url||'').replace(/^https?:\/\//,'')+')';
  document.getElementById('lanesel').innerHTML = (L.lanes||[]).map(l=>
    '<button class="'+(l.name===LANECUR?'start':'ghost')+'" '
    + (l.name===LANECUR?'disabled ':'')
    + 'onclick="laneSel(\''+esc(l.name)+'\')">'+esc(l.name)
    + ' <span style="opacity:.6">:'+(l.url||'').split(':').pop()+'</span></button>'
  ).join(' ');

  let hdr='';
  if(!L.available){
    hdr = '<span style="color:var(--red)">The lane model directory is not mounted in this '
        + 'container.</span> Expected <code>'+esc(L.dir)+'</code>. Add the bind mount to '
        + 'docker-compose.yml and <b>Deploy</b> — a Restart will not do it, volumes only '
        + 'apply on container creation.';
  } else {
    hdr = 'Models in <code>'+esc(L.dir)+'</code> · config <code>'+esc(L.config)+'</code> '
        + (L.config_readable ? '' : '<span style="color:var(--red)">(not readable — mount '
          + 'the lane config directory to add models from here)</span>');
    if(L.api_error) hdr += '<br><span style="color:var(--amb)">llama-swap at '
        + esc(L.url)+' did not answer ('+esc(L.api_error)+') — it may be stopped.</span>';
  }
  document.getElementById('lanehdr').innerHTML=hdr;
  document.getElementById('laneunloadbtn').disabled = !!L.api_error;

  /* Loaded state comes from llama-swap itself rather than being inferred, so
     this cannot disagree with reality the way a cached list would. */
  const ms=L.models||[];
  document.getElementById('lanemodels').innerHTML = ms.length
    ? '<table class="tbl"><thead><tr><th>Model</th><th>Also answers to</th><th>State</th></tr></thead><tbody>'
      + ms.map(m=>{
          const on = (m.state||'')==='loaded';
          return '<tr><td class="tg"><b>'+esc(m.id)+'</b></td>'
            + '<td class="note" style="margin:0">'+esc((m.aliases||[]).join(', ')||'—')+'</td>'
            + '<td class="nw"><span style="color:'+(on?'var(--grn)':'var(--mut)')+'">'
            + (on?'● resident in VRAM':'○ unloaded')+'</span></td></tr>';
        }).join('')
      + '</tbody></table>'
    : '<div class="note" style="margin:0">llama-swap is not offering any models.</div>';

  const fs=L.files||[];
  let body = fs.length
    ? '<table class="tbl"><thead><tr><th>File</th><th>Size</th><th>Status</th><th></th></tr></thead><tbody>'
      + fs.map(f=>{
          const known=f.configured;
          /* A continuation shard of a split GGUF. It is loaded automatically
             when shard 1 is served — it must never be offered as its own
             model, and "unused" would invite deleting a file the model needs. */
          if(f.shard_of){
            const s = known
              ? '<span style="color:var(--grn)">in use via '+esc(f.shard_of)+'</span>'
              : '<span class="note" style="margin:0">loads with '+esc(f.shard_of)+'</span>';
            return '<tr style="opacity:.65"><td class="tg">&nbsp;&nbsp;↳ '+esc(f.name)
                 + '</td><td class="nw">'+laneFmt(f.bytes)+'</td><td>'+s+'</td><td></td></tr>';
          }
          let name = esc(f.name), size = laneFmt(f.bytes), warn = '';
          if(f.parts){
            /* Shard 1: the one addable file. Size shown is the WHOLE model —
               the fit question is about the total, not the first file. */
            name += ' <span class="note" style="margin:0">('+f.parts+' shards)</span>';
            size = laneFmt(f.group_bytes)+' total';
            if(f.shards_incomplete)
              warn = '<div style="color:var(--red)">only '+f.parts_present+' of '
                   + f.parts+' shards on disk — will NOT load until the pull finishes</div>';
          }
          const badge = known
            ? '<span style="color:var(--grn)">in the config</span>'
            : (f.mmproj ? '<span class="note" style="margin:0">vision projector — attach it to a model</span>'
                        : '<span style="color:var(--amb)">not in the config — unused</span>');
          const btn = (known||f.mmproj||f.shards_incomplete) ? ''
            : '<button class="start" onclick="laneAddOpen(\''+esc(f.name)+'\')">Add to lane</button>';
          return '<tr><td class="tg">'+name+'</td><td class="nw">'+size
               + '</td><td>'+badge+warn+'</td><td class="nw">'+btn+'</td></tr>';
        }).join('')
      + '</tbody></table>'
    : '<div class="note" style="margin:0">No .gguf files in '+esc(L.dir)+' yet. The HF import tab can put one there.</div>';
  if((L.partials||[]).length){
    body += '<div class="note" style="color:var(--amb)"><b>Unfinished downloads:</b> '
          + L.partials.map(esc).join(', ')
          + '. These are <code>.part</code> files from an interrupted or failed pull. '
          + 'Re-running the same pull resumes them; nothing else can read them.</div>';
  }
  document.getElementById('lanefiles').innerHTML=body;
}

/* Push a cheat-sheet row into the add form. Only fills fields the row
   actually specifies — the helper row sets context but leaves top-k alone,
   because "as family" is the honest answer there, not a number. */
function laneCheat(topk, ctx){
  const p=document.getElementById('laneaddpanel');
  if(p.style.display==='none'){
    document.getElementById('laneerr').textContent =
      'Pick a file with "Add to lane" first, then apply a preset.';
    return;
  }
  if(topk!==null) document.getElementById('laneaddtopk').value=topk;
  if(ctx!==null)  document.getElementById('laneaddctx').value=ctx;
  document.getElementById('laneerr').textContent='';
}

function laneAddOpen(fname){
  const p=document.getElementById('laneaddpanel');
  p.style.display='';
  document.getElementById('laneaddttl').textContent='Add '+fname+' to the lane';
  /* A reasonable default name: strip the extension, the shard suffix
     (-00001-of-00003 — the id names the MODEL, not a file of it), and the
     quant suffix noise. */
  const guess=fname.replace(/\.gguf$/i,'')
                   .replace(/-\d{4,5}-of-\d{4,5}$/,'')
                   .replace(/[^A-Za-z0-9._-]/g,'-').slice(0,48);
  document.getElementById('laneaddid').value=guess.toLowerCase();
  const mm=document.getElementById('laneaddmm');
  const cands=(LANE.files||[]).filter(f=>f.mmproj);
  mm.innerHTML='<option value="">none — text only</option>'
    + cands.map(f=>'<option value="'+esc(f.name)+'">'+esc(f.name)+'</option>').join('');
  document.getElementById('laneaddnote').innerHTML =
    'These flags are <b>defaults, not measured</b>. Context 131072 and top-k 64 suit '
    + 'Muse Glimmer; Qwen wants top-k 20. Check what this model actually asks for — '
    + 'running one model with another\'s sampler settings gives you answers that are '
    + 'quietly worse rather than obviously broken.'
    + (cands.length ? '' : ' No mmproj file is present, so this will be text-only.');
  p.scrollIntoView({behavior:'smooth', block:'nearest'});
  document.getElementById('laneaddid').focus();
}

async function laneAdd(){
  const err=document.getElementById('laneerr'); err.textContent='';
  const ttl=document.getElementById('laneaddttl').textContent;
  const file=ttl.replace(/^Add /,'').replace(/ to the lane$/,'');
  const body={ id:(document.getElementById('laneaddid').value||'').trim(),
               file:file,
               lane:LANECUR,
               ctx:parseInt(document.getElementById('laneaddctx').value,10)||131072,
               top_k:parseInt(document.getElementById('laneaddtopk').value,10)||64,
               mmproj:document.getElementById('laneaddmm').value||'' };
  try{
    const r=await fetch('/api/lane/add',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d=await r.json();
    if(!d.ok){ err.textContent=d.msg||'could not add it'; return; }
    document.getElementById('laneaddpanel').style.display='none';
    LANE=d.lane; laneLoad();
  }catch(e){ err.textContent=String(e); }
}

async function laneUnload(){
  const err=document.getElementById('laneerr'); err.textContent='';
  try{
    const r=await fetch('/api/lane/unload',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({lane:LANECUR})});
    const d=await r.json();
    if(!d.ok){ err.textContent=d.msg||'could not unload'; }
    laneLoad();
  }catch(e){ err.textContent=String(e); }
}

async function hfStart(repo, quant){
  const err=document.getElementById('hferr');
  err.textContent='';
  const dsel=document.getElementById('hfdest');
  const dest=!OLLAMA_ON ? 'lane'
             : (dsel && document.getElementById('hfdestrow').style.display!=='none')
             ? dsel.value : 'ollama';
  try{
    const r=await fetch('/api/hf/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({repo:repo, quant:quant, dest:dest})});
    const d=await r.json();
    if(!d.ok){ err.textContent=d.msg||'could not start'; return; }
    document.getElementById('hfprogpanel').style.display='';
    /* Lane pulls carry a "lane:" prefix so they can share the queue with
       Ollama pulls without colliding. Both belong in this panel. */
    renderPulls((d.pulls||[]).filter(p=>p.tag.indexOf('hf.co/')===0
                                      || p.tag.indexOf('lane:')===0),'hfout',
                'Queued. It will appear here in a moment.');
    pullTick();
  }catch(e){ err.textContent=String(e); }
}

function hfManual(){
  if(!HFRES) return;
  const q=(document.getElementById('hfmanual').value||'').trim();
  if(!q){ document.getElementById('hferr').textContent='Type a quantisation name, e.g. Q4_K_M.'; return; }
  hfStart(HFRES.repo, q);
}

/* The alias picker is filled from finished hf.co pulls, not from the whole
   installed list: naming something you did not just download is not what this
   panel is for, and a select with forty entries in it is not a convenience. */
function hfAliasOptions(pulls){
  const done=(pulls||[]).filter(p=>p.state==='done' && p.tag.indexOf('hf.co/')===0).map(p=>p.tag);
  const sel=document.getElementById('hfaliassrc');
  if(!sel) return;
  document.getElementById('hfaliaspanel').style.display = done.length? '' : 'none';
  const cur=sel.value;
  const want=done.join('|');
  if(sel.dataset.k===want) return;
  sel.dataset.k=want;
  sel.innerHTML=done.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join('');
  if(done.indexOf(cur)>=0) sel.value=cur;
  const dst=document.getElementById('hfaliasdst');
  if(dst && !dst.value && done.length){
    /* A sane default: the repo name, lowercased, with the noise stripped. */
    const repo=done[0].split(':')[0].split('/').pop();
    dst.value=repo.replace(/-?GGUF$/i,'').replace(/[^A-Za-z0-9._-]/g,'-').toLowerCase();
  }
}

async function hfAlias(){
  const src=document.getElementById('hfaliassrc').value;
  const dst=(document.getElementById('hfaliasdst').value||'').trim();
  const err=document.getElementById('hfaliaserr');
  err.textContent='';
  if(!src){ err.textContent='Nothing to name yet.'; return; }
  if(!dst){ err.textContent='Type the short name you want.'; return; }
  try{
    const r=await fetch('/api/hf/alias',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({source:src,name:dst})});
    const d=await r.json();
    if(!d.ok){ err.textContent=d.msg||'failed'; return; }
    err.innerHTML=`<span style="color:var(--grn)">Done. <code>${esc(d.name)}</code> now points at the `
      + `same weights. The long tag is still there; delete it from the Models tab if you want it gone.</span>`;
    loadInstalled();
  }catch(e){ err.textContent=String(e); }
}

/* ---------------- thermal test ---------------- */
/* The browser is a VIEW here, not the controller. Every timer, threshold and
   abort lives in the server thread, so closing this tab can't leave the cards
   under load with nobody watching. */
let curTab='live', tmodels=null;
/* Driven by a list rather than a line per pane, because the previous version
   hardcoded two tabs in four places and adding a third meant touching all of
   them. A pane that isn't rendered (the thermal test, when SHOW_THERMAL_TEST
   is off) is simply skipped — no null dereference, no missing-element error in
   the console. */
const TABS=['live','models','pulls','hf','lane','bench','scripts','test'];

/* ---------------- Bench tab ---------------- */
let benchTimer=null;
async function benchLoad(){
  const r=await fetch('/api/bench'); const d=await r.json();
  const sel=document.getElementById('benchmodel');
  const loaded=(d.models||[]).filter(m=>m.state&&m.state!=='stopped');
  document.getElementById('benchloaded').textContent =
    loaded.length? loaded.map(m=>m.id).join(', ') : 'nothing';
  if(!d.bench.running || sel.options.length===0){
    const cur=sel.value;
    sel.innerHTML=(d.models||[]).map(m=>{
      const tag=(m.state&&m.state!=='stopped')?' (loaded)':'';
      return `<option value="${m.id}">${m.id}${tag}</option>`;}).join('');
    /* default to the resident model so a bench doesn't surprise-evict */
    const first=loaded[0]; 
    if(cur && [...sel.options].some(o=>o.value===cur)) sel.value=cur;
    else if(first) sel.value=first.id;
  }
  benchRender(d);
  if(d.bench.running && !benchTimer) benchTimer=setInterval(benchLoad,2000);
  if(!d.bench.running && benchTimer){clearInterval(benchTimer);benchTimer=null;}
}
function benchRender(d){
  const b=d.bench;
  document.getElementById('benchgo').disabled=!!b.running;
  document.getElementById('benchnote').innerHTML =
    (b.running?('running <b>'+b.model+'</b> since '+b.started+' — levels '+b.levels.join(',')+'. ')
             :(b.model?('last run: '+b.model+'. '):'')) +
    (b.note||'') + (b.error?(' <span style="color:var(--bad)">'+b.error+'</span>'):'');
  document.getElementById('benchrows').innerHTML=(b.results||[]).map(r=>
    `<tr><td>${r.n}</td><td><b>${r.agg}</b></td><td>${r.user_avg}</td>`+
    `<td>${r.user_min}</td><td>${r.ttft_max}s</td><td>${r.ok}${r.fail?('/'+(r.ok+r.fail)):''}</td></tr>`).join('')
    || '<tr><td colspan="6" class="muted">no results yet</td></tr>';
  const hist=d.history||[];
  const ns=[...new Set(hist.flatMap(h=>Object.keys(h.levels).map(Number)))].sort((a,b)=>a-b);
  document.getElementById('benchhisthead').innerHTML =
    '<th>when</th><th>model</th><th>slots</th>'+ns.map(n=>'<th>N='+n+'</th>').join('');
  document.getElementById('benchhistrows').innerHTML = hist.slice().reverse().map(h=>
    '<tr><td class="nw">'+h.ts+'</td><td>'+h.model+'</td><td>'+(h.slots??'?')+'</td>'+
    ns.map(n=>{const a=h.levels[n];const u=(h.user||{})[n];
      return '<td>'+(a==null?'':(a+(u!=null?' <span class="muted">('+u+')</span>':'')))+'</td>';}).join('')+
    '</tr>').join('') || '<tr><td colspan="9" class="muted">no history yet</td></tr>';
}
async function benchStart(){
  const levels=[...document.querySelectorAll('#benchlevels input:checked')].map(c=>+c.value);
  const model=document.getElementById('benchmodel').value;
  if(!model||!levels.length){alert('pick a model and at least one level');return;}
  const r=await fetch('/api/bench/start',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model,levels,max_tokens:+document.getElementById('benchmaxtok').value||200})});
  const d=await r.json();
  if(!d.ok) alert(d.msg);
  benchLoad();
}

function showTab(t){
  curTab=t;
  TABS.forEach(n=>{
    const p=document.getElementById('pane-'+n), b=document.getElementById('tab-'+n);
    if(p) p.style.display = (n===t)?'':'none';
    if(b) b.className = 'tab'+((n===t)?' on':'');
  });
  if(t==='test')   loadTestMeta();
  if(t==='models') loadModelsTab();
  if(t==='lane')   laneLoad();
  if(t==='bench')  benchLoad();
  /* The HF tab shows live download progress, so it wants the fast poll
     for the same reason the Pulls tab does. */
  scrPoll(t==='scripts');
  /* The pull poller runs while the tab is open. It keeps running server-side
     either way — this only controls how often the page asks. */
  pullPoll(t==='pulls' || t==='hf');
}
/* ---- lane-only boxes ----
   The server says whether Ollama exists here (OLLAMA_URL empty in the compose
   means it does not). One dashboard serves both kinds: with Ollama nothing
   changes; on a lane-only box the Ollama furniture is removed rather than
   left to error — the Models tab (an Ollama catalogue), the ollama-pull form,
   the HF destination picker (everything goes to the lane), and the bench
   tab's "drive load with Ollama" option. Runs every poll, applies once. */
let OLLAMA_ON=true, _olApplied=null;
function applyOllamaMode(on){
  if(on===undefined || on===_olApplied) return;
  _olApplied=on; OLLAMA_ON=!!on;
  if(OLLAMA_ON) return;              /* Ollama present: change nothing */
  const hide=id=>{const e=document.getElementById(id); if(e) e.style.display='none';};
  hide('tab-models');                 /* the tab is 100% Ollama */
  hide('pullformpanel');              /* ollama-pull form; lane pulls still list below */
  const tm=document.getElementById('tmode');
  if(tm){ const o=tm.querySelector('option[value="ollama"]');
          if(o) o.remove(); tm.value='observe'; tmodeChanged(); }
  /* If the hidden Models tab was somehow active, fall back to Live. */
  if(typeof curTab!=='undefined' && curTab==='models') showTab('live');
}

function tmodeChanged(){
  const ol = document.getElementById('tmode').value==='ollama';
  document.getElementById('tmodelwrap').style.display = ol?'':'none';
  document.getElementById('tworkwrap').style.display  = ol?'':'none';
}
async function loadTestMeta(){
  if(!document.getElementById('tmodel')) return;
  try{
    const d = await (await fetch('/api/test')).json();
    if(tmodels===null){
      const sel=document.getElementById('tmodel');
      tmodels=d.models||[];
      sel.innerHTML = tmodels.length
        ? tmodels.map(m=>`<option value="${esc(m)}">${esc(m)}</option>`).join('')
        : '<option value="">(Ollama not reachable)</option>';
    }
    if(d.fan_dir===false){
      document.getElementById('terr').textContent =
        'gpu-fan-control’s folder isn’t mounted, so this run has no protective action available at all — it can watch and stop the load, but it cannot raise the fans.';
    }
  }catch(e){}
}
const PHASES=[['baseline','Baseline'],['load','Load'],['hold','Hold'],['cooldown','Cooldown']];
function mmss(s){ if(s==null) return '—'; s=Math.max(0,Math.round(s));
  return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }
function renderTest(t){
  /* The second half of this guard is what lets the pane be absent. tick() calls
     this every two seconds whether the thermal tab was rendered or not, and
     without it every one of those calls throws on the first getElementById. */
  if(!t || !document.getElementById('trunBtn')) return;
  const running = t.phase!=='idle' && t.phase!=='done';
  document.getElementById('trunBtn').style.display  = running?'none':'';
  document.getElementById('tstopBtn').style.display = running?'':'none';
  document.getElementById('tdot').className = 'dot'+(running?' on':'');
  document.getElementById('tstat').textContent = running
    ? `${t.phase} · ${mmss(t.elapsed)} elapsed` : (t.phase==='done'?'finished':'not running');
  /* A run must be visible from the Live tab too — you should never have to be
     looking at the right tab to find out the cards are being cooked. */
  const flag=document.getElementById('tabflag');
  flag.innerHTML = running
    ? `<span style="color:${t.abort?'var(--red)':'var(--amb)'}">● thermal run: ${esc(t.phase)}${t.abort?' — ABORTING':''}</span>`
    : '';

  document.getElementById('tprogpanel').style.display = (running||t.phase==='done')?'':'none';
  let ph='', hit=false;
  PHASES.forEach(([k,label])=>{
    const isnow = t.phase===k;
    if(isnow) hit=true;
    const done = !hit && !isnow;
    const tot = isnow ? t.phase_total : null;
    const pct = isnow ? (tot?Math.min(100,100*t.phase_elapsed/tot):50) : (done?100:0);
    ph+=`<div class="ph ${isnow?'now':''} ${done?'done':''}">
           <span class="nm">${label}</span>
           <span class="tr"><span class="fl" style="width:${pct}%"></span></span>
           <span class="tm">${isnow?mmss(t.phase_elapsed):(done?'done':'—')}</span></div>`;
  });
  document.getElementById('tphases').innerHTML=ph;

  const p=t.peak||{}; let pk='';
  Object.keys(p).filter(k=>k.startsWith('j:')).sort().forEach(k=>{
    const pci=k.slice(2);
    pk+=itm(`${pci} peak junction`, p[k]==null?'—':p[k].toFixed(0), '°C', gcol(p[k]));
  });
  Object.keys(p).filter(k=>k.startsWith('w:')).sort().forEach(k=>{
    pk+=itm(`${k.slice(2)} peak board`, p[k]==null?'—':p[k].toFixed(0), 'W');
  });
  if(p.watts!=null) pk+=itm('peak system power', p.watts.toFixed(0), 'W');
  if(t.reqs)        pk+=itm('requests completed', t.reqs, '');
  if(t.errors)      pk+=itm('load errors', t.errors, '', 'var(--amb)');
  document.getElementById('tpeaks').innerHTML=pk||'<span class="muted">no samples yet</span>';
  document.getElementById('tlogline').innerHTML = t.logfile
    ? `Logging to <code>${esc(t.logfile)}</code>${t.steady?' · reached thermal steady state':''} — it’s in the History panel on the Live tab, with a <code>test_phase</code> column.`
    : '';

  const vp=document.getElementById('tverdictpanel');
  if(t.verdict){
    vp.style.display='';
    document.getElementById('tverdict').innerHTML =
      `<div class="vd ${t.verdict.ok?'pass':'fail'}"><h4>${esc(t.verdict.headline)}</h4>${esc(t.verdict.detail)}</div>`;
  } else vp.style.display='none';

  const nb=document.getElementById('tnotes');
  nb.innerHTML = (t.notes&&t.notes.length)
    ? '<div class="nlog">'+t.notes.map(n=>{
        const d=new Date(n.t*1000).toTimeString().slice(0,8);
        return `${d}  ${/ABORT/.test(n.msg)?'<b style="color:var(--red)">'+esc(n.msg)+'</b>':esc(n.msg)}`;
      }).join('<br>')+'</div>'
    : '<span class="muted">Nothing yet.</span>';
}
async function startTest(){
  const e=document.getElementById('terr'); e.textContent='';
  const body={
    mode: document.getElementById('tmode').value,
    model: document.getElementById('tmodel').value,
    abort_c: document.getElementById('tabort').value,
    abort_watts: document.getElementById('twatts').value,
    baseline_s: document.getElementById('tbase').value,
    max_load_s: document.getElementById('tload').value,
    hold_s: document.getElementById('thold').value,
    cooldown_s: document.getElementById('tcool').value,
    workers: document.getElementById('twork').value,
    protect: document.getElementById('tprot').checked,
  };
  try{
    const r=await fetch('/api/test/start',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(!d.ok){ e.textContent=d.error||'could not start'; return; }
    renderTest(d.test); tick();
  }catch(err){ e.textContent=String(err); }
}
async function stopTest(){
  try{
    const d=await (await fetch('/api/test/stop',{method:'POST'})).json();
    renderTest(d.test); loadLogs();
  }catch(err){ document.getElementById('terr').textContent=String(err); }
}


/* ---------------- Models browser + pull manager ----------------
   Sizes are fetched one tag at a time from /api/catalog/size, which caches the
   registry answer for an hour server-side. The pool below is four wide because
   a family can have thirty tags and thirty parallel requests to the same host
   is how you get rate-limited by a service that has been nothing but polite. */
const VERD = {
  one_card:       ['ONE CARD',        'var(--grn)'],
  one_card_tight: ['ONE CARD, TIGHT', 'var(--amb)'],
  spans:          ['SPANS BOTH',      'var(--amb)'],
  too_big:        ['TOO BIG',         'var(--red)'],
  unknown:        ['UNKNOWN',         'var(--mut)']
};
function fmtB(b){
  if(b==null) return '—';
  const u=['B','KiB','MiB','GiB','TiB']; let i=0, v=Number(b);
  while(v>=1024 && i<u.length-1){ v/=1024; i++; }
  return (i>=3? v.toFixed(1) : Math.round(v)) + ' ' + u[i];
}
function fmtDur(s){
  if(s==null) return '—';
  s=Math.max(0,Math.round(s));
  if(s<60) return s+'s';
  const h=Math.floor(s/3600), m=Math.floor(s/60)%60;
  return h? h+'h '+m+'m' : m+'m '+String(s%60).padStart(2,'0')+'s';
}
let HW=null, CAT=null, SIZES={}, catCtx=32768;
/* LIB is the whole ollama.com index as scraped, held in the page. 234 records
   at ~450 bytes each is ~100 KB, which is nothing, and holding it means search,
   filter and sort are all local: no request per keystroke. */
let LIB=null, mOpen=null, mBlocked=false;
let LIBINST=new Set(), LIBCUR=new Set();
const mCaps=new Set(), mFlags=new Set();
/* The sheet's fetched payload, kept so the tag filter can re-render the table
   without asking the registry again. */
let mData=null, mFitOnly=false, mTagQ="";

async function loadHw(){
  try{
    HW = await (await fetch('/api/hw')).json();
  }catch(e){ return; }
  const cards = HW.cards||[];
  document.getElementById('hwrow').innerHTML =
    (cards.length
      ? cards.map((c,i)=>`<div class="item"><span class="k">${esc(c.slot)}</span>`
          + `<span class="v">${(c.vram_total/1073741824).toFixed(1)}<span class="u"> GiB</span></span>`
          + `<span class="k">${fmtB(c.vram_free)} free right now</span></div>`).join('')
      : '<div class="note">No amdgpu cards are visible to this container.</div>')
    + itm('Total VRAM', (HW.total/1073741824).toFixed(1), 'GiB', 'var(--txt)')
    + itm('Free now',   (HW.free/1073741824).toFixed(1),  'GiB', 'var(--mut)');
  const d = HW.disk||{};
  let dn = d.mounted
    ? `Blob store: <b>${fmtB(d.free)}</b> free of ${fmtB(d.total)} at <code>${esc(d.path)}</code>.`
    : `Ollama's models directory is not mounted, so free-space checks and the orphan list on the Pulls tab are unavailable. Add a volume for it and set <code>OLLAMA_MODELS_DIR</code>.`;
  if(HW.ctx_env) dn += ` Ollama's own OLLAMA_CONTEXT_LENGTH is ${esc(HW.ctx_env)}.`;
  document.getElementById('kvnote').innerHTML = dn;
}

function mctxChanged(){
  catCtx = parseInt(document.getElementById('mctx').value,10) || 32768;
  /* The grid's fit estimate is computed here, in the page, so it just re-draws.
     The sheet's verdicts come from the server and have to be re-asked. */
  renderCatalog();
  if(mOpen) openFamily(mOpen, mData && mData.all);
}
function reserveArg(){
  const v = parseFloat(document.getElementById('mreserve').value);
  return (isFinite(v) && v>=0) ? ('&reserve='+v) : '';
}

/* A tiny promise pool. Keeps the first paint fast without opening thirty
   sockets at once. */
async function pool(items, n, fn){
  const it = items[Symbol.iterator]();
  const work = async()=>{ for(;;){ const x=it.next(); if(x.done) return; await fn(x.value); } };
  await Promise.all(Array.from({length:Math.min(n,items.length)}, work));
}

async function sizeTag(repo, tag, force){
  const key = repo+'|'+tag+'|'+catCtx+'|'+document.getElementById('mreserve').value;
  if(!force && SIZES[key]) return SIZES[key];
  try{
    const d = await (await fetch(`/api/catalog/size?repo=${encodeURIComponent(repo)}`
      + `&tag=${encodeURIComponent(tag)}&ctx=${catCtx}${reserveArg()}`)).json();
    SIZES[key]=d; return d;
  }catch(e){ return {src:'none', error:String(e)}; }
}

function tagRow(repo, t, d){
  const id = 'sz-'+repo.replace(/[^a-z0-9]/gi,'')+'-'+t.tag.replace(/[^a-z0-9]/gi,'');
  if(!d) return `<tr id="${id}"><td class="tg">${esc(t.tag)}</td>`
    + `<td colspan="3" class="note" style="margin:0">checking…</td></tr>`;
  const fit = d.fit||{};
  const [vl,vc] = VERD[fit.verdict] || VERD.unknown;
  const srcPill = d.src==='registry' ? '<span class="pill ok">registry</span>'
                : d.src==='catalog'  ? '<span class="pill guess">catalog</span>' : '';
  const bad = d.warn==='bad';
  let warn='';
  if(d.warn_why) warn = `<div class="note" style="margin:3px 0 0;color:${bad?'var(--red)':'var(--amb)'}">`
    + `${bad?'✕ ':'⚠ '}${esc(d.warn_why)}</div>`;
  const note = t.note ? `<div class="note" style="margin:3px 0 0">${esc(t.note)}</div>` : '';
  const btn = bad
    ? `<button class="ghost" disabled title="This format cannot run on gfx1030.">—</button>`
    : `<button class="ghost" onclick="queuePull('${esc(repo)}:${esc(t.tag)}')">Pull</button>`;
  return `<tr id="${id}">`
    + `<td class="tg">${esc(t.tag)}${note}${warn}</td>`
    + `<td class="nw">${d.gib!=null? d.gib.toFixed(1)+' GiB' : '—'}${srcPill}</td>`
    + `<td class="nw" style="color:${vc};font-weight:600">${vl}`
    + `<div class="note" style="margin:2px 0 0;font-weight:400">`
    + `needs ${fit.need_gib!=null? fit.need_gib+' GiB':'—'}`
    + `${fit.kv_gib!=null? ' (+'+fit.kv_gib+' KV est.)':''}</div></td>`
    + `<td class="nw">${btn}</td></tr>`;
}

/* ---- the grid ----------------------------------------------------------
   Everything on a card comes from what ollama.com/library publishes: name,
   description, capability badges, parameter-size tokens, pull count, tag count,
   last update. It does NOT come from a list written by hand in this file, which
   is the whole point of the change -- a model that shipped this morning is here
   the next time you press Sync.

   What the index does not publish is byte sizes, and asking the registry for
   them would be 234 repos x ~15 tags of manifest fetches to paint one page. So
   the fit badge here is an ESTIMATE from the parameter size, and it says so.
   Click a card and the sheet asks the registry for the real bytes.  */

/* Measured on this machine, not looked up: qwen3.6 35B-a3b q4_K_M is 23 GB on
   disk, gemma4:31b is 19 GB, llama3.1:8b is 4.9 GB. That is 0.57-0.61 GiB per
   billion parameters. Rounded up, so the estimate errs toward "too big" rather
   than toward a promise the sheet then has to take back. */
const GIB_PER_B_Q4 = 0.62;
const FITV = {one:0, tight:2, no:3, unk:4};

/* A parameter-size token as ollama writes it. 8x7b is a Mixtral-style MoE and
   does not weigh 56B; a3b/e4b name the ACTIVE parameters, and the file on disk
   is the total, so those under-read -- which the sheet then corrects. */
function paramsOf(tok){
  if(!tok) return null;
  const t = String(tok).toLowerCase();
  let m = t.match(/^(\d+)x([\d.]+)b$/);   if(m) return +m[1] * +m[2] * 0.85;
  m = t.match(/^[ea]([\d.]+)b$/);         if(m) return +m[1];
  m = t.match(/^([\d.]+)b$/);             if(m) return +m[1];
  m = t.match(/^([\d.]+)m$/);             if(m) return +m[1]/1000;
  m = t.match(/^([\d.]+)t$/);             if(m) return +m[1]*1000;
  return null;
}

/* How many billion parameters fit, at q4, right now. Read off the cards that
   are actually in the machine -- pull one out and every badge below moves. */
function fitBudget(){
  const per = (LIB && LIB.biggest_gib) || 30;
  const n   = (LIB && (LIB.cards||[]).length) || 1;
  /* The KV anchor in this app is 1.3 GiB per 32K per model and live numbers on
     this box have run higher, so the grid hedges it upward. The sheet uses the
     server's figure; when the two disagree, the sheet is right. */
  const kv  = 1.3 * (catCtx/32768) * 1.6;
  return {one: Math.max(0,(per - 0.8 - kv)) / GIB_PER_B_Q4,
          all: Math.max(0,(per*n - 0.8*n - kv)) / GIB_PER_B_Q4,
          per: per, n: n};
}

/* The badge reports the LARGEST variant that fits, not merely that something
   does. An earlier version judged a repo by its smallest tag, so llama3.1 and
   gemma3 both read "fits" and the badge told you nothing. */
function fitOf(m){
  const caps = m.capabilities||[];
  const sizes = m.sizes||[];
  if(caps.indexOf('cloud')>=0 && !sizes.length)
    return {k:'unk', t:'CLOUD ONLY', why:"this one runs on ollama's servers, not on your hardware"};
  const ps = sizes.map(s=>({s:s,p:paramsOf(s)})).filter(x=>x.p!==null);
  if(!ps.length) return {k:'unk', t:'SIZE ?', why:'the library index publishes no parameter size for this one — open it to size the tags'};
  const B = fitBudget();
  const fits = ps.filter(x=>x.p<=B.one);
  if(fits.length){
    if(fits.length===ps.length)
      return {k:'one', t:'FITS ONE CARD', why:'every published size should fit one '+B.per.toFixed(0)+' GiB card at q4'};
    let best=fits[0]; fits.forEach(x=>{ if(x.p>best.p) best=x; });
    return {k:'one', t:'UP TO '+String(best.s).toUpperCase(),
            why:'the '+best.s+' build should fit one card at q4; the bigger ones will not'};
  }
  const min = Math.min.apply(null, ps.map(x=>x.p));
  if(B.n>1 && min<=B.all)
    return {k:'tight', t:'SPANS BOTH', why:'the smallest build needs both cards — every token then crosses PCIe'};
  return {k:'no', t:'TOO BIG', why:'the smallest published size is past what '+B.n+' card'+(B.n===1?'':'s')+' can hold at q4'};
}

function fmtPulls(n){
  if(n==null) return '—';
  if(n>=1e9) return (n/1e9).toFixed(1)+'B';
  if(n>=1e6) return (n/1e6).toFixed(n>=1e7?0:1)+'M';
  if(n>=1e3) return (n/1e3).toFixed(n>=1e4?0:1)+'K';
  return String(n);
}
function tsOf(m){ const t = Date.parse(m.updated||''); return isFinite(t)? t : 0; }
function minP(m){
  const ps=(m.sizes||[]).map(paramsOf).filter(x=>x!==null);
  return ps.length? Math.min.apply(null,ps) : null;
}
function maxP(m){
  const ps=(m.sizes||[]).map(paramsOf).filter(x=>x!==null);
  return ps.length? Math.max.apply(null,ps) : null;
}

const FLAGS = [
  ['fit',       'Fits one card'],
  ['local',     'Runs locally'],
  ['installed', 'Already installed'],
  ['curated',   'Has notes in this app']
];

function toggleCap(c){  if(mCaps.has(c)) mCaps.delete(c); else mCaps.add(c); renderCatalog(); }
function toggleFlag(f){ if(mFlags.has(f)) mFlags.delete(f); else mFlags.add(f); renderCatalog(); }
function clearFilters(){ mCaps.clear(); mFlags.clear();
  const q=document.getElementById('mq'); if(q) q.value='';
  renderCatalog(); }

function renderCatalog(){
  const g = document.getElementById('mgrid');
  if(!g || !LIB) return;
  const q    = (document.getElementById('mq').value||'').trim().toLowerCase();
  const sort = document.getElementById('msort').value;

  let list = (LIB.models||[]).slice();
  if(q) list = list.filter(m =>
    (m.name+' '+m.repo+' '+(m.description||'')+' '+(m.capabilities||[]).join(' ')
     +' '+(m.sizes||[]).join(' ')).toLowerCase().includes(q));
  /* Capabilities are ANDed: picking tools and vision means you want a model
     that does both, not either. */
  if(mCaps.size) list = list.filter(m => {
    const c=new Set(m.capabilities||[]);
    for(const w of mCaps) if(!c.has(w)) return false;
    return true;
  });
  if(mFlags.has('fit'))       list = list.filter(m => fitOf(m).k==='one');
  if(mFlags.has('local'))     list = list.filter(m => (m.capabilities||[]).indexOf('cloud')<0);
  if(mFlags.has('installed')) list = list.filter(m => LIBINST.has(m.repo));
  if(mFlags.has('curated'))   list = list.filter(m => LIBCUR.has(m.repo));

  const cmp = {
    popular:(a,b)=> (b.pulls||0)-(a.pulls||0) || (a.rank||0)-(b.rank||0),
    newest: (a,b)=> tsOf(b)-tsOf(a) || (a.rank||0)-(b.rank||0),
    name:   (a,b)=> String(a.name).localeCompare(String(b.name)),
    small:  (a,b)=> (minP(a)==null?1e9:minP(a)) - (minP(b)==null?1e9:minP(b)),
    big:    (a,b)=> (maxP(b)||0)-(maxP(a)||0)
  }[sort] || ((a,b)=>(a.rank||0)-(b.rank||0));
  list.sort(cmp);

  /* Capability chips carry their own counts, so an empty vocabulary is visible
     rather than silently absent -- if a restyle breaks the badge parse this row
     goes empty and you can see it did. */
  const caps = LIB.caps||{};
  const order = Object.keys(caps).sort((a,b)=>caps[b]-caps[a]);
  document.getElementById('mcaps').innerHTML = order.map(c =>
    `<button class="chip cap${mCaps.has(c)?' on':''}" onclick="toggleCap('${esc(c)}')"
      >${esc(c)} <span class="muted">${caps[c]}</span></button>`).join('')
    || '<span class="note" style="margin:0">No capability badges parsed — press Sync.</span>';

  document.getElementById('mflags').innerHTML = FLAGS.map(([k,lab]) =>
    `<button class="chip${mFlags.has(k)?' on':''}" onclick="toggleFlag('${k}')">${esc(lab)}</button>`
  ).join('')
   + ((mFlags.size||mCaps.size||q)
      ? ` <button class="chip" onclick="clearFilters()">× clear</button>` : '');

  if(!list.length){
    g.innerHTML = '<div class="mempty">Nothing matches. Clear a filter or widen the search.</div>';
  } else {
    g.innerHTML = list.map(cardHtml).join('');
  }

  const B = fitBudget();
  document.getElementById('mfoot').innerHTML =
    `${list.length} of ${(LIB.models||[]).length} models`
    + ` · fit is <b>estimated</b> from the published parameter size at q4 against `
    + `${B.n} × ${B.per.toFixed(0)} GiB and ${Math.round(catCtx/1024)}K of context `
    + `(about ${B.one.toFixed(0)}B on one card). Open a model for real byte sizes from the registry.`;

  document.getElementById('libcount').textContent = (LIB.count||0) + ' models';
  const age = LIB.age_hours;
  const when = age==null ? 'never synced'
    : age<1.5 ? 'synced just now'
    : age<48  ? 'synced '+Math.round(age)+' hours ago'
    : 'synced '+Math.round(age/24)+' days ago';
  let note = `Discovered from <a class="lk" href="https://ollama.com/library" target="_blank" rel="noopener">ollama.com/library</a>`
    + ` — ${when}. Nothing on this list is written into this app by hand.`;
  if(LIB.problems_n)
    note += ` <span style="color:var(--amb)">${LIB.problems_n} record${LIB.problems_n===1?'':'s'} `
          + `parsed oddly: ${esc((LIB.problems||[]).slice(0,3).join('; '))}${LIB.problems_n>3?'…':''}</span>`;
  if(LIB.installed_error)
    note += ` <span style="color:var(--amb)">Ollama did not answer, so the installed badges are missing: ${esc(LIB.installed_error)}</span>`;
  const nEl = document.getElementById('libnote');
  if(!nEl.dataset.sticky) nEl.innerHTML = note;

  // The cards are on the page now, so the chip budget can stop being a guess.
  if(syncChipBudget()) renderCatalog();
}

/* One row of chips, never two.

   Counting chips does not work, because they are not the same width: a card
   with `thinking` + `embedding` + a seven-way size list overflows at the same
   count that `tools` + `3b` sits comfortably inside. So budget by width. The
   34 + 6.6*len estimate is padding + border + gap plus the average advance of
   the 11px UI font -- it does not have to be exact, only conservative, and it
   was checked against the nine cards that were wrapping.

   Capabilities go first because they change what a model is FOR; sizes fill
   whatever room is left. Room for the "+n" chip is reserved up front whenever
   anything is going to be left over, so adding it can never itself overflow. */
let CHIP_BUDGET = 292;     // px of usable width in a card's chip row
const CHIP_MORE   = 50;    // px reserved for "+12"
/* The starting figure was measured at one window size; the grid is responsive,
   so read the real width back off the page and redraw if it differs. A
   .pillrow is a full-width flex container -- its width does not depend on how
   many chips are in it -- so this converges after one extra pass and cannot
   oscillate. */
function syncChipBudget(){
  let changed = false;
  const row = document.querySelector('#mgrid .pillrow');
  if(!row) return false;
  const w = Math.floor(row.getBoundingClientRect().width) - 8;
  if(w > 60 && Math.abs(w - CHIP_BUDGET) > 4){ CHIP_BUDGET = w; changed = true; }
  if(chipMetrics()) changed = true;
  return changed;
}

/* Guessing text width from a per-character average was wrong by enough to wrap
   a row: "embedding" is nine narrow-looking characters that are not narrow.
   Measure the real fonts instead -- one canvas, reading from chips that are
   already on the page, so it picks up whatever the browser actually resolved
   from each font stack. Capability chips and size chips are measured
   separately because the size chips are monospace and the capability chips
   are not. */
function chipMetrics(){
  let changed = false;
  for(const k of ['cap','sz']){
    const el = document.querySelector('#mgrid .chip.flat.' + k);
    if(!el) continue;
    const cs = getComputedStyle(el);
    const font = cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily;
    const pad = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight)
              + parseFloat(cs.borderLeftWidth) + parseFloat(cs.borderRightWidth)
              + 6;   // the flex gap that follows it
    if(CHIP_M[k] && CHIP_M[k].font === font
       && Math.abs(CHIP_M[k].pad - pad) < 0.5) continue;
    CHIP_M[k] = {font: font, pad: pad};
    changed = true;
  }
  if(changed && !CHIP_CTX)
    CHIP_CTX = document.createElement('canvas').getContext('2d');
  return changed;
}
let CHIP_RSZ = null;
window.addEventListener('resize', ()=>{
  clearTimeout(CHIP_RSZ);
  CHIP_RSZ = setTimeout(()=>{ if(LIB) renderCatalog(); }, 200);
});
let CHIP_CTX = null;
const CHIP_M = {};
function chipW(kind, s){
  s = String(s);
  const m = CHIP_M[kind];
  if(CHIP_CTX && m){
    CHIP_CTX.font = m.font;
    return CHIP_CTX.measureText(s).width + m.pad + 1;
  }
  return 34 + 7.2*s.length;   // before the first paint there is nothing to measure
}
function chipRow(m){
  const caps = m.capabilities||[], sizes = m.sizes||[];
  const all = caps.map(c=>['cap',c]).concat(sizes.map(s=>['sz',s]));
  let used = 0, take = 0;
  for(let i=0;i<all.length;i++){
    const w = chipW(all[i][0], all[i][1]);
    // If anything would remain after this one, keep room for the +n chip.
    const need = (i < all.length-1) ? w + CHIP_MORE : w;
    if(used + need > CHIP_BUDGET) break;
    used += w; take++;
  }
  const hidden = all.length - take;
  return all.slice(0,take).map(([k,v])=>`<span class="chip flat ${k}">${esc(v)}</span>`)
    .concat(hidden? [`<span class="chip flat more">+${hidden}</span>`] : []);
}

/* One library record, reduced to a card. The fit badge is the loudest thing on
   it because it is the only thing here ollama.com cannot tell him. */
function cardHtml(m){
  const f = fitOf(m);
  const v = FITV[f.k];
  const chips = chipRow(m);
  const inst = LIBINST.has(m.repo);
  const cur  = LIBCUR.has(m.repo);
  return `<div class="mcard v${v}" onclick="openFamily('${esc(m.repo)}')"
      title="${esc(f.why)}">
    <div class="hd">
      <div><div class="nm">${esc(m.name)}</div>
        <div class="vn"><code>${esc(m.repo)}</code></div></div>
      <div class="rt">
        <span class="fitb v${v}">${esc(f.t)}</span>
        ${(inst||cur)? `<div class="rt2">
          ${inst? '<span class="instb">INSTALLED</span>':''}
          ${cur?  '<span class="curb">NOTES</span>':''}
        </div>`:''}
      </div>
    </div>
    <div class="bl">${esc(m.description||'')}</div>
    <div class="pillrow">${chips.join('')}</div>
    <div class="ft">
      <span class="stat">
        <span>${fmtPulls(m.pulls)} pulls</span>
        <span>${m.tag_count||0} tag${m.tag_count===1?'':'s'}</span>
        <span>${esc(m.updated_rel||'')}</span>
      </span>
      <span style="color:var(--accent)">details →</span>
    </div>
  </div>`;
}

/* Re-scrape ollama.com. A refusal comes back 409 with the reason, and the
   reason is shown verbatim because it is the interesting part: it means the
   parse produced far fewer models than last time, which almost always means
   their markup changed and these regexes need looking at, not that the library
   shrank. "Sync anyway" is there for the day it really did shrink. */
async function libSync(force){
  const b = document.getElementById('libsync');
  const n = document.getElementById('libnote');
  b.disabled = true; b.textContent = '↻ Syncing…';
  n.dataset.sticky = '1';
  n.innerHTML = 'Fetching ollama.com/library…';
  let d;
  try{
    const r = await fetch('/api/library/sync'+(force?'?force=1':''), {method:'POST'});
    d = await r.json();
  }catch(e){ d = {ok:false, message:String(e)}; }
  b.disabled = false; b.textContent = '↻ Sync from ollama.com';
  if(d.ok){
    delete n.dataset.sticky;
    await loadLibrary();
    n.dataset.sticky = '1';
    n.innerHTML = `<span style="color:var(--grn)">Synced — ${d.count} models`
      + `${d.problems_n? ', '+d.problems_n+' parsed oddly':''}.</span>`;
    setTimeout(()=>{ delete n.dataset.sticky; renderCatalog(); }, 6000);
  } else {
    n.innerHTML = `<span style="color:var(--amb)">Sync refused: ${esc(d.message||'no reason given')}</span>`
      + ` <button class="chip" onclick="libSync(1)">Sync anyway</button>`
      + ` <button class="chip" onclick="delete this.parentNode.dataset.sticky;renderCatalog()">dismiss</button>`;
  }
}

/* ---- the detail sheet ----
   An in-page overlay, not a dialog: confirm() and alert() block the event loop
   and this page polls on a timer. Esc and a backdrop click close it. */
async function openFamily(repo, all){
  const ovl = document.getElementById('msheet');
  const body = document.getElementById('msheetbody');
  mOpen = repo;
  ovl.hidden = false;
  document.body.style.overflow = 'hidden';
  body.innerHTML = `<div class="sh"><div class="nm">${esc(repo)}</div>
    <button class="x" onclick="closeFamily()" title="Close (Esc)">×</button></div>
    <div class="note">Asking registry.ollama.ai for every tag and sizing each one…</div>`;
  let d;
  try{
    d = await (await fetch('/api/catalog/family?repo='+encodeURIComponent(repo)
        + '&ctx='+catCtx + reserveArg() + (all?'&all=1':''))).json();
  }catch(e){ d = {error:String(e)}; }
  if(mOpen !== repo) return;            // he moved on while we were fetching
  if(d.error){
    body.innerHTML = `<div class="sh"><div class="nm">${esc(repo)}</div>
      <button class="x" onclick="closeFamily()">×</button></div>
      <div class="note" style="color:var(--amb)">${esc(d.error)}</div>`;
    return;
  }
  mData = {repo:repo, all:all, d:d};
  mTagQ = '';
  renderSheet();
}

/* The sheet is drawn in two pieces so typing in the tag filter re-draws only
   the table. Re-rendering the whole sheet on every keystroke would take the
   focus out of the box you are typing in. */
function renderSheet(){
  if(!mData) return;
  const repo = mData.repo, d = mData.d;
  /* The server widens on its own for a repo with no hand-written notes, so
     trust its answer over what we asked for. */
  const all = (d.all !== undefined) ? d.all : mData.all;
  const m = d.meta||{};
  const L = m.library||{};
  const body = document.getElementById('msheetbody');
  const lib = LIB && (LIB.models||[]).filter(x=>x.repo===repo)[0];
  const est = lib ? fitOf(lib) : null;
  const chips = [m.moe? 'MoE':null, m.ctx? Math.round(m.ctx/1024)+'K ctx':null, m.license]
    .filter(Boolean)
    .concat(L.capabilities||[])
    .concat(m.tasks||[]);
  const stats = [
    L.pulls!=null? fmtPulls(L.pulls)+' pulls' : null,
    L.tag_count!=null? L.tag_count+' tags on ollama.com' : null,
    L.updated_rel? 'updated '+L.updated_rel : null
  ].filter(Boolean).join(' · ');
  body.innerHTML = `
    <div class="sh">
      <div>
        <div class="nm">${esc(m.name||repo)}</div>
        <div class="vn note" style="margin:2px 0 0">
          ${m.vendor? esc(m.vendor)+' · ':''}<code>${esc(repo)}</code>
          ${L.url? ` · <a class="lk" href="${esc(L.url)}" target="_blank" rel="noopener">ollama.com ↗</a>`:''}
        </div>
      </div>
      <button class="x" onclick="closeFamily()" title="Close (Esc)">×</button>
    </div>
    <div class="note">${esc(m.blurb||'')}</div>
    ${stats? `<div class="note" style="margin-top:6px">${esc(stats)}</div>`:''}
    ${m.curated? '' : `<div class="note" style="margin-top:6px;color:var(--mut)">
        This one has no hand-written notes in this app — everything below is straight from the
        registry, which is why there is no licence or context line.</div>`}
    <div class="pillrow" style="margin-top:10px">
      ${chips.map(c=>`<span class="chip flat">${esc(c)}</span>`).join('')}</div>
    ${est? `<div class="note" style="margin-top:8px">The card said <b>${esc(est.t)}</b>; that was an
       estimate from the parameter size. The verdicts below are the real thing — actual bytes from
       the manifest, against the cards in this machine.</div>`:''}
    <div class="ftools">
      <input type="search" id="ftq" placeholder="Filter ${d.tags.length} tags…"
             value="${esc(mTagQ)}" oninput="mTagQ=this.value;renderSheetTags()" autocomplete="off">
      <button class="chip${mFitOnly?' on':''}" onclick="mFitOnly=!mFitOnly;renderSheet()">Only tags that fit</button>
      ${d.blocked? `<button class="chip${mBlocked?' on':''}" onclick="mBlocked=!mBlocked;renderSheet()">`
        + `${mBlocked?'Hiding':'Showing'} the ${d.blocked} format${d.blocked===1?'':'s'} gfx1030 cannot run</button>`:''}
    </div>
    <div id="sheettags"></div>
    <div class="trow" style="margin-top:12px">
      ${all? `<span class="note" style="margin:8px 0 0">${d.tags.length} tags, `
             + `${d.src==='registry'?'live from registry.ollama.ai':'from the built-in catalog — the registry was unreachable'}</span>`
           : `<button class="ghost" onclick="openFamily('${esc(repo)}',1)">Show every tag from the registry</button>`}
    </div>`;
  renderSheetTags();
}

/* The filter exists because llama3.1 publishes 93 tags. A wall of 93 rows is
   not a model browser, it is a log file. */
function renderSheetTags(){
  const el = document.getElementById('sheettags');
  if(!el || !mData) return;
  const repo = mData.repo, d = mData.d;
  const q = (mTagQ||'').trim().toLowerCase();
  let shown = d.tags;
  if(!mBlocked) shown = shown.filter(t=>t.warn!=='bad');
  if(mFitOnly)  shown = shown.filter(t=>{
    const v=(t.fit||{}).verdict; return v==='one_card'||v==='one_card_tight'; });
  if(q) shown = shown.filter(t=>
    (t.tag+' '+(t.quant||'')+' '+(t.note||'')).toLowerCase().includes(q));
  const hid = d.tags.length - shown.length;
  el.innerHTML = `<table class="mt">
      <thead><tr><th>Tag</th><th>Params</th><th>Size</th><th>Verdict</th><th></th></tr></thead>
      <tbody>${shown.map(t=>ftagRow(repo,t)).join('')
        || '<tr><td colspan="5" class="note">Nothing matches that filter.</td></tr>'}</tbody>
    </table>`
    + (hid>0? `<div class="note" style="margin-top:6px">${hid} tag${hid===1?'':'s'} hidden by the filters above.</div>`:'');
}

/* A detail row. Wider than the card: params, active params and quantisation
   are what you compare tags of one model on, and they only exist in the name. */
function ftagRow(repo, t){
  const fit = t.fit||{};
  const [vl,vc] = VERD[fit.verdict] || VERD.unknown;
  const bad = t.warn==='bad';
  const srcPill = t.src==='registry' ? '<span class="pill ok">registry</span>'
                : t.src==='catalog'  ? '<span class="pill guess">catalog</span>' : '';
  const warn = t.warn_why
    ? `<div class="note" style="margin:3px 0 0;color:${bad?'var(--red)':'var(--amb)'}">`
      + `${bad?'✕ ':'⚠ '}${esc(t.warn_why)}</div>` : '';
  const note = t.note ? `<div class="note" style="margin:3px 0 0">${esc(t.note)}</div>` : '';
  const params = t.params_b
    ? t.params_b+'B' + (t.active_b? ` <span class="muted">${t.active_b}B active</span>`:'')
    : '—';
  const btn = bad
    ? '<button class="ghost" disabled title="This format cannot run on gfx1030.">—</button>'
    : `<button class="ghost" onclick="event.stopPropagation();queuePull('${esc(repo)}:${esc(t.tag)}')">Pull</button>`;
  return `<tr>
    <td class="tg">${esc(t.tag)}${t.quant? ` <span class="pill">${esc(t.quant)}</span>`:''}${note}${warn}</td>
    <td class="nw">${params}</td>
    <td class="nw">${t.gib!=null? t.gib.toFixed(1)+' GiB':'—'}${srcPill}</td>
    <td class="nw" style="color:${vc};font-weight:600">${vl}
      <div class="note" style="margin:2px 0 0;font-weight:400">
        needs ${fit.need_gib!=null? fit.need_gib+' GiB':'—'}`
      + `${fit.kv_gib!=null? ' (+'+fit.kv_gib+' KV est.)':''}</div></td>
    <td class="nw">${btn}</td></tr>`;
}

function closeFamily(){
  mOpen = null; mData = null;
  document.getElementById('msheet').hidden = true;
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e=>{ if(e.key==='Escape' && mOpen) closeFamily(); });

/* Pull the library index off disk. No network in this call -- the scrape is a
   separate, explicit thing (the Sync button), because a page load should never
   depend on ollama.com being up. */
async function loadLibrary(){
  const g = document.getElementById('mgrid');
  try{
    LIB = await (await fetch('/api/library')).json();
  }catch(e){
    if(g) g.innerHTML = '<div class="mempty" style="color:var(--amb)">Could not load the library: '
      + esc(String(e)) + '</div>';
    return;
  }
  LIBINST = new Set(LIB.installed||[]);
  LIBCUR  = new Set(LIB.curated||[]);
  if(!LIB.count){
    if(g) g.innerHTML = '<div class="mempty">Nothing cached yet. Press '
      + '<b>Sync from ollama.com</b> above and this fills with every model in the library.</div>';
    document.getElementById('libcount').textContent = 'empty';
    document.getElementById('libnote').innerHTML =
      'No catalog on disk yet. The Sync button scrapes <code>ollama.com/library</code> once and '
      + 'writes <code>/data/catalog.json</code>; after that this page is served from that file.';
    return;
  }
  renderCatalog();
  if(mOpen) openFamily(mOpen, mData && mData.all);
}

async function sizeAny(){
  const v=(document.getElementById('anytag').value||'').trim();
  const out=document.getElementById('anyout');
  if(!v){ out.innerHTML=''; return; }
  const repo=v.split(':')[0], tag=v.includes(':')? v.slice(v.indexOf(':')+1) : 'latest';
  out.innerHTML='<div class="note">checking…</div>';
  const d=await sizeTag(repo,tag,true);
  if(d.error && d.src==='none'){
    out.innerHTML=`<div class="note" style="color:var(--amb)">${esc(d.error)}</div>`; return;
  }
  out.innerHTML='<table class="mt"><tbody>'+tagRow(repo,{tag:tag},d)+'</tbody></table>';
}

async function loadInstalled(){
  try{
    const d=await (await fetch('/api/models/installed')).json();
    const el=document.getElementById('instout');
    if(d.error){ el.innerHTML=`Ollama is not answering: ${esc(d.error)}`; return; }
    if(!d.models.length){ el.textContent='Ollama has no models.'; return; }
    el.innerHTML='<table class="mt"><thead><tr><th>Name</th><th>Size</th><th>Params</th><th>Quant</th></tr></thead><tbody>'
      + d.models.map(m=>`<tr><td class="tg">${esc(m.name)}</td><td class="nw">${fmtB(m.size)}</td>`
        + `<td class="nw">${esc(m.params||'—')}</td><td class="nw">${esc(m.quant||'—')}</td></tr>`).join('')
      + '</tbody></table>';
    /* This function already runs when a pull lands. Recompute the INSTALLED
       badges from the same answer rather than asking the server again -- the
       repo key is the model name with the tag and any namespace stripped, the
       same rule _installed_repos() applies on the other side. */
    const was = LIBINST.size;
    LIBINST = new Set(d.models.map(m=>String(m.name).split(':')[0].split('/').pop()));
    if(LIB && was!==LIBINST.size) renderCatalog();
  }catch(e){}
}

/* ---- pulls ---- */
function queuePull(tag){
  document.getElementById('pulltag').value = tag;
  showTab('pulls');
  startPull();
}
async function startPull(){
  const t=(document.getElementById('pulltag').value||'').trim();
  const err=document.getElementById('pullerr');
  err.textContent='';
  if(!t){ err.textContent='Type a model tag first.'; return; }
  try{
    const r=await fetch('/api/pull/start',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({model:t})});
    const d=await r.json();
    if(!d.ok) err.textContent = d.msg || 'could not start';
    renderPulls(d.pulls);
  }catch(e){ err.textContent=String(e); }
}
async function cancelPull(t){
  try{
    const r=await fetch('/api/pull/cancel',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({model:t})});
    renderPulls((await r.json()).pulls);
  }catch(e){}
}
async function forgetPull(t){
  try{
    const r=await fetch('/api/pull/forget',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({model:t})});
    renderPulls((await r.json()).pulls);
  }catch(e){}
}
async function delPartial(n){
  const el=document.getElementById('partout');
  try{
    const r=await fetch('/api/pull/orphans/delete',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:n})});
    const d=await r.json();
    if(d.error){ el.insertAdjacentHTML('afterbegin',
      `<div class="note" style="color:var(--amb)">${esc(d.error)}</div>`); return; }
    renderPartials(d.partials);
  }catch(e){}
}

const PSTATE={queued:['queued','var(--mut)'],running:['downloading','var(--accent)'],
  done:['complete','var(--grn)'],error:['failed','var(--red)'],cancelled:['cancelled','var(--amb)']};
function renderPulls(pulls, elId, emptyMsg){
  /* Takes a target element because two tabs draw the same downloads: the Pulls
     tab draws all of them, the HF tab draws only the hf.co ones next to the
     repo you just picked from. One renderer, so the two can never disagree. */
  const el=document.getElementById(elId||'pullout');
  if(!el) return;
  if(!pulls || !pulls.length){ el.innerHTML='<div class="note" style="margin:0">'
    + (emptyMsg||'Nothing downloading. Nothing has been downloaded this session.')
    + '</div>'; return; }
  el.innerHTML = pulls.map(p=>{
    const [lbl,col]=PSTATE[p.state]||['?','var(--mut)'];
    const pct = p.total? (100*p.completed/p.total) : 0;
    const bars = (p.layers||[]).filter(l=>l.total>0).map(l=>{
      const lp = 100*l.completed/l.total;
      return `<div class="ph${lp>=99.9?' done':''}"><span class="nm">${esc(l.digest)}</span>`
        + `<span class="tr"><span class="fl" style="width:${lp.toFixed(1)}%"></span></span>`
        + `<span class="tm">${fmtB(l.total)}</span></div>`;
    }).join('');
    let meta = [];
    if(p.state==='running'){
      /* Elapsed is shown WHILE running, not only once it is over. A download
         with no visible clock on it is the thing you end up timing by hand. */
      if(p.elapsed!=null) meta.push(fmtDur(p.elapsed)+' so far');
      if(p.rate) meta.push(fmtB(p.rate)+'/s');
      if(p.eta!=null) meta.push('~'+fmtDur(p.eta)+' left');
      if(p.quiet_s!=null && p.quiet_s>45) meta.push(
        `<span style="color:var(--amb)">no data for ${fmtDur(p.quiet_s)}</span>`);
    } else if(p.elapsed!=null) meta.push(fmtDur(p.elapsed));
    const btn = (p.state==='running'||p.state==='queued')
      ? `<button class="ghost" onclick="cancelPull('${esc(p.tag)}')">Stop</button>`
      : `<button class="ghost" onclick="forgetPull('${esc(p.tag)}')">Clear</button>`;
    return `<div style="margin-bottom:16px">
      <div class="row" style="gap:12px;align-items:center">
        <b style="font-size:15px">${esc(p.tag)}</b>
        <span style="color:${col};font-size:13px">${lbl}</span>
        <span class="note" style="margin:0" title="${esc(p.status||'')}">${esc(stageText(p))}</span>
        <span style="flex:1"></span>${btn}</div>
      <div class="ph" style="margin-top:8px"><span class="nm">total</span>
        <span class="tr"><span class="fl" style="width:${pct.toFixed(1)}%;background:${col}"></span></span>
        <span class="tm">${pct.toFixed(0)}%</span></div>
      <div class="note" style="margin:2px 0 0">${fmtB(p.completed)} of ${fmtB(p.total)}${meta.length?' · '+meta.join(' · '):''}</div>
      ${p.error? `<div class="note" style="color:var(--red)">${esc(p.error)}</div>`:''}
      ${bars}
    </div>`;
  }).join('');
}

function renderPartials(parts, disk){
  const el=document.getElementById('partout');
  if(disk && disk.mounted===false){
    el.innerHTML = `Ollama's models directory is not mounted here, so this app cannot see the `
      + `blob store. Add <code>- /path/to/ollama/models:/ollama-models</code> to the compose `
      + `(read-write — deleting a dead partial is the point) and everything else on this tab `
      + `keeps working either way.`;
    return;
  }
  if(!parts || !parts.length){ el.innerHTML='No half-finished downloads. Nothing to reclaim.'; return; }
  const tot = parts.reduce((a,p)=>a+p.on_disk,0);
  el.innerHTML = '<table class="mt"><thead><tr><th>Blob</th><th>On disk</th><th>Target</th>'
    + '<th>Progress</th><th>Idle</th><th></th></tr></thead><tbody>'
    + parts.map(p=>{
        const pct = p.target? 100*p.on_disk/p.target : 0;
        const stale = p.age_s > 600;
        return `<tr><td class="tg">${esc(p.digest)}…</td>`
          + `<td class="nw"><b>${fmtB(p.on_disk)}</b></td>`
          + `<td class="nw" style="color:var(--mut)">${fmtB(p.target)}</td>`
          + `<td class="nw">${pct.toFixed(0)}%</td>`
          + `<td class="nw" style="color:${stale?'var(--amb)':'var(--mut)'}">${fmtDur(p.age_s)}</td>`
          + `<td class="nw"><button class="ghost" onclick="delPartial('${esc(p.name)}')">Delete</button></td></tr>`;
      }).join('')
    + '</tbody></table>'
    + `<div class="note" style="margin:8px 0 0">${parts.length} partial${parts.length>1?'s':''}, `
    + `<b>${fmtB(tot)}</b> allocated. Deleting one you still want costs you the re-download; `
    + `deleting one you have changed your mind about is free.</div>`;
}

let pullTimer=null, pullFast=null, pullDone=null;
async function pullTick(){
  try{
    const d=await (await fetch('/api/pull')).json();
    renderPulls(d.pulls);
    renderPartials(d.partials, d.disk);
    /* The HF tab's own progress block and its alias picker ride on this same
       poll rather than running a second timer against the same endpoint. */
    const hfp=(d.pulls||[]).filter(p=>p.tag.indexOf('hf.co/')===0);
    if(document.getElementById('hfout')){
      const panel=document.getElementById('hfprogpanel');
      if(panel) panel.style.display = hfp.length? '' : 'none';
      if(hfp.length) renderPulls(hfp,'hfout');
      hfAliasOptions(hfp);
    }

    /* When a pull finishes, the installed-models table is stale and nothing
       refetched it — which is why a completed download only showed up after a
       manual page refresh. Watch for the queued/running -> done transition
       rather than polling /api/models/installed on a timer: that list only
       changes when a pull lands, so a timer would be all cost and no signal.

       pullDone starts null so the FIRST tick only records what is already
       finished. Without that, every page load would fire a spurious refresh
       for downloads that completed hours ago. */
    const done=new Set((d.pulls||[]).filter(p=>p.state==='done').map(p=>p.tag));
    if(pullDone===null){ pullDone=done; }
    else{
      let fresh=false;
      done.forEach(t=>{ if(!pullDone.has(t)) fresh=true; });
      pullDone=done;
      if(fresh) loadInstalled();
    }

    /* A running download should be visible from any tab. You should not have to
       be looking at the right page to know something is using the uplink. */
    const run=(d.pulls||[]).filter(p=>p.state==='running');
    const f=document.getElementById('pullflag');
    if(f) f.textContent = run.length
      ? '▼ ' + run[0].tag + (run[0].total? ' · '+Math.round(100*run[0].completed/run[0].total)+'%' : '')
      : '';
  }catch(e){}
}
function pullPoll(on){
  /* This loop never stops, it only changes pace: 1.5s while you are watching
     the Pulls tab, 6s otherwise. It used to be switched off entirely when you
     left that tab, which broke the two things above that are deliberately NOT
     on the Pulls tab — the header download flag, and refreshing the installed
     list when a download lands while you are looking at the Models tab. */
  if(pullTimer!==null && pullFast===on) return;
  if(pullTimer!==null) clearInterval(pullTimer);
  pullFast=on;
  pullTick();
  pullTimer=setInterval(pullTick, on?1500:6000);
}

let modelsLoaded=false;
async function loadModelsTab(){
  if(modelsLoaded) return;
  modelsLoaded=true;
  await loadHw();
  loadInstalled();
  await loadLibrary();
}

/* ---------------- BMC fan mode: the panic switch ----------------
   Independent of gpu-fan-control on purpose: this has to work when the daemon
   is stopped, wedged, or was never installed. */
let fanModeBusy=false, fanModeMsg='';
function renderFanMode(m){
  const box=document.getElementById('fanmode');
  if(!box) return;
  if(!m || m.mode==null){
    box.innerHTML=`<div class="bad">Can't read the BMC fan mode`+
      (m&&m.error?': '+esc(m.error):'')+
      `. This panel needs <code>/dev/ipmi0</code> mapped into the container.</div>`;
    return;
  }
  const full = (m.mode==='full');
  const rpm  = (m.rpm==null)?'—':Math.round(m.rpm).toLocaleString();
  const hot  = (m.hottest_c==null)?'—':(Math.round(m.hottest_c)+' °C');
  let h=`<div class="fanrow" style="align-items:center;gap:14px;flex-wrap:wrap">
      <span style="background:${full?'#5a1b1b':'#16351f'};color:#fff;padding:7px 14px;
                   border-radius:16px;font-weight:700;letter-spacing:.3px">
        ${full?'🔊 FANS AT MAXIMUM':'🔇 BMC quiet curve'}
      </span>
      <span class="muted">fan wall <b>${rpm}</b> RPM · hottest junction <b>${hot}</b></span>
      ${full
        ? `<button class="ghost" ${fanModeBusy?'disabled':''}
             onclick="setFanMode('standard')">↩ Back to quiet</button>`
        : `<button class="ghost" ${fanModeBusy?'disabled':''}
             style="border-color:#a33;color:#f99;font-weight:700"
             onclick="setFanMode('full')">🔥 MAX FANS</button>`}
    </div>`;
  if(m.latched_suspect){
    h+=`<div class="bad">Mode reads "standard" but the wall is still at ${rpm} RPM.
        That is the BMC's fan-fault latch: it will keep accepting mode commands and
        ignoring every one. Clear it with <code>ipmitool mc reset cold</code>
        (~90 s, the host keeps running).</div>`;
  }
  h+=`<div class="note">Binary control — this chassis has no intermediate fan speed
      (measured 2026-08-23). Spin-up to full takes about 40 s, so RPM lags the button.</div>`;
  if(fanModeMsg) h+=`<div class="note">${esc(fanModeMsg)}</div>`;
  box.innerHTML=h;
}
async function setFanMode(mode, force){
  if(fanModeBusy) return;
  fanModeBusy=true; fanModeMsg=(mode==='full'?'going loud…':'handing back to the BMC…');
  try{
    const r=await fetch('/api/fanmode',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode:mode,force:!!force})});
    const d=await r.json();
    if(d.needs_force){
      fanModeBusy=false;
      if(confirm(d.error)){ return setFanMode(mode,true); }
      fanModeMsg=''; tick(); return;
    }
    fanModeMsg = d.ok
      ? ('mode is now '+d.mode+' — RPM will take ~40 s to follow')
      : ('⚠ '+(d.error||'the BMC accepted the command but the mode did not change'));
  }catch(e){ fanModeMsg='⚠ '+e; }
  fanModeBusy=false; tick();
}

/* ---------------- fan control ---------------- */
let fanBusy=false, fanMsg='';
function renderFanCtl(f){
  const box=document.getElementById('fanctl');
  if(!f||!f.status){
    box.innerHTML='<span class="muted">gpu-fan-control isn\'t reporting.</span>'+
      '<div class="note">This panel needs the daemon\'s folder mounted here as <code>/fanctl</code> and gpu-fan-control v4+ running. '+
      'Until then the BMC (or an older daemon) is driving the fans and nothing here will change them.</div>';
    return;
  }
  const s=f.status;
  const presets=(f.presets||[]);
  let h='';
  h+=`<div class="row" style="align-items:flex-end">
        <div class="item"><span class="k">commanded duty</span><span class="duty">${s.duty==null?'—':s.duty}<span class="u"> %</span></span></div>
        <div class="item"><span class="k">preset</span><span class="v" style="font-size:20px">${esc(s.preset||'—')}</span></div>
        <div class="item"><span class="k">floor</span><span class="v" style="font-size:20px">${s.min_duty}<span class="u"> %</span></span></div>
        <div class="item"><span class="k">hysteresis</span><span class="v" style="font-size:20px">${s.hyst}<span class="u"> °C</span></span></div>
        <div class="item"><span class="k">GPUs seen</span><span class="v" style="font-size:20px">${s.gpu_count}<span class="u"> / ${s.gpu_expected}</span></span></div>
      </div>`;
  if(s.stale) h+=`<div class="bad">Status is ${s.age==null?'?':Math.round(s.age)}s old — the daemon looks stopped. The BMC may still be latched in manual mode at ${s.duty}%. Restart gpu-fan-control, or run <code>ipmitool raw 0x30 0x45 0x01 0x02</code> to hand cooling back to the BMC.</div>`;
  else if(s.failsafe) h+=`<div class="bad">FAILSAFE — only ${s.gpu_count} of ${s.gpu_expected} GPUs are reporting. Fans pinned at ${s.failsafe_duty}% until the missing card comes back.</div>`;
  else if(s.mode!=='01') h+=`<div class="warn">BMC fan mode is "${esc(s.mode)}", not manual — the daemon will re-assert it on the next cycle.</div>`;
  if(s.min_duty<40) h+=`<div class="warn">Floor is ${s.min_duty}%. Below about 40% a fan can drop under the BMC's Lower Critical threshold, which makes it declare a fan failure and blast every zone to 100% for a few seconds. Check <code>ipmitool sensor | grep -i fan</code> before leaving it here.</div>`;

  if(!f.writable){
    h+=`<div class="warn">fan-curve.conf isn't writable from here, so these controls are read-only. Mount the gpu-fan-control folder read-write as <code>/fanctl</code>.</div>`;
  }
  const dis=f.writable?'':'disabled';
  if(s.disabled){
    h+=`<div class="warn">Daemon is <b>DISABLED</b> — fans are on the BMC's own curve and the cards are uncapped. It is still watching (temps below stay live) but will not intervene, even if a card overheats. Pick any preset to re-enable.</div>`;
  }
  if(s.binary_control){
    h+=`<div class="note">This chassis has binary fan control (BMC quiet ≈4.9k RPM / Full ≈23.4k — the duty command does nothing, measured 2026-08-23). Presets bundle a per-card power cap with fan thresholds; the silent/quiet presets cap the cards so the fans never need to leave the BMC's curve.</div>`;
  }
  h+=`<div class="row" style="align-items:flex-end">
        <div class="item"><span class="k">power cap</span><span class="v" style="font-size:20px">${s.cap_w==null?'card max':s.cap_w+'<span class="u"> W</span>'}</span></div>
        <div class="item"><span class="k">hottest draw</span><span class="v" style="font-size:20px">${s.gpu_max_power_w==null?'—':s.gpu_max_power_w}<span class="u"> W</span></span></div>
        <div class="item"><span class="k">fan trigger</span><span class="v" style="font-size:20px">${s.pwr_on_w?('&gt;'+s.pwr_on_w+'<span class="u"> W</span>'):'temp only'}</span></div>
      </div>`;
  h+=`<div class="fanrow">
        <span class="lbl">power limit W</span><input type="number" id="f_cap" min="90" max="300" step="5" placeholder="${s.cap_w==null?'max':s.cap_w}" ${dis}>
        <button class="ghost" ${dis} onclick="setFan({cap_w:v('f_cap')})">Apply cap</button>
        <button class="ghost" ${dis} onclick="setFan({cap_w:0})">Uncap</button>
        <span class="lbl" style="margin-left:16px"></span>
        ${s.disabled
          ? `<button class="ghost" ${dis} onclick="setFan({disabled:0})">▶ Enable daemon</button>`
          : `<button class="ghost" ${dis} onclick="if(confirm('Hand fans back to the BMC and uncap the cards? The daemon stops intervening entirely — under sustained load the cards WILL run hot (102°C was measured this way).')) setFan({disabled:1})">⏻ Disable daemon</button>`}
      </div>`;
  h+=`<div class="fanrow"><span class="lbl">preset</span>`+
     presets.map(p=>`<button class="ghost ${p===s.preset?'sel':''}" ${dis} onclick="setFan({preset:'${p}'})">${p}</button>`).join('')+
     `</div>`;
  h+=`<div class="fanrow">
        <span class="lbl">floor %</span><input type="number" id="f_min" min="20" max="100" value="${s.min_duty}" ${dis}>
        <span class="lbl">hysteresis °C</span><input type="number" id="f_hyst" min="0" max="25" value="${s.hyst}" ${dis}>
        <span class="lbl">failsafe %</span><input type="number" id="f_fs" min="0" max="100" value="${s.failsafe_duty}" ${dis}>
        <button class="ghost" ${dis} onclick="setFan({min_duty:v('f_min'),hyst:v('f_hyst'),failsafe_duty:v('f_fs')})">Apply</button>
      </div>`;
  h+=`<details><summary>Custom curves (temp:duty pairs — highest matching duty wins, floor applies underneath)</summary>
       <div class="fanrow"><span class="lbl">GPU</span><input type="text" id="f_gc" value="${esc(s.gpu_curve)}" ${dis}></div>
       <div class="fanrow"><span class="lbl">CPU</span><input type="text" id="f_cc" value="${esc(s.cpu_curve)}" ${dis}></div>
       <div class="fanrow"><button class="ghost" ${dis} onclick="setFan({gpu_curve:v('f_gc'),cpu_curve:v('f_cc')})">Apply curves</button></div>
       <div class="note">GPU curve reads the hottest card's junction temp. Leave a field empty for "floor only".</div>
      </details>`;
  h+=`<div class="note" id="fanmsg">${esc(fanMsg)}</div>`;
  h+=`<div class="note">Changes are written to fan-curve.conf; the daemon picks them up on its next cycle (~${s.interval}s). It re-validates everything, so a bad value here can't reach the BMC.</div>`;
  box.innerHTML=h;
}
function v(id){const e=document.getElementById(id);return e?e.value:null;}
async function setFan(body){
  if(fanBusy)return; fanBusy=true; fanMsg='applying…';
  try{
    const r=await fetch('/api/fan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    fanMsg = d.ok ? ('applied — takes effect within '+d.takes_effect_in_s+'s') : ('⚠ '+d.error);
  }catch(e){ fanMsg='⚠ '+e; }
  fanBusy=false; tick();
}

async function tick(){
  try{
    const r=await fetch('/api/power'); const d=await r.json();
    document.getElementById('cur').textContent = d.current==null?'—':d.current;
    document.getElementById('err').textContent = d.error?('⚠ '+d.error):'';
    samples = d.samples||[];
    renderCpu(d.cpu,d.util); renderGpus(d.gpus);
    renderVram(d.gpumem,d.gpumem_meta);
    renderRam(d.ram);
    applyOllamaMode(d.ollama_enabled);
  renderTest(d.test); renderFans(d.fans);
    try{ const fm=await fetch('/api/fanmode'); renderFanMode(await fm.json()); }
    catch(e){ renderFanMode(null); }
    const fr=await fetch('/api/fan'); renderFanCtl(await fr.json());
    const logging=d.logging;
    document.getElementById('startBtn').style.display = logging?'none':'';
    document.getElementById('stopBtn').style.display  = logging?'':'none';
    document.getElementById('dot').className = 'dot'+(logging?' on':'');
    document.getElementById('stat').textContent = logging?('logging → '+d.logfile):'not logging';
    document.getElementById('tiles').style.display = (logging&&d.stats)?'grid':'none';
    if(logging&&d.stats){
      const s=d.stats;
      document.getElementById('mx').textContent=s.max+' W';
      document.getElementById('um').textContent=s.umax==null?'—':(s.umax+' %');
      const ct=Math.max(s.c1max||0,s.c2max||0)||null;
      document.getElementById('cm').textContent=ct==null?'—':(ct+' °C');
      document.getElementById('gm').textContent=s.gjmax==null?'—':(s.gjmax+' °C');
      document.getElementById('vm').textContent=s.vmax==null?'—':(s.vmax+' GiB');
      document.getElementById('gw').textContent=s.gpwmax==null?'—':(s.gpwmax+' W');
      document.getElementById('fm').textContent=s.fmax==null?'—':(s.fmax+' RPM');
      document.getElementById('fn').textContent=s.fmin==null?'—':(s.fmin+' RPM');
      document.getElementById('mn').textContent=s.min+' W';
      document.getElementById('av').textContent=s.avg+' W';
      document.getElementById('ns').textContent=s.n;
      document.getElementById('el').textContent=fmtEl(s.elapsed);
    }
    draw();
  }catch(e){document.getElementById('err').textContent='⚠ '+e;}
}
function draw(){
  const c=document.getElementById('c'),x=c.getContext('2d');
  const W=c.width,H=c.height,pad=6;x.clearRect(0,0,W,H);
  if(samples.length<2)return;
  const ys=samples.map(s=>s[1]);let lo=Math.min(...ys),hi=Math.max(...ys);
  if(hi-lo<10){hi+=5;lo-=5;} lo=Math.max(0,lo-5);hi+=5;
  const sx=(W-2*pad)/(samples.length-1),sy=(H-2*pad)/(hi-lo);
  x.strokeStyle='#3fb6ff';x.lineWidth=2;x.beginPath();
  samples.forEach((s,i)=>{const px=pad+i*sx,py=H-pad-(s[1]-lo)*sy;i?x.lineTo(px,py):x.moveTo(px,py);});
  x.stroke();x.fillStyle='#3fb6ff22';x.lineTo(pad+(samples.length-1)*sx,H-pad);x.lineTo(pad,H-pad);x.closePath();x.fill();
}
async function startLog(){await fetch('/api/log/start',{method:'POST'});tick();loadLogs();}
async function stopLog(){await fetch('/api/log/stop',{method:'POST'});tick();}

/* ---------------- history chart ----------------
   Series are derived from the CSV header rather than hardcoded, so a log with
   two GPUs and six fans charts every one of them, and an old single-GPU log
   still charts fine.                                                        */
const PALETTE=['#3fb6ff','#e5484d','#2ec26a','#e0a63b','#a06bff','#3fd9d0','#ff8f3f','#8be04e','#f472b6','#60a5fa'];
// The first seven PALETTE entries are already spoken for by the fixed series
// below, so the dynamic per-GPU / per-fan series get their own list. With two
// cards and four fans the overlay carries 14 lines - if any two share a colour
// the chart is useless for exactly the thing it's meant to diagnose.
const DYN=['#f472b6','#8be04e','#c084fc','#fde047','#22d3ee','#fb7185',
           '#4ade80','#93c5fd','#facc15','#a3e635','#f0abfc','#7dd3fc'];
function seriesFor(cols){
  const out=[]; let ci=0;
  const add=(key,label,unit,color)=>{ if(cols.includes(key)) out.push({key,label,unit,color}); };
  add('watts','Power','W','#3fb6ff');
  add('gpu_junction_c','GPU junction (hottest)','°C','#e5484d');
  cols.filter(c=>/^gpu\d+_junction_c$/.test(c)).forEach(c=>{
    out.push({key:c,label:'GPU '+c.match(/\d+/)[0]+' junction',unit:'°C',color:DYN[(ci++)%DYN.length]});});
  add('gpu_edge_c','GPU edge','°C','#2ec26a');
  cols.filter(c=>/^gpu\d+_power_w$/.test(c)).forEach(c=>{
    out.push({key:c,label:'GPU '+c.match(/\d+/)[0]+' board power',unit:'W',color:DYN[(ci++)%DYN.length]});});
  cols.filter(c=>/^gpu\d+_vram_used_mb$/.test(c)).forEach(c=>{
    out.push({key:c,label:'GPU '+c.match(/\d+/)[0]+' VRAM',unit:'MB',color:DYN[(ci++)%DYN.length]});});
  out.push({key:'cpu_max_c',label:'CPU max',unit:'°C',color:'#e0a63b'});
  add('cpu_util_pct','CPU util','%','#a06bff');
  add('fan_duty_pct','Fan duty (commanded)','%','#ffffff');
  add('fan_rpm_max','Fan max','RPM','#3fd9d0');
  add('fan_rpm_min','Fan min (slowest)','RPM','#ff8f3f');
  cols.filter(c=>/_rpm$/.test(c)&&!/^fan_rpm_/.test(c)).forEach(c=>{
    out.push({key:c,label:c.replace(/_rpm$/,'').toUpperCase(),unit:'RPM',color:DYN[(ci++)%DYN.length]});});
  return out;
}
let SERIES=[];
let hist={times:[],labels:[],data:{},avail:[]};
let off=new Set(), hoverIdx=null;

function fillMetricSelect(){
  const sel=document.getElementById('metric'); const prev=sel.value;
  sel.innerHTML='<option value="all">All (overlay)</option>'+
    hist.avail.map(k=>{const s=SERIES.find(x=>x.key===k);
      return `<option value="${k}">${esc(s.label)} (${esc(s.unit)})</option>`;}).join('');
  if(prev && [...sel.options].some(o=>o.value===prev)) sel.value=prev;
  else if(hist.avail.includes('gpu_junction_c')) sel.value='gpu_junction_c';
}
let SRVTZ=null;
/* The offset the container stamped onto the row, or null for a log written
   before TZ was set -- those are bare UTC and should say so rather than be
   silently relabelled as local. */
function tzLabel(ts){
  const m=/([+-])(\d\d):?(\d\d)$/.exec(ts||'');
  return m ? ('UTC'+m[1]+m[2]+':'+m[3]) : null;
}
async function loadLogs(){
  try{
    const r=await fetch('/api/logs'); const d=await r.json();
    SRVTZ = d.tz || null;
    const sel=document.getElementById('logsel'); const prev=sel.value;
    sel.innerHTML = (d.logs||[]).map(l=>{
      const t=new Date(l.mtime*1000).toLocaleString();
      const kb=(l.size/1024).toFixed(0);
      const act=(l.name===d.active)?' · logging now':'';
      return `<option value="${l.name}">${t} (${kb} KB)${act}</option>`;
    }).join('');
    if(!d.logs||!d.logs.length){sel.innerHTML='<option value="">no logs yet — hit Start logging</option>';
      hist={times:[],labels:[],data:{},avail:[]};document.getElementById('hsum').textContent='';drawHistory();return;}
    if(prev && [...sel.options].some(o=>o.value===prev)) sel.value=prev; else sel.selectedIndex=0;
    loadLog(sel.value);
  }catch(e){document.getElementById('hsum').textContent='⚠ '+e;}
}
async function loadLog(name){
  if(!name){return;}
  document.getElementById('dl').href='/api/logs/'+name;
  document.getElementById('dl').setAttribute('download',name);
  try{
    const r=await fetch('/api/logs/'+name); const d=await r.json();
    if(d.error){document.getElementById('hsum').textContent='⚠ '+d.error;return;}
    SERIES=seriesFor(d.cols);
    const ci={}; d.cols.forEach((c,i)=>ci[c]=i);
    const num=(v)=>{const n=parseFloat(v);return isNaN(n)?null:n;};
    const times=[], labels=[], data={}; SERIES.forEach(s=>data[s.key]=[]);
    d.rows.forEach(row=>{
      const ts=row[ci['timestamp']]||'';
      times.push(ts); labels.push(ts.slice(11,19)||ts);
      SERIES.forEach(s=>{
        if(s.key==='cpu_max_c'){
          const c1=num(row[ci['cpu1_c']]),c2=num(row[ci['cpu2_c']]);
          data[s.key].push((c1==null&&c2==null)?null:Math.max(c1==null?-99:c1,c2==null?-99:c2));
        }else{
          data[s.key].push(s.key in ci ? num(row[ci[s.key]]) : null);
        }
      });
    });
    const avail=SERIES.filter(s=>data[s.key].some(v=>v!=null)).map(s=>s.key);
    hist={times,labels,data,avail};
    fillMetricSelect(); renderLegend(); renderSummary(); drawHistory();
  }catch(e){document.getElementById('hsum').textContent='⚠ '+e;}
}
function stat(arr){let mn=null,mx=null,sum=0,n=0;arr.forEach(v=>{if(v!=null){mn=mn==null?v:Math.min(mn,v);mx=mx==null?v:Math.max(mx,v);sum+=v;n++;}});return n?{mn,mx,avg:sum/n,n}:null;}
function renderSummary(){
  const u=s=>s.unit==='°C'?'°C':(' '+s.unit);
  const parts=hist.avail.map(k=>{const s=SERIES.find(x=>x.key===k);const st=stat(hist.data[k]);
    return st?`${s.label}: peak ${Math.round(st.mx)}${u(s)} · avg ${Math.round(st.avg)}${u(s)}`:'';}).filter(Boolean);
  const n=hist.times.length;
  let zone='';
  if(n){
    const z=tzLabel(hist.times[0]);
    if(z) zone='clock '+z;
    else zone='clock not stamped (written as UTC)'
      + (SRVTZ&&SRVTZ.label ? ' · server now '+[SRVTZ.name,SRVTZ.label].filter(Boolean).join(' ') : '');
  }
  document.getElementById('hsum').innerHTML = n?(`${n} samples &nbsp;·&nbsp; ${esc(zone)} &nbsp;·&nbsp; `+parts.join(' &nbsp;|&nbsp; ')):'';
}
function metricMode(){return document.getElementById('metric').value||'all';}
function renderLegend(){
  const box=document.getElementById('legend'); const mode=metricMode();
  if(mode!=='all'){box.innerHTML='';return;}
  box.innerHTML=hist.avail.map(k=>{const s=SERIES.find(x=>x.key===k);
    return `<span class="lg${off.has(k)?' off':''}" onclick="toggleSeries('${k}')"><span class="sw" style="background:${s.color}"></span>${esc(s.label)}</span>`;}).join('');
}
function toggleSeries(k){if(off.has(k))off.delete(k);else off.add(k);renderLegend();drawHistory();}
function niceTicks(lo,hi,n){const span=hi-lo||1;let step=Math.pow(10,Math.floor(Math.log10(span/n)));const err=span/n/step;if(err>=5)step*=5;else if(err>=2)step*=2;const t=[];let v=Math.ceil(lo/step)*step;for(;v<=hi+1e-9;v+=step)t.push(v);return t;}
function drawHistory(){
  const c=document.getElementById('hc'),x=c.getContext('2d');
  const W=c.width,H=c.height,L=48,R=14,T=14,B=30;const pw=W-L-R,ph=H-T-B;
  x.clearRect(0,0,W,H);
  const n=hist.times.length;
  x.strokeStyle='#232a33';x.fillStyle='#8b98a5';x.font='11px system-ui';x.lineWidth=1;
  x.strokeRect(L,T,pw,ph);
  if(n<2){x.fillText('Pick a log with at least a few samples.',L+10,T+20);return;}
  const xat=i=>L+pw*(i/(n-1));
  const xt=Math.min(6,n);
  for(let k=0;k<xt;k++){const i=Math.round(k*(n-1)/(xt-1));const px=xat(i);
    x.strokeStyle='#1b2129';x.beginPath();x.moveTo(px,T);x.lineTo(px,T+ph);x.stroke();
    x.fillStyle='#8b98a5';x.textAlign='center';x.fillText(hist.labels[i],px,H-10);}
  x.textAlign='right';
  const mode=metricMode();
  const drawLine=(vals,lo,hi,color)=>{const yat=v=>T+ph-ph*((v-lo)/((hi-lo)||1));
    x.strokeStyle=color;x.lineWidth=1.8;x.beginPath();let pen=false;
    for(let i=0;i<n;i++){const v=vals[i];if(v==null){pen=false;continue;}const px=xat(i),py=yat(v);
      if(!pen){x.moveTo(px,py);pen=true;}else x.lineTo(px,py);}x.stroke();};
  if(mode==='all'){
    hist.avail.filter(k=>!off.has(k)).forEach(k=>{const s=SERIES.find(x=>x.key===k);const st=stat(hist.data[k]);
      if(st)drawLine(hist.data[k],st.mn,st.mx,s.color);});
    x.fillStyle='#8b98a5';x.textAlign='left';x.fillText('each line auto-scaled — hover for real values',L+6,T+14);
  }else{
    const s=SERIES.find(x2=>x2.key===mode);const st=stat(hist.data[mode]);
    if(!s||!st){x.fillStyle='#8b98a5';x.fillText('no data for this metric in this log',L+10,T+20);return;}
    let lo=st.mn,hi=st.mx;if(hi-lo<1){hi+=1;lo-=1;}const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;
    niceTicks(lo,hi,4).forEach(v=>{const py=T+ph-ph*((v-lo)/((hi-lo)||1));
      x.strokeStyle='#1b2129';x.beginPath();x.moveTo(L,py);x.lineTo(L+pw,py);x.stroke();
      x.fillStyle='#8b98a5';x.textAlign='right';x.fillText(Math.round(v),L-6,py+4);});
    drawLine(hist.data[mode],lo,hi,s.color);
  }
  if(hoverIdx!=null&&hoverIdx>=0&&hoverIdx<n){const px=xat(hoverIdx);
    x.strokeStyle='#4a5663';x.setLineDash([3,3]);x.beginPath();x.moveTo(px,T);x.lineTo(px,T+ph);x.stroke();x.setLineDash([]);
    const keys=(mode==='all')?hist.avail.filter(k=>!off.has(k)):[mode];
    keys.forEach(k=>{const st=stat(hist.data[k]);const v=hist.data[k][hoverIdx];if(v==null||!st)return;
      let lo=st.mn,hi=st.mx;if(mode!=='all'){if(hi-lo<1){hi+=1;lo-=1;}const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;}
      const py=T+ph-ph*((v-lo)/((hi-lo)||1));const s=SERIES.find(x2=>x2.key===k);
      x.fillStyle=s.color;x.beginPath();x.arc(px,py,3,0,7);x.fill();});
  }
}
function hoverMove(ev){
  const c=document.getElementById('hc');const rect=c.getBoundingClientRect();
  const n=hist.times.length;if(n<2){return;}
  const L=48,R=14;const sx=c.width/rect.width;
  const cx=(ev.clientX-rect.left)*sx;const pw=c.width-L-R;
  let i=Math.round((cx-L)/pw*(n-1));i=Math.max(0,Math.min(n-1,i));hoverIdx=i;drawHistory();
  const mode=metricMode();const keys=(mode==='all')?hist.avail.filter(k=>!off.has(k)):[mode];
  const rows=keys.map(k=>{const s=SERIES.find(x=>x.key===k);const v=hist.data[k][i];
    return v==null?'':`<div><span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${s.color};margin-right:5px"></span>${esc(s.label)}: <b style="color:${s.color}">${Math.round(v)}</b> ${esc(s.unit)}</div>`;}).filter(Boolean).join('');
  const tip=document.getElementById('tip');
  tip.innerHTML=`<div style="color:#8b98a5;margin-bottom:3px">${hist.labels[i]}</div>`+rows;
  tip.style.display='block';
  const wrap=c.parentElement.getBoundingClientRect();
  let lx=ev.clientX-wrap.left+14;if(lx+220>wrap.width)lx=ev.clientX-wrap.left-220;
  tip.style.left=Math.max(0,lx)+'px';tip.style.top=(ev.clientY-wrap.top+10)+'px';
}
function hoverOut(){hoverIdx=null;document.getElementById('tip').style.display='none';drawHistory();}
document.getElementById('logsel').addEventListener('change',e=>loadLog(e.target.value));
document.getElementById('metric').addEventListener('change',()=>{renderLegend();drawHistory();});
document.getElementById('hc').addEventListener('mousemove',hoverMove);
document.getElementById('hc').addEventListener('mouseleave',hoverOut);

tick();setInterval(tick,2000);
loadLogs();setInterval(()=>{if(document.getElementById('logsel').value)loadLog(document.getElementById('logsel').value);},15000);
/* Start the pull poller at load rather than on the first tab click. A download
   started in a previous browser session is still running server-side, and the
   header flag should say so before you go looking for it. */
pullPoll(curTab==='pulls');
</script></body></html>"""


# The thermal-test tab, parked. It is substituted into INDEX only when
# SHOW_THERMAL_TEST is set. Keeping it as a string rather than deleting it means
# the feature is one env var away instead of one git archaeology session away,
# and the /api/test routes it drives are untouched either way.
TEST_TAB = """  <button class="tab" id="tab-test" onclick="showTab('test')">Thermal test</button>"""

TEST_PANE = r"""<div id="pane-test" style="display:none">

<div class="panel">
  <div class="ttl">Supervised thermal run</div>
  <div class="note" style="margin-top:0">
    Applies a known load, logs everything at the normal poll rate, and stops the moment
    the machine looks unhappy. Read <b>what this can't do</b> below before trusting a green result.
  </div>

  <div class="trow">
    <label class="fld"><span>Mode</span>
      <select id="tmode" onchange="tmodeChanged()">
        <option value="ollama">Drive load with Ollama</option>
        <option value="observe">Observe only (I'll run the load)</option>
      </select></label>
    <label class="fld" id="tmodelwrap"><span>Model</span>
      <select id="tmodel"></select></label>
    <label class="fld"><span>Abort at junction °C</span>
      <input id="tabort" type="number" min="70" max="99" value="95"></label>
    <label class="fld"><span>Abort at system W</span>
      <input id="twatts" type="number" min="300" max="1600" value="1000"></label>
  </div>
  <div class="trow">
    <label class="fld"><span>Baseline s</span><input id="tbase" type="number" min="10" max="600" value="60"></label>
    <label class="fld"><span>Max load s</span><input id="tload" type="number" min="60" max="7200" value="900"></label>
    <label class="fld"><span>Hold s</span><input id="thold" type="number" min="0" max="3600" value="300"></label>
    <label class="fld"><span>Cooldown s</span><input id="tcool" type="number" min="30" max="3600" value="300"></label>
    <label class="fld" id="tworkwrap"><span>Concurrent requests</span><input id="twork" type="number" min="1" max="8" value="2"></label>
  </div>
  <div class="trow">
    <label class="chk"><input id="tprot" type="checkbox" checked>
      <span>On abort, force the fans to <b>max</b> (restored afterwards)</span></label>
  </div>

  <div class="btns">
    <button class="start" id="trunBtn" onclick="startTest()">▶ Run thermal test</button>
    <button class="stop"  id="tstopBtn" onclick="stopTest()" style="display:none">■ Abort now</button>
    <span class="status"><span class="dot" id="tdot"></span><span id="tstat">not running</span></span>
  </div>
  <div class="err" id="terr"></div>
</div>

<div class="panel" id="tprogpanel" style="display:none">
  <div class="ttl">Run in progress</div>
  <div id="tphases"></div>
  <div id="tpeaks" class="row" style="margin-top:14px"></div>
  <div class="note" id="tlogline"></div>
</div>

<div class="panel" id="tverdictpanel" style="display:none">
  <div class="ttl">Result</div>
  <div id="tverdict"></div>
</div>

<div class="panel">
  <div class="ttl">Run log</div>
  <div id="tnotes"><span class="muted">Nothing yet.</span></div>
</div>

<div class="panel">
  <div class="ttl">What this can't do</div>
  <div class="note" style="margin-top:0">
    <b>It can't choose which card gets the load.</b> Ollama picks its GPU at process start from
    <code>HIP_VISIBLE_DEVICES</code>; there is no per-request control and no API that reports the choice.
    So this isn't a "card 0 then card 1" test — it applies load and reports which cards actually got hot.
    For a single-card run, start Ollama pinned to one card and run it again.<br><br>
    <b>It can't stop a load it didn't start.</b> In observe-only mode the watchdog still runs and still
    maxes the fans, but ending a ComfyUI job is on you.<br><br>
    <b>It can't power-cap the cards.</b> gpu-fan-control owns the card the way it owns the fan zones.
    The only protective action available here is raising the fan preset, by writing the same config file
    the fan panel writes — never by touching <code>/dev/ipmi0</code>.<br><br>
    <b>An Ollama load is not the worst case.</b> Sustained diffusion (ComfyUI/SDXL) is typically hotter.
    A pass here means "survived the load we can actually generate", which is worth knowing and is not
    the same claim as "thermally validated".
  </div>
</div>

</div><!-- /pane-test -->"""


# ---------------- scripts tab ----------------
# Every *.sh / *.py in SCRIPTS_DIR (a HOST path) becomes a button. The
# commands run ON THE HOST via nsenter into PID 1's namespaces. The same
# argument the compose makes for pid:host applies here: this container is
# already privileged, so nsenter grants nothing it did not have - it just
# lets the scripts see the real machine (docker, zfs, the real /mnt), which
# is the entire point of running them.
#
# Output is captured into DATA_DIR/script-logs/<name>.log (the mounted,
# persistent volume), one appended section per run. One instance of each
# script at a time; different scripts may run concurrently.
#
# If ollama-gpu-guard.sh exists in the folder, a background thread runs it
# every SCRIPTS_GUARD_EVERY seconds (default 900, 0 disables) and once at
# startup - so with this container on restart:unless-stopped, the
# every-reboot CPU-fallback problem needs no cron and no Init/Shutdown entry.
SCRIPTS_DIR   = os.environ.get("SCRIPTS_DIR", "/host-scripts")
HOST_EXEC     = os.environ.get("HOST_EXEC", "nsenter -t 1 -m -u -i -n -p --").split()
GUARD_NAME    = "ollama-gpu-guard.sh"
GUARD_EVERY   = int(os.environ.get("SCRIPTS_GUARD_EVERY", "900"))
SCRIPT_LOGS   = os.path.join(os.environ.get("DATA_DIR", "/data"), "script-logs")

_script_jobs  = {}   # name -> {"proc","started","exit"}
_script_lock  = threading.Lock()

def _scripts_list():
    # Gated: the runner executes host files as root. Off unless asked for.
    if not ENABLE_SCRIPTS:
        return []
    try:
        out = subprocess.run(HOST_EXEC + ["ls", "-1", SCRIPTS_DIR],
                             capture_output=True, text=True, timeout=10)
        names = [n for n in out.stdout.split() if n.endswith((".sh", ".py"))]
        return sorted(names)
    except Exception:
        return []

def _script_run(name, args):
    if not ENABLE_SCRIPTS:
        return ("the host-script runner is disabled. Set ENABLE_SCRIPTS=1 in "
                "the compose if you want it — it runs files as root on the host.")
    if name not in _scripts_list():
        return f"{name} is not in {SCRIPTS_DIR}"
    with _script_lock:
        j = _script_jobs.get(name)
        if j and j["proc"] is not None:
            return f"{name} is already running"
        try:
            import shlex as _shlex
            extra = _shlex.split(args or "")
        except ValueError as e:
            return f"bad args: {e}"
        runner = "bash" if name.endswith(".sh") else "python3"
        cmd = HOST_EXEC + [runner, os.path.join(SCRIPTS_DIR, name)] + extra
        os.makedirs(SCRIPT_LOGS, exist_ok=True)
        logf = open(os.path.join(SCRIPT_LOGS, name + ".log"), "a")
        logf.write("\n==== %s  %s ====\n" % (
            datetime.datetime.now().strftime("%F %T"), " ".join(cmd)))
        logf.flush()
        try:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        except OSError as e:
            logf.write("failed to start: %s\n" % e); logf.close()
            return "failed to start: %s" % e
        _script_jobs[name] = {"proc": proc, "started": time.time(), "exit": None}
    def _reap(name=name, proc=proc, logf=logf):
        rc = proc.wait()
        with _script_lock:
            _script_jobs[name]["exit"] = rc
            _script_jobs[name]["proc"] = None
        logf.close()
    threading.Thread(target=_reap, daemon=True).start()
    return None

def _guard_loop():
    while True:
        try:
            if GUARD_NAME in _scripts_list():
                with _script_lock:
                    j = _script_jobs.get(GUARD_NAME)
                    busy = bool(j and j["proc"] is not None)
                if not busy:
                    _script_run(GUARD_NAME, "")
        except Exception:
            pass
        time.sleep(max(GUARD_EVERY, 60))

if GUARD_EVERY > 0:
    threading.Thread(target=_guard_loop, daemon=True).start()

@app.route("/api/scripts")
def api_scripts():
    rows = []
    for n in _scripts_list():
        with _script_lock:
            j = _script_jobs.get(n)
            if j and j["proc"] is not None:
                st, code = "running", None
            elif j:
                st, code = "done", j["exit"]
            else:
                st, code = "idle", None
        rows.append({"name": n, "state": st, "exit": code,
                     "is_guard": n == GUARD_NAME})
    return jsonify({"dir": SCRIPTS_DIR, "guard_every": GUARD_EVERY,
                    "scripts": rows})

@app.route("/api/scripts/run", methods=["POST"])
def api_scripts_run():
    d = request.get_json(silent=True) or {}
    err = _script_run(str(d.get("name", "")), str(d.get("args", "")))
    return jsonify({"ok": err is None, "error": err})

@app.route("/api/scripts/log")
def api_scripts_log():
    name = os.path.basename(request.args.get("name", ""))
    p = os.path.join(SCRIPT_LOGS, name + ".log")
    if not os.path.exists(p):
        return jsonify({"log": "(no runs yet)"})
    with open(p, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 12000))
        txt = f.read().decode("utf-8", "replace")
    return jsonify({"log": "\n".join(txt.splitlines()[-60:])})

# ---------------- GPU ECC panel ----------------
# The V620s reserve ~2 GiB/card for ECC (RAS). Turning ECC off reclaims it:
# 30 -> 32 GiB per card. It is NOT a live switch — it's the kernel boot
# parameter amdgpu.ras_enable=0, staged through TrueNAS middleware and applied
# by REBOOTING (the full reclaim historically took two boot cycles).
#
# The buttons here only STAGE the setting via midclt on the host (same
# nsenter channel the Scripts tab uses). They never reboot anything — you
# reboot deliberately, from the TrueNAS UI, when the GPUs are idle.
#
# Status is read from three independent places so every phase of the dance is
# visible: what's STAGED (middleware config), what this BOOT is running
# (/proc/cmdline), and what the cards actually DID (mem_info_vram_total).
ECC_TOKEN = "amdgpu.ras_enable=0"
GRUB_FILE = os.environ.get("GRUB_FILE", "/etc/default/grub")
GRUB_KEY  = "GRUB_CMDLINE_LINUX_DEFAULT"

# Two mechanisms. TrueNAS stages kernel options through middleware; Debian /
# Ubuntu stage them in GRUB. Detected, not assumed -
# the panel previously hard-coded midclt and simply read "unreachable" forever
# on the Ubuntu box.
_ecc_be = {"v": None}

def _host_out(cmd, timeout=15):
    try:
        r = subprocess.run(HOST_EXEC + cmd, capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout or ""), (r.stderr or ""), r.returncode
    except Exception as e:
        return "", "%s: %s" % (type(e).__name__, e), -1

def _ecc_backend():
    if _ecc_be["v"]:
        return _ecc_be["v"]
    out, _e, rc = _host_out(
        ["sh", "-c",
         "command -v midclt >/dev/null 2>&1 && echo midclt && exit 0; "
         "test -f %s && echo grub && exit 0; echo none" % GRUB_FILE])
    v = (out or "").strip().splitlines()
    _ecc_be["v"] = (v[-1] if v else "none") if rc == 0 else "none"
    return _ecc_be["v"]

# ---- pure helpers: unit-tested, no I/O ------------------------------------

def _grub_parse(text):
    """Current value of GRUB_CMDLINE_LINUX_DEFAULT, or None if unparseable.

    GRUB sources the file as shell, so the LAST uncommented assignment wins,
    and commented lines are ignored - matched here exactly.
    """
    val = None
    for ln in text.splitlines():
        st = ln.strip()
        if not st.startswith(GRUB_KEY + "=") or st.startswith("#"):
            continue
        m = re.match(r'^%s=(["\'])(.*)\1$' % GRUB_KEY, st)
        if m:
            val = m.group(2)
        else:
            m2 = re.match(r"^%s=(\S*)$" % GRUB_KEY, st)
            val = m2.group(1) if m2 else None
    return val

def _grub_apply(text, mode, token=ECC_TOKEN):
    """Pure read-modify-write. -> (new_text, new_value) or (None, reason).

    Only the GRUB_CMDLINE_LINUX_DEFAULT line changes; every other byte is
    preserved. This edits the host's real bootloader config and the blast
    radius of getting it wrong is a machine that will not boot.
    """
    if mode not in ("on", "off"):
        return None, "mode must be 'on' or 'off'"
    cur = _grub_parse(text)
    if cur is None and (GRUB_KEY + "=") in text:
        return None, "%s is present but not in a form this can safely edit" % GRUB_KEY
    toks = [t for t in (cur or "").split() if t != token]
    if mode == "off":
        toks.append(token)
    new_val = " ".join(toks)
    new_line = '%s="%s"' % (GRUB_KEY, new_val)
    lines = text.splitlines(keepends=True)
    last = None
    for i, ln in enumerate(lines):
        st = ln.strip()
        if st.startswith(GRUB_KEY + "=") and not st.startswith("#"):
            last = i
    if last is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(new_line + "\n")
    else:
        lines[last] = new_line + "\n"
    return "".join(lines), new_val

# ---- backend: TrueNAS middleware ------------------------------------------

def _ecc_advanced_config():
    out, err, rc = _host_out(["midclt", "call", "system.advanced.config"])
    if rc != 0:
        return None, (err or out or "midclt failed").strip()[:300]
    try:
        return json.loads(out), None
    except ValueError as e:
        return None, "midclt returned non-JSON: %s" % e

def _stage_midclt(mode):
    cfg, err = _ecc_advanced_config()
    if cfg is None:
        return False, ("cannot reach TrueNAS middleware: " + err)
    tokens = (cfg.get("kernel_extra_options") or "").split()
    want = [t for t in tokens if t != ECC_TOKEN]
    if mode == "off":
        want.append(ECC_TOKEN)
    body = json.dumps({"kernel_extra_options": " ".join(want)})
    out, err2, rc = _host_out(["midclt", "call", "system.advanced.update", body],
                              timeout=30)
    if rc != 0:
        return False, "midclt update failed: " + (err2 or out).strip()[:300]
    cfg2, err3 = _ecc_advanced_config()
    if cfg2 is None:
        return False, "staged, but could not read back to verify: " + err3
    now = (cfg2.get("kernel_extra_options") or "").split()
    ok = (ECC_TOKEN in now) if mode == "off" else (ECC_TOKEN not in now)
    if not ok:
        return False, "read-back does not show the change - nothing staged"
    return True, ("staged: kernel_extra_options = %r. Reboot when the GPUs are "
                  "idle." % (cfg2.get("kernel_extra_options") or ""))

# ---- backend: GRUB --------------------------------------------------------

def _grub_read():
    out, err, rc = _host_out(["cat", GRUB_FILE])
    if rc != 0:
        return None, (err or "cannot read " + GRUB_FILE).strip()[:300]
    return out, None

def _stage_grub(mode):
    text, err = _grub_read()
    if text is None:
        return False, "cannot read %s from this container: %s" % (GRUB_FILE, err)
    new_text, res = _grub_apply(text, mode)
    if new_text is None:
        return False, res

    # temp-then-move with a backup: a half-written bootloader config is the one
    # failure mode that costs a trip to the basement with a keyboard.
    script = ("set -e; cp -a {g} {g}.bak.ecc; cat > {g}.new; "
              "chmod --reference={g} {g}.new 2>/dev/null || true; "
              "mv {g}.new {g}").format(g=GRUB_FILE)
    try:
        r = subprocess.run(HOST_EXEC + ["sh", "-c", script], input=new_text,
                           capture_output=True, text=True, timeout=30)
    except Exception as e:
        return False, "write failed: %s: %s" % (type(e).__name__, e)
    if r.returncode != 0:
        return False, "write failed: " + ((r.stderr or r.stdout).strip()[:300])

    # verify the file BEFORE regenerating - never run update-grub against a
    # file we have not confirmed
    back, err2 = _grub_read()
    if back is None:
        return False, "wrote, but could not read back: " + err2
    now = _grub_parse(back) or ""
    ok = (ECC_TOKEN in now.split()) if mode == "off" else (ECC_TOKEN not in now.split())
    if not ok:
        return False, "read-back does not show the change - nothing staged"

    for cmd in (["update-grub"], ["update-grub2"],
                ["grub-mkconfig", "-o", "/boot/grub/grub.cfg"]):
        out, err3, rc = _host_out(cmd, timeout=120)
        if rc == 0:
            return True, ("staged: %s=%r and %s regenerated the boot config. "
                          "Reboot when the GPUs are idle. The reserve has "
                          "historically taken up to two boot cycles to release."
                          % (GRUB_KEY, now, cmd[0]))
    return False, ("%s updated, but no update-grub/grub-mkconfig succeeded on "
                   "the host. Run it by hand: sudo update-grub" % GRUB_FILE)

# ---- status ---------------------------------------------------------------

def _ecc_status():
    be = _ecc_backend()
    st = {"token": ECC_TOKEN, "backend": be}

    if be == "midclt":
        cfg, err = _ecc_advanced_config()
        if cfg is not None:
            opts = (cfg.get("kernel_extra_options") or "").strip()
            st["staged_options"] = opts
            st["staged_off"] = ECC_TOKEN in opts.split()
        else:
            st["backend_error"] = err
    elif be == "grub":
        text, err = _grub_read()
        if text is not None:
            val = _grub_parse(text)
            if val is None:
                st["backend_error"] = "%s is not in an editable form" % GRUB_KEY
            else:
                st["staged_options"] = val
                st["staged_off"] = ECC_TOKEN in val.split()
        else:
            st["backend_error"] = err
    else:
        st["backend_error"] = ("no supported mechanism found on the host "
                               "(looked for midclt, then %s)" % GRUB_FILE)

    out, _e, rc = _host_out(["cat", "/proc/cmdline"])
    if rc != 0:
        try:
            out = open("/proc/cmdline").read()
            rc = 0
        except OSError:
            pass
    if rc == 0:
        st["running_off"] = ECC_TOKEN in out.split()

    cards = []
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]/device")):
        vt = _readint(os.path.join(dev, "mem_info_vram_total"))
        if not vt:
            continue
        gib = vt / (1 << 30)
        cards.append({"card": os.path.basename(os.path.dirname(dev)),
                      "vram_gib": round(gib, 1),
                      "reclaimed": gib > 31.0})
    st["cards"] = cards

    staged, running = st.get("staged_off"), st.get("running_off")
    recl = [c["reclaimed"] for c in cards]
    if staged is None:
        st["phase"] = "unknown (%s)" % (st.get("backend_error") or "no backend")
    elif staged and not running:
        st["phase"] = "ECC-off staged - REBOOT to apply (boot 1 of up to 2)"
    elif staged and running and cards and not all(recl):
        st["phase"] = "running with ECC off, reserve not yet released - reboot once more"
    elif staged and running and cards and all(recl):
        st["phase"] = "ECC OFF - full VRAM reclaimed"
    elif not staged and running:
        st["phase"] = "ECC-on staged - REBOOT to re-enable"
    elif not staged and cards and not all(recl):
        st["phase"] = "ECC ON (default)"
    else:
        st["phase"] = "ECC on staged; cards still show reclaimed VRAM - reboot pending"
    return st

def _ecc_stage(mode):
    be = _ecc_backend()
    if be == "midclt":
        return _stage_midclt(mode)
    if be == "grub":
        return _stage_grub(mode)
    return False, ("no supported mechanism on this host. Do it by hand: add "
                   "%s to %s in %s, then sudo update-grub, then reboot."
                   % (ECC_TOKEN, GRUB_KEY, GRUB_FILE))

@app.route("/api/ecc")
def api_ecc():
    return jsonify(_ecc_status())

@app.route("/api/ecc/stage", methods=["POST"])
def api_ecc_stage():
    d = request.get_json(silent=True) or {}
    mode = str(d.get("mode", ""))
    if mode not in ("on", "off"):
        return jsonify({"ok": False, "msg": "mode must be 'on' or 'off'"}), 400
    ok, msg = _ecc_stage(mode)
    return jsonify({"ok": ok, "msg": msg, "status": _ecc_status()})

@app.route("/")
def index():
    # Substituted server-side rather than filled in by JS, so the build marker
    # is present in view-source even if the page's scripts never run or
    # /api/power is unreachable. That matters precisely in the case it exists
    # for: a container quietly serving a stale app.py.
    html = INDEX.replace("__APP_VERSION__", APP_VERSION)
    html = html.replace("__SITE_NAME__", SITE_NAME)
    # The thermal-test tab is substituted in, not conditionally hidden with CSS.
    # With SHOW_THERMAL_TEST off the markup is never sent, so there is no pane
    # to accidentally reveal and nothing for the JS to find - which is why
    # renderTest() and loadTestMeta() each check for their own elements first.
    html = html.replace("__TEST_PANE__", TEST_PANE if SHOW_THERMAL_TEST else "")
    html = html.replace("__TEST_TAB__", TEST_TAB if SHOW_THERMAL_TEST else "")
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    bind = DASH_BIND
    if not DASH_TOKEN and bind not in ("127.0.0.1", "localhost", "::1"):
        print("=" * 70)
        print("DASH_TOKEN is not set. Refusing to listen on %s." % bind)
        print("This dashboard exposes host telemetry and (optionally) a root")
        print("script runner. Set DASH_TOKEN=<a long random string>, or set")
        print("DASH_BIND=127.0.0.1 to run it loopback-only on purpose.")
        print("Suggested token: %s" % _secrets.token_urlsafe(24))
        print("=" * 70)
        bind = "127.0.0.1"
    print("%s telemetry on http://%s:%d  (auth: %s · scripts: %s)"
          % (SITE_NAME, bind, DASH_PORT, "token" if DASH_TOKEN else "NONE",
             "ENABLED" if ENABLE_SCRIPTS else "off"))
    app.run(host=bind, port=DASH_PORT, threaded=True)
