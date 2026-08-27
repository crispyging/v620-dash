# v620-dash

A single-file telemetry and model-lane dashboard for **AMD GPU inference
boxes** — built and used daily on 3× Radeon Pro V620 (gfx1030) in a
Supermicro chassis running llama-swap.

It answers the questions that made me write it: *what is actually resident on
each card right now, which process is holding it, is the box about to cook
itself, and why did that model not appear in the lane?*

![tabs: Live · Pulls · HF import · Lane · Bench · Scripts]

## What it shows

**Per-card VRAM occupancy, attributed to processes.** The bar's length is
always amdgpu's own `mem_info_vram_used` — ground truth. The coloured
segments inside it are labels for that length, read from
`/proc/<pid>/fdinfo`, so ComfyUI and a llama-server show up side by side.
Anything allocated that nothing can name stays visible as "other" rather
than being folded into free space, because a card that is 80 % full for
reasons nothing can explain is exactly the thing worth seeing.

**System RAM as its own bar** — per-process private RSS, page cache as a
separate band (that is where mmap'd GGUFs live, so a big mapper is annotated
"+N mapped" rather than double-counted), per-NUMA-node free, and a warning
when available RAM gets low on a box with no swap.

**Power, CPU/GPU temperatures and per-fan RPM** from IPMI, logged together
to CSV so you can correlate a thermal event with what was running. Per-fan,
not just the maximum — on a chassis where one zone drives everything, the
spread between fans is the interesting signal.

**A multi-lane llama-swap tab.** Point it at one llama-swap or several
(production + test lanes); it lists what each is serving, what GGUFs are on
disk but *not* in any config, and can add a model entry for you. Split GGUFs
are handled properly: only shard 1 is addable, continuation shards are shown
greyed as "loads with shard 1", and an incomplete set is flagged red instead
of producing a config that fails at load.

**A Hugging Face import tab** that resolves a repo, shows the real byte size
of every quantisation and what it would do to *your* cards, then downloads
into your models directory — inside the container, so closing the browser
does not cancel it.

## Requirements

- **AMD GPUs on amdgpu.** VRAM/temperature panels read amdgpu sysfs and DRM
  fdinfo. There is no NVIDIA support (see Limitations).
- **Linux host**, Docker + compose.
- **IPMI** (`/dev/ipmi0`) for power, CPU temps and fans. Without it those
  panels report unavailable and everything else still works.
- Optional: llama-swap, Ollama, `gpu-fan-control`.

If `/dev/ipmi0` is missing on the host:

```bash
sudo modprobe ipmi_si && sudo modprobe ipmi_devintf
echo -e "ipmi_si\nipmi_devintf" | sudo tee /etc/modules-load.d/ipmi.conf
```

## Install

```bash
git clone <this repo> v620-dash && cd v620-dash
cp env.example .env
python3 -c "import secrets;print(secrets.token_urlsafe(24))"   # paste into DASH_TOKEN
$EDITOR .env docker-compose.yml     # set your model + lane-config paths
docker compose up -d
```

Then open `http://<host>:8500` and sign in with the token.

`app.py` is bind-mounted rather than baked into an image, so `docker restart
v620-dash` always picks up edits. It is one file on purpose — read it, change
it, it is about 8,500 lines of commented Python with no framework and no
build step.

## Security — read before exposing this

This dashboard is **infrastructure control**, not a status page.

- **Auth is a single shared token.** No users, no roles, no CSRF protection.
  Anyone with the token has everything.
- **It refuses to start on a non-loopback interface without `DASH_TOKEN`**
  and prints a generated one. That is deliberate.
- **`ENABLE_SCRIPTS=1` is a remote root shell.** It runs files from
  `SCRIPTS_DIR` on the *host* as root, via `nsenter`. Off by default. Turn it
  on only if you understand exactly that.
- **`privileged: true` and `pid: host`** are required for IPMI and
  per-process VRAM attribution respectively. Drop them if you do not need
  those panels; the app degrades instead of crashing.
- **Do not put this on the public internet.** LAN or VPN only. If you need
  remote access, front it with a reverse proxy that does real
  authentication.

Destructive actions are separately gated and default to off/safe:
`ALLOW_MODEL_DELETE=0`, `ALLOW_MODEL_UNLOAD=1`, `SHOW_THERMAL_TEST=0`.

## Configuration

Everything is environment variables; see `docker-compose.yml` for the
annotated set. The ones people actually change:

| variable | meaning |
|---|---|
| `DASH_TOKEN` | shared access token (required off-loopback) |
| `SITE_NAME` | what the page calls this box |
| `LANE_URL`, `LANE_CONFIG`, `LANE_MODELS_DIR` | a single llama-swap lane |
| `LANE_EXTRA_PORTS` | quick multi-lane: `8092,8096` adds lane2/lane3 on the same host |
| `LANES` | full control: `name\|config-path\|url;name2\|...` |
| `OLLAMA_URL` | Ollama endpoint; **empty = lane-only mode**, all Ollama UI hides itself |
| `ENABLE_SCRIPTS` | expose the host-root script runner (default off) |
| `GPU_SYS`, `PROC_ROOT`, `RAM_SYS` | override hardware roots — used for testing against fake trees |

## Limitations — the honest list

- **AMD only.** The VRAM panel's per-process attribution is built on amdgpu
  sysfs + DRM fdinfo. NVIDIA would need a parallel NVML implementation;
  contributions welcome, but nothing is stubbed for it today.
- **Fan control assumes Supermicro IPMI raw commands** and quirks measured on
  one board (`0x30 0x45` mode switching; on that BMC a commanded duty ≤ 20 %
  *latches every fan at maximum* until `ipmitool mc reset cold`). Other
  vendors' BMCs will do something different or nothing. The fan *readings*
  are generic IPMI and work anywhere.
- **The ECC panel** stages kernel parameters via TrueNAS `midclt` or GRUB,
  detected at runtime. It stages only — you reboot yourself.
- **Flask's development server**, single process. Fine for one operator on a
  LAN; not a hardened deployment.
- **No historical database.** Live state is in memory (last 360 samples);
  longer runs are written to CSV in `DATA_DIR` when you press Start log.

## Why it exists

Because `rocm-smi` in a terminal tells you a card is 80 % full and nothing
else — not which process, not which model, not whether the fans can even
feel it. On a passively-cooled multi-GPU box the gap between "a card is hot"
and "the BMC has no idea a card is hot" is the difference between a working
machine and a dead one.

## License

MIT. No warranty — this thing writes to your BMC and your model configs.
