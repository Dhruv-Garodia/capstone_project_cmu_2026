#!/usr/bin/env python3
"""pfib-synth studio: generate stacks from a browser form and look through them.

    python studio.py            # opens http://127.0.0.1:8770
    python studio.py --port 9000 --no-browser

Pick a preset or set the microscope style, material family, porosity and any
render setting; each job runs generate.py, packs the result for the viewer and
renders its showcase GIFs. Everything lands in out/<name>/. Local only: the
server listens on 127.0.0.1.
"""
import argparse, inspect, json, mimetypes, subprocess, sys, threading, time, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "pfibsynth"))
import generate as G  # noqa: E402

JOBS, LOCK = {}, threading.Lock()


def schema():
    import materials
    fam = {"v5": {"doc": "Voronoi foam catalyst layer (cell sizes per style in slicing.json)", "params": {}}}
    for k, f in materials.FAMILIES.items():
        ps = {p.name: p.default for p in list(inspect.signature(f).parameters.values())[3:]
              if p.default is not p.empty and isinstance(p.default, (int, float, bool, tuple, list, type(None)))}
        fam[k] = {"doc": (f.__doc__ or "").strip().split("\n")[0], "params": {k2: list(v) if isinstance(v, tuple) else v for k2, v in ps.items()}}
    sl = json.loads((HERE / "pfibsynth/slicing.json").read_text())
    knobs = {k: {"doc": d, "default": {s: (sl[s].get("v7", {}).get(k, sl[s].get(k))) for s in G.STYLES}} for k, d in G.KNOBS.items()}
    return {"presets": G.PRESETS, "styles": G.STYLES, "families": fam, "knobs": knobs}


def packages():
    out = []
    if OUT.exists():
        for d in sorted(OUT.iterdir(), key=lambda p: -p.stat().st_mtime):
            if not (d / "stack.json").exists():
                continue
            t = json.loads((d / "truth/truth.json").read_text()) if (d / "truth/truth.json").exists() else {}
            out.append({"name": d.name, "frames": t.get("frames"), "style": t.get("style_calibrated_to"),
                        "family": t.get("material_family"), "porosity": t.get("true_porosity"),
                        "rule_porosity": t.get("start_mask_porosity"),
                        "pfibv": next((f"/out/{d.name}/{p.name}" for p in d.glob("*.pfibv")), None),
                        "gifs": [f"/out/{d.name}/gifs/{p.name}" for p in sorted((d / "gifs").glob("*.gif"))] if (d / "gifs").exists() else []})
    return out


def run_job(jid, job):
    out = Path(job["out"]); out.mkdir(parents=True, exist_ok=True)
    cfg = out.parent / f".{job['name']}.studio.json"; cfg.write_text(json.dumps(job))
    log = out.parent / f".{job['name']}.studio.log"
    steps = [("generate", [sys.executable, str(HERE / "generate.py"), "--config", str(cfg)]),
             ("pack for viewer", [sys.executable, str(HERE / "tools/export_viewer.py"), job["out"]]),
             ("showcase GIFs", [sys.executable, str(HERE / "tools/showcase.py"), job["out"], str(out / "gifs"), "--label", job["name"]])]
    if not job.get("gifs", True):
        steps = steps[:2]
    with open(log, "w") as lf:
        for label, cmd in steps:
            with LOCK:
                JOBS[jid]["step"] = label
            lf.write(f"\n== {label}\n"); lf.flush()
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
            if rc:
                with LOCK:
                    JOBS[jid].update(state="failed", step=f"{label} failed (exit {rc})")
                return
    with LOCK:
        JOBS[jid].update(state="done", step="done", finished=time.time())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def file(self, root, rel):
        p = (root / rel).resolve()
        if root.resolve() not in p.parents and p != root.resolve() or not p.is_file():
            return self.send(404, {"error": "not found"})
        self.send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.file(HERE / "docs/studio", "index.html")
        if path.startswith("/viewer"):
            rel = path[len("/viewer"):].lstrip("/") or "index.html"
            return self.file(HERE / "docs/viewer", rel)
        if path.startswith("/out/"):
            return self.file(OUT, path[5:])
        if path.startswith("/showcase/"):
            return self.file(HERE / "docs/showcase", path[10:])
        if path == "/api/schema":
            return self.send(200, schema())
        if path == "/api/packages":
            return self.send(200, packages())
        if path == "/api/jobs":
            with LOCK:
                js = [dict(j) for j in JOBS.values()]
            for j in js:
                lg = OUT / f".{j['name']}.studio.log"
                j["log"] = lg.read_text()[-3000:] if lg.exists() else ""
            return self.send(200, sorted(js, key=lambda j: -j["started"]))
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            return self.send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            name = "".join(c for c in str(req.get("name") or "") if c.isalnum() or c in "_-")[:60]
            if not name:
                raise ValueError("give the stack a name (letters, digits, _ or -)")
            if req.get("style") not in G.STYLES:
                raise ValueError("style must be one of " + ", ".join(G.STYLES))
            job = {"style": req["style"], "family": str(req.get("family") or "v5"), "porosity": float(req["porosity"]),
                   "seed": int(req.get("seed", 1)), "frames": max(10, min(int(req.get("frames", 60)), 700)),
                   "name": name, "out": str(OUT / name), "mat_kw": dict(req.get("mat_kw") or {}),
                   "override": dict(req.get("override") or {}), "gifs": bool(req.get("gifs", True))}
            if req.get("width"):
                job["width"] = max(128, min(int(req["width"]), 2000))
            if not 0.02 <= job["porosity"] <= 0.95:
                raise ValueError("porosity must be between 0.02 and 0.95")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self.send(400, {"error": str(e)})
        jid = f"{name}-{int(time.time() * 1000)}"
        with LOCK:
            if any(j["name"] == name and j["state"] == "running" for j in JOBS.values()):
                return self.send(409, {"error": f"{name} is already running"})
            JOBS[jid] = {"id": jid, "name": name, "state": "running", "step": "queued", "started": time.time(), "job": job}
        threading.Thread(target=run_job, args=(jid, job), daemon=True).start()
        self.send(200, {"id": jid})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770); ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"pfib-synth studio at {url}  (ctrl-c to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
