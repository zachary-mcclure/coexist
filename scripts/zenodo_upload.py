#!/usr/bin/env python3
"""Drive the Zenodo deposit workflow for the Coexist paper-2 archive.

Everything up to — but never including — the irreversible "Publish" click,
which is done by a human on the Zenodo web page.

Usage (token from https://zenodo.org/account/settings/applications/tokens/new
with scopes `deposit:write` and `deposit:actions`):

    export ZENODO_TOKEN=...
    python scripts/zenodo_upload.py reserve            # create draft, print reserved DOI
    python scripts/zenodo_upload.py package [git-ref]  # git archive -> dist/ tarball (default HEAD)
    python scripts/zenodo_upload.py upload FILE        # upload a file to the draft
    python scripts/zenodo_upload.py metadata           # re-push .zenodo.json to the draft
    python scripts/zenodo_upload.py status             # show draft state + files

Set ZENODO_SANDBOX=1 to run the same flow against sandbox.zenodo.org first
(sandbox needs its own account and token). Deposition id/DOI are cached in
scripts/zenodo_deposition.json (safe to commit: it holds no secrets).
"""

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(REPO_ROOT, "scripts", "zenodo_deposition.json")
METADATA_PATH = os.path.join(REPO_ROOT, ".zenodo.json")

SANDBOX = os.environ.get("ZENODO_SANDBOX") == "1"
BASE = "https://sandbox.zenodo.org/api" if SANDBOX else "https://zenodo.org/api"


def token():
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        sys.exit("ZENODO_TOKEN is not set. Create one at "
                 f"https://{'sandbox.' if SANDBOX else ''}zenodo.org/account/settings/"
                 "applications/tokens/new (scopes: deposit:write, deposit:actions).")
    return tok


def request(method, url, data=None, content_type="application/json"):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token()}")
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo API error {e.code} on {method} {url}:\n{e.read().decode()}")
    return json.loads(body) if body else {}


def load_state():
    if not os.path.exists(STATE_PATH):
        sys.exit(f"No deposition state at {STATE_PATH} — run `reserve` first.")
    with open(STATE_PATH) as f:
        return json.load(f)


def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def cmd_reserve():
    if os.path.exists(STATE_PATH):
        sys.exit(f"{STATE_PATH} already exists (deposition already reserved). "
                 "Delete it only if you deliberately want a NEW deposition/DOI.")
    meta = load_metadata()
    meta["metadata"]["prereserve_doi"] = True
    dep = request("POST", f"{BASE}/deposit/depositions",
                  data=json.dumps(meta).encode())
    doi = dep["metadata"]["prereserve_doi"]["doi"]
    state = {
        "sandbox": SANDBOX,
        "id": dep["id"],
        "doi": doi,
        "bucket": dep["links"]["bucket"],
        "html": dep["links"]["html"],
    }
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Draft deposition {dep['id']} created ({'SANDBOX' if SANDBOX else 'production'}).")
    print(f"Reserved DOI: {doi}")
    print(f"Draft page:   {dep['links']['html']}")
    print()
    print("Next: put the DOI in docs/phase_diagrams.tex (Data and code")
    print("availability) and CITATION.cff, rebuild the PDF, commit, tag, then")
    print("`package` and `upload`. See docs/SUBMISSION.md.")


def cmd_package(ref="HEAD"):
    short = subprocess.check_output(
        ["git", "rev-parse", "--short", ref], cwd=REPO_ROOT).decode().strip()
    label = ref if ref != "HEAD" else short
    dist = os.path.join(REPO_ROOT, "dist")
    os.makedirs(dist, exist_ok=True)
    out = os.path.join(dist, f"Coexist-{label}.tar.gz")
    subprocess.check_call(
        ["git", "archive", "--format=tar.gz", f"--prefix=Coexist-{label}/",
         "-o", out, ref], cwd=REPO_ROOT)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    size_mb = os.path.getsize(out) / 1e6
    print(f"Wrote {out} ({size_mb:.1f} MB)")
    print(f"sha256: {sha}")


def cmd_upload(path):
    state = load_state()
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    info = request("PUT", f"{state['bucket']}/{name}", data=data,
                   content_type="application/octet-stream")
    print(f"Uploaded {name} ({len(data)/1e6:.1f} MB) to deposition {state['id']}.")
    print(f"Zenodo checksum: {info.get('checksum', '?')}")
    local = hashlib.md5(data).hexdigest()
    remote = info.get("checksum", "").removeprefix("md5:")
    print("Checksum match." if local == remote
          else f"CHECKSUM MISMATCH — local md5 {local}. Re-upload before publishing.")
    print()
    print(f"Review the draft, then click Publish yourself: {state['html']}")
    print("(Publishing is irreversible: files cannot be changed afterward,")
    print(" only new versions added.)")


def cmd_metadata():
    state = load_state()
    dep = request("PUT", f"{BASE}/deposit/depositions/{state['id']}",
                  data=json.dumps(load_metadata()).encode())
    print(f"Metadata updated on deposition {state['id']} "
          f"(title: {dep['metadata']['title'][:60]}...).")


def cmd_status():
    state = load_state()
    dep = request("GET", f"{BASE}/deposit/depositions/{state['id']}")
    print(f"Deposition {state['id']}  state={dep.get('state')}  "
          f"submitted={dep.get('submitted')}")
    print(f"DOI: {state['doi']}")
    print(f"Page: {state['html']}")
    for f_ in dep.get("files", []):
        print(f"  file: {f_['filename']}  {int(f_['filesize'])/1e6:.1f} MB  {f_['checksum']}")


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    cmd, rest = args[0], args[1:]
    if cmd == "reserve":
        cmd_reserve()
    elif cmd == "package":
        cmd_package(*rest) if rest else cmd_package()
    elif cmd == "upload":
        if not rest:
            sys.exit("upload needs a file path (run `package` first).")
        cmd_upload(rest[0])
    elif cmd == "metadata":
        cmd_metadata()
    elif cmd == "status":
        cmd_status()
    else:
        sys.exit(f"Unknown command {cmd!r}.\n\n{__doc__}")


if __name__ == "__main__":
    main()
