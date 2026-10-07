"""Check local saved visualization resources without starting a simulation."""
from pathlib import Path
import argparse,http.server,threading,urllib.request,re,subprocess,os
from functools import partial
from PIL import Image
from .provenance import read,write,sha,PROGRESS


def check(run):
    p=Path(run)
    if (p/'release.json').exists():raise ValueError('sealed report cannot be rewritten')
    case=read(p/'S6/final-protocol.json')['default_case'];folder=p/'visualization'/case
    (p/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    scripts=re.findall(r'<script>([\s\S]*?)</script>',(folder/'index.html').read_text())
    if len(scripts)!=1:raise ValueError('unexpected script layout')
    js=p/'S6/visualization-script.js';js.write_text(scripts[0]);env=os.environ.copy();sqlite=Path('/root/miniconda3/lib/libsqlite3.so.3.53.4')
    if sqlite.exists():env['LD_PRELOAD']=str(sqlite)
    node=subprocess.run(['node','--check',str(js)],capture_output=True,text=True,env=env)
    if node.returncode:raise ValueError(node.stderr)
    with Image.open(folder/'cycle.gif') as im:
        n=im.n_frames
        for i in range(n):im.seek(i);im.load()
    if n!=12:raise ValueError('unexpected animation frame count')
    images=[]
    for img in (folder/'scene-summary.png',p/'visualization/diagnostics/diagnostics.png'):
        with Image.open(img) as im:im.verify()
        images.append(dict(path=str(img.relative_to(p)),sha256=sha(img)))
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',8768),partial(Quiet,directory=str(p)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();checks=[]
    # The environment proxy returned 502 for loopback; a local resource check
    # must connect directly without modifying any global proxy setting.
    client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    paths=['index.html',f'visualization/{case}/index.html',f'visualization/{case}/scene-summary.png',f'visualization/{case}/cycle.gif','visualization/diagnostics/index.html','visualization/diagnostics/diagnostics.png','implementation-report.md']
    try:
        for rel in paths:
            with client.open(f'http://127.0.0.1:8768/{rel}',timeout=10) as response:
                body=response.read()
                if response.status!=200 or body!=(p/rel).read_bytes():raise ValueError('HTTP bytes differ '+rel)
                checks.append(dict(path=rel,status=response.status,bytes=len(body),sha256=sha(p/rel)))
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
    previous=read(p/'S6/visual-review.json') if (p/'S6/visual-review.json').exists() else {}
    write(p/'S6/visual-review.json',dict(status='passed_scoped',http_resources=checks,loopback_port=8768,direct_loopback_without_environment_proxy=True,script_syntax='node --check passed',script_sha256=sha(js),process_local_sqlite_preload=str(sqlite) if sqlite.exists() else None,gif_frames=n,images=images,image_visual_review=previous.get('image_visual_review','pending'),browser_interaction_exercised=False,display_deformation_multiplier=10,no_extra_integration=True,scope='Image inspection plus HTTP bytes/GIF decoding/JS syntax; browser controls not exercised.'))
    print('VISUAL_CHECK',len(checks),'HTTP resources',n,'GIF frames; JS passed; browser controls not exercised',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();check(a.run)
