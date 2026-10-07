"""Read generated assets through local HTTP and check inline script syntax."""
from pathlib import Path
from functools import partial
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urljoin,urlparse,unquote
from urllib.request import build_opener,ProxyHandler
import argparse,threading,subprocess,re,shutil
from PIL import Image
from .provenance import *

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value:self.links.append(value)

class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass

def check_visual(run):
    run=Path(run);verify(run)
    if (run/'release.json').exists():raise ValueError('sealed visual review')
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(run)));worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start();base=f'http://127.0.0.1:{server.server_port}/';records=[];scripts=[]
    try:
        urls={urljoin(base,'index.html')}
        for html in run.rglob('*.html'):
            relative=html.relative_to(run).as_posix();here=urljoin(base,relative);urls.add(here);parser=Links();text=html.read_text();parser.feed(text)
            for link in parser.links:
                address=urljoin(here,link)
                if urlparse(address).netloc!=urlparse(base).netloc:raise ValueError('unexpected external visualization dependency')
                urls.add(address)
            for i,script in enumerate(re.findall(r'<script[^>]*>(.*?)</script>',text,re.S)):
                path=run/'S6'/f'visual-script-{len(scripts)}.js';path.write_text(script)
                isolated=Path('/root/miniconda3/envs/codex-cli/bin/node')
                node=str(isolated) if isolated.is_file() else shutil.which('node')
                if node:
                    result=subprocess.run([node,'--check',str(path)],text=True,capture_output=True)
                    if result.returncode:raise ValueError(result.stderr)
                    scripts.append(dict(html=relative,index=i,status='syntax_passed',node=node,source_sha256=sha(path),environment_note='base conda Node has an unresolved sqlite3session_attach symbol; used existing isolated Node without changing environment'))
                else:scripts.append(dict(html=relative,index=i,status='node_unavailable'))
        local=build_opener(ProxyHandler({}))
        for url in sorted(urls):
            with local.open(url,timeout=10) as result:
                body=result.read();records.append(dict(path=unquote(urlparse(url).path),status=result.status,bytes=len(body),content_type=result.headers['Content-Type']))
                if result.status!=200 or not body:raise ValueError('empty/missing HTTP resource')
    finally:server.shutdown();server.server_close();worker.join(timeout=2)
    gif=run/'visualization/daily-q5-retry/cycle.gif'
    with Image.open(gif) as im:
        frames=im.n_frames
        for i in range(frames):im.seek(i);im.load()
        for i in (0,frames//2,frames-1):im.seek(i);im.convert('RGB').save(run/'S6'/f'gif-sample-{i}.png')
    write(run/'S6/visual-assets-check.json',dict(status='passed_scoped',http=records,inline_scripts=scripts,gif_decoded_frames=frames,gif_sha256=sha(gif),browser_interaction='not_executed',reason='HTTP, decoded imagery and JavaScript syntax checked; no browser-control run claimed'))
    print('VISUAL_ASSETS',len(records),'GIF',frames,'scripts',scripts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):check_visual(a.run)
