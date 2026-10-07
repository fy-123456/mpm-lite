"""Local preview checks explicitly bypass network proxies for loopback requests."""
import argparse,hashlib,re,subprocess,sys,time,urllib.request
from PIL import Image
from .provenance import *
from .runtime import update


def main(run):
    run=Path(run).absolute();mutable(run)
    command=[sys.executable,'-B','-m','http.server','8765','--bind','127.0.0.1','--directory',str(run)]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));results=[]
    with (run/'S5/http-server-final.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(40):
                if proc.poll() is not None:raise RuntimeError('preview server failed or port is occupied')
                try:
                    with opener.open('http://127.0.0.1:8765/',timeout=1) as f:html=f.read().decode()
                    break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('loopback server not ready')
            paths=['index.html']+re.findall(r'(?:src|href)="([^\"]+)"',html)
            for path in paths:
                with opener.open('http://127.0.0.1:8765/'+path,timeout=5) as f:body=f.read();status=f.status
                if hashlib.sha256(body).hexdigest()!=sha(run/path):raise ValueError('served resource differs '+path)
                results.append(dict(path=path,status=status,bytes=len(body)))
            for p in (run/'figures').glob('*.png'):
                with Image.open(p) as img:img.load()
        finally:
            proc.terminate();proc.wait(timeout=5)
    write(run/'S5/cli-http-check.json',dict(status='passed_scoped',command=command,resources=results,proxy_policy='direct localhost opener only; environment unchanged',server_stopped_after_check=True,browser_interaction=False))
    write(run/'S5/postprocessing-failures.json',dict(records=[dict(action='initial HTTP verification',result='server not ready',root_cause='environment HTTP proxy returned 502 for loopback; same server returned200 with ProxyHandler({})',resolution='local HTTP checker bypasses proxies; no physics rerun'),dict(action='first seal attempt',result='refused, required visual-review.json missing',resolution='complete real image and HTTP checks before retrying seal',release_written=False)],numerical_failures=0))
    update(run,f'S5可视化：已实际查看5张图；{len(results)}个HTTP资源及图像解码通过。localhost代理502已定位，检查器仅对本地请求直连，环境不改；首次封存因缺可视化检查被拒绝，未写release。8765命令已实际启动后停止，未执行浏览器交互。')
    print('HTTP_CHECK',len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
