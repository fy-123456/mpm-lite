import argparse,hashlib,io,time,unittest
from pathlib import Path
from .geometry_check import write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();start=time.perf_counter()
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('tests.research_formal_pressure_next.test_contracts')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    text=stream.getvalue();print(text);(a.run/'S7/tests.txt').write_text(text)
    root=Path(__file__).resolve().parents[2]
    engine={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'engine/aniso_phase1/research_formal_pressure_next').glob('*.py')}
    write(a.run/'S7/test-report.json',dict(successful=result.wasSuccessful(),tests=result.testsRun,engine_sources=engine,seconds=time.perf_counter()-start))
    if not result.wasSuccessful():raise SystemExit(1)
