import re, shutil, zipfile, subprocess
from pathlib import Path

R = Path('/home/chyfuture/realpde_runs')
src = R / 'submission_d_20260919.zip'
build = R / 'build_f3_20260919'
corrector = R / 'residual_h96_all81_20260919' / 'model_best.pth'
head = R / 'head_h96cache_h64logmae' / 'head_5500.pth'

if build.exists():
    shutil.rmtree(build)
build.mkdir()
with zipfile.ZipFile(src) as z:
    z.extractall(build)

shutil.copy(corrector, build / 'model.pth')
shutil.copy(head, build / 'uncertainty_head.pt')

p = build / 'submission.py'
s = p.read_text()
s = s.replace('_UNC_REL = 0.0075', '_UNC_REL = 0.005')
s = re.sub(r'^_HIDDEN = 64$', '_HIDDEN = 96', s, count=1, flags=re.M)
assert re.search(r'^_HIDDEN = 96$', s, re.M), 'corrector hidden not set'
assert re.search(r'^_HEAD_HIDDEN = 64$', s, re.M), 'head hidden must stay 64'
p.write_text(s)

print('=== CONSTANTS ===')
for line in s.splitlines():
    if line.startswith('_UNC_') or line.startswith('_HEAD_HIDDEN') or line.startswith('_MAX_DELTA') or line.startswith('_HIDDEN') or line.startswith('_BLOCKS'):
        print('  ' + line)

out = R / 'submission_f3_20260919.zip'
if out.exists():
    out.unlink()
subprocess.run(['zip', '-qr', str(out), '.'], cwd=build, check=True)
print('WROTE', out, out.stat().st_size)
