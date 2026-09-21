import json, os, re, shutil, zipfile

scan = json.load(open('/runs/scan_bounds_h96x8.json'))
top = scan['top'][0]
print('BEST_CONFIG', json.dumps(top))
tpl = '/runs/submission_f3_20260919.zip'
if not os.path.exists(tpl):
    tpl = '/runs/submission_d_20260919.zip'
print('TEMPLATE', tpl)
assert os.path.exists(tpl), 'no template zip'

work = '/runs/pack_h96x8_tmp'
outzip = '/runs/submission_h_h96x8_20260920.zip'
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
with zipfile.ZipFile(tpl) as z:
    z.extractall(work)

sub = os.path.join(work, 'submission.py')
assert os.path.exists(sub), sorted(os.listdir(work))[:40]
txt = open(sub, encoding='utf-8').read()

def setline(txt, name, val, required=True):
    new, n = re.subn(rf'^{name} = .*$', f'{name} = {val}', txt, flags=re.M)
    if n == 0 and not required:
        print('SKIP_MISSING', name)
        return txt
    if n != 1:
        raise SystemExit(f'WARN {name} replaced {n} times')
    return new

required = [
    ('_HIDDEN', '96'),
    ('_HEAD_HIDDEN', '64'),
    ('_CORRECTION_ALPHA', '1.0'),
    ('_MAX_DELTA', '0.04'),
    ('_BOUND_ABS', '0.0075'),
    ('_BOUND_REL', '0.0075'),
    ('_UNC_FLOOR_U', str(top['floor'])),
    ('_UNC_MULT_U', str(top['mult_u'])),
    ('_UNC_FLOOR_V', str(top['floor'])),
    ('_UNC_MULT_V', str(top['mult_v'])),
    ('_UNC_REL', str(top['rel'])),
]
optional = [
    ('_BLOCKS', '2'),
    ('_DROPOUT', '0.0'),
    ('_HEAD_BLOCKS', '2'),
    ('_HEAD_DROPOUT', '0.0'),
    ('_HEAD_INCLUDE_PRESSURE', 'True'),
    ('_HEAD_HISTORY_CONTEXT', 'False'),
    ('_INCLUDE_PRESSURE', 'True'),
    ('_HISTORY_CONTEXT', 'False'),
]
for name, val in required:
    txt = setline(txt, name, val, required=True)
for name, val in optional:
    txt = setline(txt, name, val, required=False)

open(sub, 'w', encoding='utf-8').write(txt)
print('CONSTS_OK')
for line in txt.splitlines():
    if line.startswith('_') and any(k in line for k in ['HIDDEN','BLOCKS','UNC','BOUND','MAX_DELTA','CORRECTION_ALPHA','DROPOUT','INCLUDE_PRESSURE','HISTORY_CONTEXT']):
        print(line)

shutil.copy('/runs/residual_h96x8_all81_20260920/model_best.pth', os.path.join(work, 'model.pth'))
shutil.copy('/runs/head_h96x8_h64logmae/head_5000.pth', os.path.join(work, 'uncertainty_head.pt'))
with zipfile.ZipFile(outzip, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(work):
        for fn in files:
            p = os.path.join(root, fn)
            z.write(p, os.path.relpath(p, work))
print('WROTE', outzip, os.path.getsize(outzip))
