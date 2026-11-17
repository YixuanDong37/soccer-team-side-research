import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, 'C:/Research/plugins/tools')
sys.path.insert(0, 'C:/Research/plugins/tools/reid_lab/work/v_repro_1')
import diag_points as d
from reid_lab import data, score
out = {}
def run(name, links_of, cuts_of):
    tot = 0; comp = [0, 0]; imp = 0; rows = 0
    for s in data.DEV_CLIPS:
        x = d.clip_diag(s, links_of(s), cuts_of(s))
        tot += x['identities_with_two_people']; comp[0] += x['completeness'][0]; comp[1] += x['completeness'][1]; imp += x['impure']; rows += x['rows']
    out[name] = {"identities_with_two_people": tot, "completeness": round(comp[0] / comp[1], 4), "impure_rows": imp, "rows": rows, "impure_share": round(imp / rows, 5)}
    print(name, out[name], flush=True)
run('gbf now (split)', lambda s: [], lambda s: [])
for name in ('plugin', 'mutual'):
    b = score.baseline_links(name)
    run(name, lambda s: b[s]['links'], lambda s: [])
m = json.load(open('C:/Research/plugins/tools/reid_lab/work/v_repro_1/METHOD_cuts_only.json'))['clips']
run('cuts only', lambda s: [], lambda s: m[s]['cuts'])
json.dump(out, open('C:/Research/plugins/tools/reid_lab/work/v_repro_1/out/diag_baselines.json', 'w'), indent=1)
