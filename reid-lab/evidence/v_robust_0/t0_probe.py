"""probe (LABELS, diagnosis): what the reference loader returns; sizes of the raw LOCO pickle."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score
from gbf.evaluate import gsr
ref = gsr.load_reference(score.TRAIN / "SNGS-162" / "Labels-GameState.json")
print(type(ref), [a for a in dir(ref) if not a.startswith("_")])
print(ref.people.columns.tolist())
print(ref.people.head(3).to_string())
print(ref.people.groupby("role").ref_id.nunique().to_dict())
c = score.clip("SNGS-162")
print(len(c.returns), c.returns[:3])
t = data.tracks("SNGS-162")
print(t.columns.tolist())
print(t.role.value_counts(dropna=False).to_dict(), t.team.value_counts(dropna=False).to_dict())
print(t[t.jersey.notna()].shape if "jersey" in t else None)
import pickle
raw = pickle.load(open("C:/Research/plugins/tools/reid_lab/work/fuse/cache/loco_raw.pkl", "rb"))
print(raw.keys(), list(raw["folds"].keys()), list(raw["folds"]["SNGS-107"].keys()))
k = list(raw["folds"]["SNGS-107"]["res"].keys())
print(len(k), k[:3], raw["folds"]["SNGS-107"]["res"][k[0]].keys())
