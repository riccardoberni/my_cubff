#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bin import cubff

checkpoint_path = sys.argv[1]
with open(checkpoint_path + ".json") as f:
    import json
    meta = json.load(f)

language = cubff.GetLanguage(meta.get("lang", "bff_noheads"))
params = cubff.SimulationParams()
params.num_programs = meta["num_programs"]
params.seed = meta["seed"]
params.callback_interval = 1
params.eval_selfrep = True
params.load_from = checkpoint_path

def callback(state):
    scores = list(state.replication_per_prog)
    above = [(i, s) for i, s in enumerate(scores) if s >= cubff.kSelfrepThreshold]
    print(f"Programmi sopra soglia ({cubff.kSelfrepThreshold}): {len(above)} su {len(scores)}")
    for i, s in sorted(above, key=lambda t: -t[1])[:20]:
        print(f"  slot #{i}: score {s}/64")
    return True

language.RunSimulation(params, None, callback)
