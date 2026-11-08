"""Para cada arena: testa seeds/MULT, escolhe a partida mais disputada e renderiza."""
import sys, os, subprocess, json
import numpy as np
import arena as A

def run(name, seeds=range(0, 5)):
    ar = A.Arena(A.CONFIGS[name])
    log = open(f"out_{name}_auto.log", "a")
    best = None
    tried = {}
    for mult in (2, 3):
        res = []
        for seed in seeds:
            _, au, pops, leads, st, counts = A.simulate(ar, seed, mult, keep=False)
            sc, desc = A.score(st, leads, counts)
            res.append((sc, seed, mult, desc, st["t_over"]))
            print(name, mult, seed, round(sc, 1), desc, file=log, flush=True)
        tried[mult] = res
        b = max(res)
        if best is None or b[0] > best[0]:
            best = b
        fulls = [r[4] for r in res if r[4] is not None]
        if best[0] > 0:
            break
        if mult == 3 and fulls and np.median(fulls) < 28:
            # rápido demais -> pula direto pro 2
            continue
    sc, seed, mult, desc, _ = best
    print(name, "ESCOLHIDO", mult, seed, round(sc, 1), desc, file=log, flush=True)
    subprocess.run([sys.executable, "arena.py", "render", name, str(seed)], env=dict(os.environ, MULT=str(mult)),
                   stdout=log, stderr=log, check=False)
    print(name, "FIM", file=log, flush=True)

if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
