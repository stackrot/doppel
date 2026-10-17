"""Score `doppel --json` output against a make_samples.py manifest.

python -I tools/score.py ROOT doppel.json
"""

import collections
import itertools
import json
import sys
from pathlib import Path

RANK = {"original": 0, "exact": 0, "near": 1, "similar": 2}


def length(ranges):
	return sum(e - s for _, s, e in ranges)


def overlap(a, b):
	return sum(max(0, min(e1, e2) - max(s1, s2))
		for m1, s1, e1 in a for m2, s2, e2 in b if m1 == m2)


def content(path, m):
	"""Byte-identical files share a content id."""
	if "copy_of" in m:
		return m["copy_of"]
	if m["class"] in ("original", "exact"):
		return ("orig", m.get("source") or str(m.get("ranges")))
	return path


def expected(a, b, ma, mb):
	if content(a, ma) == content(b, mb):
		return "exact"
	return "near" if max(RANK[ma["class"]], RANK[mb["class"]]) <= 1 \
		else "similar"


def related(kind, ma, mb):
	if kind == "images":
		return ma["source"] == mb["source"]
	short = min(length(ma["ranges"]), length(mb["ranges"]))
	return overlap(ma["ranges"], mb["ranges"]) >= 0.5 * short


def truth_pairs(kind, files):
	by_key = collections.defaultdict(list)
	for p, m in files.items():
		if kind == "images":
			by_key[m["source"]].append(p)
		else:
			for name, _, _ in m["ranges"]:
				by_key[name].append(p)
	pairs = set()
	for ps in by_key.values():
		for a, b in itertools.combinations(sorted(set(ps)), 2):
			if related(kind, files[a], files[b]):
				pairs.add((a, b))
	return pairs


def main():
	root = Path(sys.argv[1]).resolve()
	corpus = root / "corpus"
	man = json.loads((root / "manifest.json").read_text())
	groups = json.loads(Path(sys.argv[2]).read_text())
	pred = {"images": set(), "videos": set()}
	member = {}
	sizes = []
	led = {"images": [], "videos": []}
	for g in groups:
		kind = g["kind"] + "s"
		paths = []
		leader = None
		for m in g["members"]:
			p = str(Path(m["path"]).relative_to(corpus))
			paths.append(p)
			member[p] = (m["link"], m["score"], g["link"], len(g["members"]))
			if m.get("leader"):
				leader = p
		sizes.append(len(paths))
		pred[kind].update(itertools.combinations(sorted(paths), 2))
		led[kind].append((leader, [p for p in paths if p != leader]))
	print(f"groups {len(groups)}, largest {max(sizes, default=0)}")
	for kind in ("images", "videos"):
		files = man[kind]
		truth = truth_pairs(kind, files)
		p = pred[kind]
		tp = p & truth
		prec = len(tp) / len(p) if p else 1.0
		rec = len(tp) / len(truth) if truth else 1.0
		print(f"\n{kind}: {len(files)} files, {len(truth)} true pairs, "
			f"{len(p)} predicted")
		print(f"  pair precision {prec:.4f}  recall {rec:.4f}")
		# Pairs of two untouched originals are mostly duplicates already
		# inside the source dataset, which the manifest cannot know about.
		orig = {ab for ab in p - truth
			if files[ab[0]]["class"] == files[ab[1]]["class"] == "original"}
		known = p - orig
		print(f"  precision excluding original~original "
			f"({len(orig)} pairs): "
			f"{len(known & truth) / len(known) if known else 1.0:.4f}")
		by_cls = collections.defaultdict(lambda: [0, 0])
		for a, b in truth:
			c = expected(a, b, files[a], files[b])
			by_cls[c][0] += (a, b) in p
			by_cls[c][1] += 1
		for c, (hit, n) in sorted(by_cls.items()):
			print(f"  recall {c:8} {hit:6}/{n:<6} {hit / n:.3f}")
		# Each variant against its own original(s).
		origs = collections.defaultdict(list)
		for f, m in files.items():
			if m["class"] == "original":
				key = m.get("source") or m["ranges"][0][0]
				origs[key].append(f)
		by_t = collections.defaultdict(lambda: [0, 0])
		for f, m in files.items():
			if m["class"] == "original":
				continue
			cands = [o for o in origs[m.get("source") or m["ranges"][0][0]]
				if related(kind, m, files[o])]
			if not cands:
				continue
			hit = any(tuple(sorted((f, o))) in p for o in cands)
			t = m["transform"] + ("+copy" if "copy_of" in m else "")
			by_t[t][0] += hit
			by_t[t][1] += 1
		print("  variant found with its original, by transform:")
		for t, (hit, n) in sorted(by_t.items(), key=lambda x: x[1][0] / x[1][1]):
			print(f"    {t:16} {hit:5}/{n:<5} {hit / n:.3f}")
		vs_leader = collections.Counter()
		for leader, rest in led[kind]:
			for f in rest:
				ml, mf = files[leader], files[f]
				exp = expected(f, leader, mf, ml) \
					if related(kind, mf, ml) else "unrelated"
				vs_leader[(exp, member[f][0])] += 1
		wrong = sum(n for (exp, _), n in vs_leader.items() if exp == "unrelated")
		total = sum(vs_leader.values())
		print(f"  leader precision {1 - wrong / max(total, 1):.4f} "
			f"({wrong} of {total} members unrelated to their leader)")
		print("  member vs leader, expected -> reported:")
		for (exp, got), n in sorted(vs_leader.items()):
			print(f"    {exp:9} -> {got:8} {n}")
		fps = sorted(p - truth - orig, key=lambda ab: -min(member[ab[0]][1],
			member[ab[1]][1]))
		print(f"  other false positive pairs: {len(fps)}; strongest:")
		for a, b in fps[:12]:
			ma, mb = files[a], files[b]
			print(f"    {member[a][0]}/{member[a][1]:.3f} "
				f"{ma['transform']}:{a}  ~  {mb['transform']}:{b}")


if __name__ == "__main__":
	main()
