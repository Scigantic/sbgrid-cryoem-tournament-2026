#!/usr/bin/env bash
#SBATCH -J ma_id
#SBATCH -p ct03g-g4dn-xlarge
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --no-requeue
#SBATCH -o slurm-%j.out
# Q1: identify the protein from map03. ModelAngelo builds without a sequence, then its
# per-chain HMM profiles are searched against the PDB chain sequences shipped with BALBES.
# Prints only counts, PDB chain ids and E-values (no sequences).
set -euo pipefail
module load sbgrid/modelangelo/1.0.18
B=$HOME/results/blush/Post_it010
MASK=$HOME/results/focus/Focus0/mask.mrc
BUNDLE=/programs/share/relion/hub/checkpoints/model_angelo_v1.0/nucleotides_no_seq
DB=/programs/share/balbes/1.1.5/seq_DB
W=$HOME/results/ma
mkdir -p "$W"
cd "$W"
ls "$BUNDLE" > /dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== fasta $(date +%T)"
python3 - "$DB" "$W/pdb_chains.fasta" <<'EOF'
import os, sys
src, out = sys.argv[1], sys.argv[2]
n = 0
with open(out, 'w') as o:
    for fn in sorted(os.listdir(src)):
        if not fn.endswith('.seq'):
            continue
        seq = []
        for line in open(os.path.join(src, fn), errors='replace'):
            s = line.strip()
            if s and not s.startswith(('>', '#')):
                seq.append(''.join(c for c in s if c.isalpha()).upper())
        s = ''.join(seq)
        if len(s) >= 20:
            o.write('>%s\n%s\n' % (fn[:-4], s))
            n += 1
print('fasta: %d chains' % n)
EOF

echo "== build_no_seq $(date +%T)"
model_angelo build_no_seq -v "$B/pp.mrc" -m "$MASK" -o "$W/build" \
  -d 0 --model-bundle-path "$BUNDLE" > "$W/build.log" 2>&1
echo "== build done $(date +%T)"
ls "$W" "$W/build" | head -30
python3 - "$W" <<'EOF'
import glob, sys, collections
cifs = sorted(glob.glob(sys.argv[1] + '/build*.cif') + glob.glob(sys.argv[1] + '/build/*.cif'))
for c in cifs:
    ca = collections.Counter()
    for line in open(c):
        if line.startswith('ATOM') and ' CA ' in line:
            f = line.split()
            ca[f[6] if len(f) > 6 else '?'] += 1
    lens = sorted(ca.values(), reverse=True)
    print('%s: %d chains, %d residues, longest %s' % (
        c.split('/')[-1], len(lens), sum(lens), lens[:8]))
EOF

echo "== hmm_search $(date +%T)"
model_angelo hmm_search -i "$W/build" -f "$W/pdb_chains.fasta" -o "$W/hmm" \
  --model-bundle-path "$BUNDLE" > "$W/hmm.log" 2>&1
echo "== hmm done $(date +%T)"
ls "$W/hmm" | head -20
python3 - "$W/hmm" <<'EOF'
import csv, glob, os, re, sys, collections
top = collections.Counter()
for p in sorted(glob.glob(sys.argv[1] + '/**/*', recursive=True)):
    if not p.endswith(('.csv', '.tsv', '.txt', '.tbl')):
        continue
    rows = list(csv.reader(open(p), delimiter='\t' if p.endswith('.tsv') else ','))
    if not rows:
        continue
    hdr = [h.strip().lower() for h in rows[0]]
    keep = [i for i, h in enumerate(hdr)
            if re.search(r'name|target|query|chain|e.?value|score|bias', h)
            and 'seq' not in h and 'align' not in h]
    print('-- %s (%d rows) cols: %s' % (os.path.relpath(p, sys.argv[1]), len(rows) - 1,
                                       [hdr[i] for i in keep]))
    for r in rows[1:6]:
        print('   ' + ' | '.join(r[i][:24] for i in keep if i < len(r)))
    t = [i for i, h in enumerate(hdr) if re.search(r'target|name', h) and 'query' not in h]
    if t and len(rows) > 1 and t[0] < len(rows[1]):
        top[rows[1][t[0]][:4].upper()] += 1
print('RESULT top hit per profile, by PDB entry: %s' % top.most_common(10))
EOF
echo "== done $(date +%T)"
