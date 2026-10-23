#!/usr/bin/env bash
# Fetch third-party sources at pinned commits into third_party/ (not committed; own licences).
#  - PIV-LiteFlowNet-en (research use only, commercial use needs authors' consent) : code + Caffe weights
#  - piv_liteflownet-pytorch (MIT)  : independent port, used only for cross-checking
#  - PIVlab (MIT)                    : cross-correlation comparator (run in MATLAB or GNU Octave)
#  - PIV_dataset (authors' dataset generator; licence per its README)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
fetch() {  # url dir commit
  if [ ! -d "third_party/$2/.git" ]; then git clone "$1" "third_party/$2"; fi
  git -C "third_party/$2" fetch --depth 1 origin "$3" 2>/dev/null || git -C "third_party/$2" fetch origin
  git -C "third_party/$2" checkout -q "$3"
  echo "$2 @ $(git -C third_party/$2 rev-parse HEAD)"
}
fetch https://github.com/shengzesnail/PIV-LiteFlowNet-en.git    PIV-LiteFlowNet-en      128768b53a52c245883b4048cdd57698617ca4cf
fetch https://github.com/abrosua/piv_liteflownet-pytorch.git    piv_liteflownet-pytorch dae393386be7a1c3abf9620ab7eb4d81d7091733
fetch https://github.com/Shrediquette/PIVlab.git                PIVlab                  ec255f2c1a87aa210577b09d422bd90a4629c9ea
fetch https://github.com/shengzesnail/PIV_dataset.git           PIV_dataset             5fecd47581b0b8277c046ff76e452a1131d4d6b4
W=third_party/PIV-LiteFlowNet-en/caffe/models/PIV-LiteFlowNet-en/PIV-LiteFlowNet-en.caffemodel
EXP=8b9a0c76eb85e5a8988654fd0b165ee0fb73088d4d3d039208567e6e96fd868b
GOT=$(python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$W")
[ "$GOT" = "$EXP" ] && echo "caffemodel sha256 OK" || { echo "caffemodel sha256 MISMATCH: $GOT"; exit 1; }
# Octave-compatible PIVlab adapter (interface-only changes) is regenerated from the pinned source
python comparators/pivlab_octave/make_octave_adapter.py third_party/PIVlab comparators/pivlab_octave
