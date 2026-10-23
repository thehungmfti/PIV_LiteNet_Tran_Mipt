# Windows PowerShell equivalent of fetch_third_party.sh (requires git and python on PATH)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force third_party | Out-Null
function Fetch($url, $dir, $commit) {
  if (-not (Test-Path "third_party/$dir/.git")) { git clone $url "third_party/$dir" }
  git -C "third_party/$dir" fetch origin
  git -C "third_party/$dir" checkout -q $commit
  Write-Host "$dir @ $(git -C third_party/$dir rev-parse HEAD)"
}
Fetch https://github.com/shengzesnail/PIV-LiteFlowNet-en.git PIV-LiteFlowNet-en 128768b53a52c245883b4048cdd57698617ca4cf
Fetch https://github.com/abrosua/piv_liteflownet-pytorch.git piv_liteflownet-pytorch dae393386be7a1c3abf9620ab7eb4d81d7091733
Fetch https://github.com/Shrediquette/PIVlab.git PIVlab ec255f2c1a87aa210577b09d422bd90a4629c9ea
Fetch https://github.com/shengzesnail/PIV_dataset.git PIV_dataset 5fecd47581b0b8277c046ff76e452a1131d4d6b4
$h = (Get-FileHash third_party/PIV-LiteFlowNet-en/caffe/models/PIV-LiteFlowNet-en/PIV-LiteFlowNet-en.caffemodel -Algorithm SHA256).Hash.ToLower()
if ($h -ne "8b9a0c76eb85e5a8988654fd0b165ee0fb73088d4d3d039208567e6e96fd868b") { throw "caffemodel sha256 mismatch: $h" }
Write-Host "caffemodel sha256 OK"
python comparators/pivlab_octave/make_octave_adapter.py third_party/PIVlab comparators/pivlab_octave
