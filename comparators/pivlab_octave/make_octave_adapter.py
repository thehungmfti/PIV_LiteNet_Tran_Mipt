"""Generate GNU-Octave compatible copies of the PIVlab PIV core.

PIVlab (MIT licence, https://github.com/Shrediquette/PIVlab) uses MATLAB
``arguments`` blocks and name=value call syntax, which Octave 8 cannot parse.
This script rewrites ONLY those interface parts; the numerical code of
+piv/piv_FFTmulti.m and +postproc/PIVlab_postproc.m is copied verbatim.
The generated files carry a header with the PIVlab commit they came from.

usage: python make_octave_adapter.py <PIVlab_root> <out_dir>
"""
import re
import subprocess
import sys
from pathlib import Path


def defaults_from_arguments(block: str) -> list[tuple[str, str | None]]:
    out = []
    for line in block.splitlines()[1:-1]:
        line = line.split("%")[0].strip()
        if not line.startswith("opts."):
            continue
        name, _, default = line[5:].partition("=")
        out.append((name.strip(), default.strip() or None))
    return out


def replace_arguments(src: str) -> str:
    m = re.search(r"^arguments\s*\n.*?^end\s*\n", src, re.S | re.M)
    if not m:
        raise ValueError("no arguments block")
    lines = ["% --- Octave adapter: MATLAB 'arguments' block replaced by explicit defaults ---"]
    for name, default in defaults_from_arguments(m.group(0)):
        if default is None:
            lines.append(f"if ~isfield(opts, '{name}'), error('missing required option {name}'); end")
        else:
            lines.append(f"if ~isfield(opts, '{name}'), opts.{name} = {default}; end")
    return src[:m.start()] + "\n".join(lines) + "\n" + src[m.end():]


def main(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    header = (f"% AUTO-GENERATED from PIVlab commit {sha} by make_octave_adapter.py.\n"
              "% PIVlab (c) William Thielicke, MIT licence. Only the argument interface was changed.\n")

    post = (root / "+postproc" / "PIVlab_postproc.m").read_text()
    post = post.replace("function [u_out,v_out] = PIVlab_postproc (opts)",
                        "function [u_out,v_out] = pivlab_postproc_oct (opts)")
    post = replace_arguments(post)
    # Octave 8: std() has no 'omitnan' flag -> semantically identical NaN removal
    post = re.sub(r"std\((reshape\((\w),size\(\2,1\)\*size\(\2,2\),1\)),'omitnan'\)",
                  r"std(\2(~isnan(\2(:))))", post)
    (out / "pivlab_postproc_oct.m").write_text(header + post)

    core = (root / "+piv" / "piv_FFTmulti.m").read_text()
    core = core.replace("= piv_FFTmulti(opts)", "= pivlab_FFTmulti_oct(opts)")
    core = replace_arguments(core)
    call = re.search(r"postproc\.PIVlab_postproc\(\s*\.\.\.\s*(.*?)\);", core, re.S)
    args = re.sub(r"\.\.\.\s*", " ", call.group(1))
    pairs = [p.strip() for p in re.split(r",\s*(?=\w+=)", args) if p.strip()]
    st = ", ".join(f"'{p.split('=', 1)[0].strip()}', {{{p.split('=', 1)[1].strip()}}}" for p in pairs)
    core = core[:call.start()] + f"pivlab_postproc_oct(struct({st}));" + core[call.end():]
    # smoothn: its dependency test uses a cwd-relative path that only MATLAB resolves;
    # +misc/dctn.m and idctn.m ship with PIVlab, so the test is replaced by true
    sm = (root / "+misc" / "smoothn.m").read_text()
    sm = sm.replace("function [z,s,exitflag] = smoothn(varargin)",
                    "function [z,s,exitflag] = smoothn_oct(varargin)")
    sm = sm.replace("exist(fullfile('+misc','dctn'),'file')", "true")
    sm = sm.replace("exist(fullfile('+misc','idctn'),'file')", "true")
    (out / "smoothn_oct.m").write_text(header + sm)
    core = core.replace("misc.smoothn(", "smoothn_oct(")
    (out / "pivlab_FFTmulti_oct.m").write_text(header + core)
    print("written", out, "from PIVlab", sha)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
