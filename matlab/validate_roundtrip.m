function ok = validate_roundtrip(pythonExe)
%VALIDATE_ROUNDTRIP MATLAB <-> Python HDF5 axis-order check with axis-distinct tensors.
%   1) MATLAB writes T(n,c,h,w) = 1000n + 100c + 10h + w (1-based) via write_piv_h5,
%      Python (h5py) checks that it reads exactly that tensor.
%   2) Python writes the same pattern, MATLAB reads it via read_piv_h5 and checks.
if nargin < 1 || isempty(pythonExe), pythonExe = 'python'; end
repo = fileparts(fileparts(mfilename('fullpath')));
tmp = fullfile(tempdir, 'piv roundtrip');            % path with a space on purpose
if ~exist(tmp, 'dir'), mkdir(tmp); end
N = 3; C = 2; H = 4; W = 5;
[n, c, h, w] = ndgrid(1:N, 1:C, 1:H, 1:W);
G = single(1000*n + 100*c + 10*h + w);
imgs = uint8(mod(reshape(1:(N+1)*H*W, [N+1, H, W]), 255));
f1 = fullfile(tmp, 'from_matlab.h5');
write_piv_h5(f1, imgs, (0:N-1)*1e-3, (1:N)*1e-3, 0:N-1, 1:N, G, true(N, H, W), [], ...
    struct('case_id', 'rt', 'run_id', 'rt', 'bit_depth', 8));
script = fullfile(repo, 'scripts', 'roundtrip_check.py');
[s1, o1] = system(sprintf('"%s" "%s" check "%s"', pythonExe, script, f1));
f2 = fullfile(tmp, 'from_python.h5');
[s2, o2] = system(sprintf('"%s" "%s" write "%s"', pythonExe, script, f2));
S = read_piv_h5(f2);
ok2 = s2 == 0 && isequal(size(S.gt__displacement), [N, C, H, W]) && isequal(S.gt__displacement, G);
ok = (s1 == 0) && ok2;
fprintf('MATLAB->Python: %s\n%s\nPython->MATLAB: %s\n', mat2str(s1 == 0), o1, mat2str(ok2));
if ~ok, error('validate_roundtrip:failed', 'axis round-trip FAILED'); end
end
