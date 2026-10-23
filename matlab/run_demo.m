function run_demo(pythonExe)
%RUN_DEMO One-command CPU demo through the Python CLI, then plot the first result.
if nargin < 1 || isempty(pythonExe), pythonExe = 'python'; end
repo = fileparts(fileparts(mfilename('fullpath')));
old = getenv('PYTHONPATH');
setenv('PYTHONPATH', [fullfile(repo, 'src'), pathsep, old]);
cleanup = onCleanup(@() setenv('PYTHONPATH', old));
here = pwd; cd(repo); back = onCleanup(@() cd(here));
[status, out] = system(['"', pythonExe, '" -m piv.cli demo --out outputs/demo_matlab']);
disp(out);
if status ~= 0, error('run_demo:failed', 'demo failed (exit %d)', status); end
d = dir(fullfile(repo, 'outputs', 'demo_matlab', 'data', '**', '*cellular*.h5'));
p = dir(fullfile(repo, 'outputs', 'demo_matlab', 'pred', '*cellular*.h5'));
plot_results(fullfile(d(1).folder, d(1).name), fullfile(p(1).folder, p(1).name), 1);
end
