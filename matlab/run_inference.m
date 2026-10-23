function status = run_inference(configPath, inputPath, outputPath, pythonExe)
%RUN_INFERENCE Call the Python CLI: python -m piv.cli infer --config ... --input ... --output ...
%   Paths may contain spaces. Errors are not swallowed: a non-zero exit code raises an
%   error that includes the captured log.
if nargin < 4 || isempty(pythonExe), pythonExe = 'python'; end
repo = fileparts(fileparts(mfilename('fullpath')));
q = @(s) ['"', char(s), '"'];
cmd = strjoin({q(pythonExe), '-m', 'piv.cli', 'infer', '--config', q(configPath), ...
    '--input', q(inputPath), '--output', q(outputPath)}, ' ');
old = getenv('PYTHONPATH');
setenv('PYTHONPATH', [fullfile(repo, 'src'), pathsep, old]);
cleanup = onCleanup(@() setenv('PYTHONPATH', old));
here = pwd; cd(repo); back = onCleanup(@() cd(here));
[status, out] = system(cmd);
logFile = [char(outputPath), '.matlab.log'];
fid = fopen(logFile, 'w'); fprintf(fid, '%s\n%s', cmd, out); fclose(fid);
if status ~= 0
    error('run_inference:failed', 'Python CLI failed (exit %d). Log: %s\n%s', status, logFile, out);
end
fprintf('%s\n', out);
end
