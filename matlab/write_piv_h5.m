function write_piv_h5(path, images, t_a, t_b, index_a, index_b, gt, valid, calib, attrs)
%WRITE_PIV_H5 Write a schema-v1 CONTINUOUS dataset from MATLAB arrays in LOGICAL order.
%   images  [T,H,W] uint8/uint16    index_a/index_b [N] (0-based frame indices!)
%   t_a/t_b [N] seconds              gt [N,2,H,W] single or []     valid [N,H,W] logical or []
%   calib   struct (mode, sx_m_per_px, sy_m_per_px) or []          attrs struct (case_id, run_id, bit_depth ...)
%   Arrays are permuted to MATLAB's reversed view before h5write so that h5py reads
%   exactly [T,H,W] / [N,2,H,W]. Verified by validate_roundtrip.m.
if exist(path, 'file'), delete(path); end
local_write(path, '/images', images, 'T,H,W');
local_write(path, '/pairs/index_a', int64(index_a(:)), 'N');
local_write(path, '/pairs/index_b', int64(index_b(:)), 'N');
local_write(path, '/time/t_a', double(t_a(:)), 'N');
local_write(path, '/time/t_b', double(t_b(:)), 'N');
if ~isempty(gt),    local_write(path, '/gt/displacement', single(gt), 'N,C,H,W'); end
if ~isempty(valid), local_write(path, '/masks/valid', uint8(valid), 'N,H,W'); end
h5writeatt(path, '/', 'schema_version', '1.0');
h5writeatt(path, '/', 'acquisition_mode', 'continuous');
fn = fieldnames(attrs);
for k = 1:numel(fn), h5writeatt(path, '/', fn{k}, attrs.(fn{k})); end
if ~isempty(calib)
    local_write(path, '/calibration/_placeholder', uint8(0), 'scalar');
    fn = fieldnames(calib);
    for k = 1:numel(fn), h5writeatt(path, '/calibration', fn{k}, calib.(fn{k})); end
end
end

function local_write(path, name, x, axes)
nd = ndims(x);
if isvector(x) && nd == 2, xs = x(:); sz = numel(x);
else, xs = permute(x, nd:-1:1); sz = size(xs); end
h5create(path, name, sz, 'Datatype', class(x));
h5write(path, name, xs);
h5writeatt(path, name, 'axes', axes);
end
