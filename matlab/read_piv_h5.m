function S = read_piv_h5(path)
%READ_PIV_H5 Read a schema-v1 dataset or prediction file written by the Python backend.
%   S = read_piv_h5(path) returns every dataset in LOGICAL axis order, i.e. the order
%   written in the dataset's 'axes' attribute (e.g. /gt/displacement -> [N,C,H,W]).
%   h5read returns the reversed order (column-major view of a row-major file); this
%   adapter reverses it back. Validate with validate_roundtrip.m before trusting it.
%   Displacement channel 1 = dx (right), channel 2 = dy_image (down), pixel/pair.
info = h5info(path);
S = struct();
S.attrs = local_attrs(info.Attributes);
S = local_group(path, info, S);
end

function S = local_group(path, g, S)
for k = 1:numel(g.Datasets)
    name = [g.Name, '/', g.Datasets(k).Name];
    name = strrep(name, '//', '/');
    x = h5read(path, name);
    nd = numel(g.Datasets(k).Dataspace.Size);
    if nd > 1
        x = permute(x, nd:-1:1);            % back to logical (row-major) order
    end
    field = matlab.lang.makeValidName(strrep(name(2:end), '/', '__'));
    S.(field) = x;
    a = local_attrs(g.Datasets(k).Attributes);
    if isfield(a, 'axes'), S.([field '__axes']) = a.axes; end
end
for k = 1:numel(g.Groups)
    S = local_group(path, g.Groups(k), S);
    a = local_attrs(g.Groups(k).Attributes);
    if ~isempty(fieldnames(a))
        S.(matlab.lang.makeValidName([strrep(g.Groups(k).Name(2:end), '/', '__') '__attrs'])) = a;
    end
end
end

function a = local_attrs(list)
a = struct();
for k = 1:numel(list)
    v = list(k).Value;
    if iscell(v) && numel(v) == 1, v = v{1}; end
    a.(matlab.lang.makeValidName(list(k).Name)) = v;
end
end
