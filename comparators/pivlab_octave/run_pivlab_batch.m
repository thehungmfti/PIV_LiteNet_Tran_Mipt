function run_pivlab_batch(in_mat, out_mat, pivlab_root)
% Batch PIVlab (FFT multi-pass, window deformation) on image pairs stored in in_mat.
% in_mat fields: images_a, images_b [N,H,W] uint8; ia1, step, passes, ia2, ia3, ia4,
%   subpix, imdeform, repeat_last; validation: do_stdev, stdthresh, do_median, neigh_thresh.
% out_mat: x, y (1-based PIVlab coords), U, V (raw) and Uf, Vf (validated + inpainted) [N,ny,nx].
pkg load image
addpath(pivlab_root);
addpath(fileparts(mfilename('fullpath')));
S = load(in_mat);
N = size(S.images_a, 1);
for k = 1:N
    im1 = reshape(S.images_a(k, :, :), size(S.images_a, 2), size(S.images_a, 3));
    im2 = reshape(S.images_b(k, :, :), size(S.images_b, 2), size(S.images_b, 3));
    o = struct('image1', im1, 'image2', im2, 'interrogationarea', double(S.ia1), 'step', double(S.step), ...
        'subpixfinder', double(S.subpix), 'passes', double(S.passes), 'int2', double(S.ia2), ...
        'int3', double(S.ia3), 'int4', double(S.ia4), 'imdeform', char(S.imdeform), 'repeat', 0, ...
        'mask_auto', 0, 'do_linear_correlation', 0, 'repeat_last_pass', double(S.repeat_last), ...
        'delta_diff_min', 0.025, 'limit_peak_search_area', 1);
    [x, y, u, v, tv] = pivlab_FFTmulti_oct(o);
    if k == 1
        U = zeros([N, size(u)], 'single'); V = U; Uf = U; Vf = U;
    end
    U(k, :, :) = u; V(k, :, :) = v;
    [uf, vf] = pivlab_postproc_oct(struct('u', {u}, 'v', {v}, 'valid_vel', {[]}, ...
        'do_stdev_check', {double(S.do_stdev)}, 'stdthresh', {double(S.stdthresh)}, ...
        'do_local_median', {double(S.do_median)}, 'neigh_thresh', {double(S.neigh_thresh)}));
    if any(isnan(uf(:))) || any(isnan(vf(:)))
        uf = misc.inpaint_nans(double(uf), 4); vf = misc.inpaint_nans(double(vf), 4);
    end
    Uf(k, :, :) = uf; Vf(k, :, :) = vf;
end
save('-v7', out_mat, 'x', 'y', 'U', 'V', 'Uf', 'Vf');
end
