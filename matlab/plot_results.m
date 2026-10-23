function plot_results(datasetPath, predictionPath, k)
%PLOT_RESULTS Quiver of prediction vs GT for pair k (1-based) and EPE map.
if nargin < 3, k = 1; end
D = read_piv_h5(datasetPath);
P = read_piv_h5(predictionPath);
pred = squeeze(P.prediction__displacement(k, :, :, :));     % [2,H,W]
figure('Color', 'w');
step = 8;
[H, W] = deal(size(pred, 2), size(pred, 3));
[X, Y] = meshgrid(0:step:W-1, 0:step:H-1);
sub = @(c) squeeze(pred(c, 1:step:end, 1:step:end));
if isfield(D, 'gt__displacement')
    gt = squeeze(D.gt__displacement(k, :, :, :));
    subplot(1, 2, 1);
    quiver(X, Y, squeeze(gt(1, 1:step:end, 1:step:end)), squeeze(gt(2, 1:step:end, 1:step:end)), 0, 'k'); hold on
    quiver(X, Y, sub(1), sub(2), 0, 'r'); axis ij equal tight; legend('GT', 'prediction');
    title('displacement (pixel/pair), y down');
    subplot(1, 2, 2);
    imagesc(squeeze(sqrt(sum((pred - gt).^2, 1)))); axis ij equal tight; colorbar; title('EPE (pixel)');
else
    quiver(X, Y, sub(1), sub(2), 0, 'r'); axis ij equal tight; title('prediction (pixel/pair)');
end
end
