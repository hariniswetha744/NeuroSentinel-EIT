
% NeuroSentinel EIT - Reproducible Validation
% Reads existing simulation results and saves validation metrics.

clear;
clc;

result_dir = 'C:/NeuroSentinel/eit/results';
mat_file = fullfile(result_dir, 'EIT_Simulation_Data.mat');

if ~exist(mat_file, 'file')
    error('Simulation MAT file not found: %s', mat_file);
end

S = load(mat_file);

% Check mesh, images, and measurements
n_nodes = size(S.fmdl.nodes, 1);
n_elems = size(S.fmdl.elems, 1);
region = logical(S.region(:));
recon = S.img_rec.elem_data(:);
truth_change = S.img_i.elem_data(:) - S.img_h.elem_data(:);

if numel(recon) ~= n_elems || numel(truth_change) ~= n_elems || numel(region) ~= n_elems
    error('Mesh, reconstruction, truth, and region dimensions do not match.');
end

if ~any(region) || ~any(~region)
    error('Anomaly and background regions must both contain elements.');
end

if any(~isfinite([recon; truth_change; S.vh.meas(:); S.vi.meas(:); S.delta_meas(:)]))
    error('Non-finite values detected in EIT validation inputs.');
end

if isempty(S.vh.meas) || numel(S.vi.meas) ~= numel(S.vh.meas) || numel(S.delta_meas) ~= numel(S.vi.meas)
    error('Baseline, anomaly, and delta measurement dimensions are inconsistent.');
end

% Measurement consistency
measured_delta = S.vi.meas(:) - S.vh.meas(:);
delta_error = max(abs(measured_delta - S.delta_meas(:)));

% Reconstruction error
rmse = sqrt(mean((recon - truth_change).^2));
mae = mean(abs(recon - truth_change));
mean_inside = mean(recon(region));
mean_outside = mean(recon(~region));
min_inside = min(recon(region));
min_outside = min(recon(~region));

% Localization using the same number of elements as the true anomaly
n_true = nnz(region);
[~, order] = sort(recon, 'ascend');
predicted = false(size(region));
predicted(order(1:n_true)) = true;

overlap = nnz(predicted & region);
union_count = nnz(predicted | region);
dice_score = 2 * overlap / (nnz(predicted) + nnz(region));
iou_score = overlap / union_count;
false_positive = nnz(predicted & ~region);
false_negative = nnz(region & ~predicted);

% Display results
fprintf('\n--- NEUROSENTINEL EIT VALIDATION ---\n');
fprintf('Mesh nodes: %d\n', n_nodes);
fprintf('Mesh elements: %d\n', n_elems);
fprintf('Baseline measurements: %d\n', numel(S.vh.meas));
fprintf('Anomaly measurements: %d\n', numel(S.vi.meas));
fprintf('True anomaly elements: %d\n', n_true);
fprintf('Measurement consistency error: %.12g\n', delta_error);
fprintf('RMSE: %.8f\n', rmse);
fprintf('MAE: %.8f\n', mae);
fprintf('Mean reconstruction inside anomaly: %.8f\n', mean_inside);
fprintf('Mean reconstruction outside anomaly: %.8f\n', mean_outside);
fprintf('Minimum inside anomaly: %.8f\n', min_inside);
fprintf('Minimum outside anomaly: %.8f\n', min_outside);
fprintf('Localization overlap: %d/%d\n', overlap, n_true);
fprintf('False positives: %d\n', false_positive);
fprintf('False negatives: %d\n', false_negative);
fprintf('Dice score: %.4f\n', dice_score);
fprintf('IoU score: %.4f\n', iou_score);

% Save named metrics to CSV
metric_names = {
    'mesh_nodes';
    'mesh_elements';
    'baseline_measurements';
    'anomaly_measurements';
    'true_anomaly_elements';
    'measurement_consistency_error';
    'RMSE';
    'MAE';
    'mean_reconstruction_inside';
    'mean_reconstruction_outside';
    'minimum_inside';
    'minimum_outside';
    'overlap_elements';
    'false_positives';
    'false_negatives';
    'Dice';
    'IoU'
};

metric_values = [
    n_nodes;
    n_elems;
    numel(S.vh.meas);
    numel(S.vi.meas);
    n_true;
    delta_error;
    rmse;
    mae;
    mean_inside;
    mean_outside;
    min_inside;
    min_outside;
    overlap;
    false_positive;
    false_negative;
    dice_score;
    iou_score
];

csv_path = fullfile(result_dir, 'EIT_Reproducible_Validation_Metrics.csv');
fid_csv = fopen(csv_path, 'w');

if fid_csv == -1
    error('Could not create validation metrics CSV.');
end

fprintf(fid_csv, 'metric,value\n');
for k = 1:numel(metric_names)
    fprintf(fid_csv, '%s,%.12g\n', metric_names{k}, metric_values(k));
end
fclose(fid_csv);

fprintf('\nMetrics saved to:\n%s\n', csv_path);
fprintf('EIT validation script completed.\n');
