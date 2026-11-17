
% NeuroSentinel - Expanded EIT noise-only threshold audit
% Independent of the existing anomaly localization outputs.
% Synthetic EIT simulation only.

clear;
clc;
format long g;

root_dir = 'C:/NeuroSentinel';
eidors_dir = [root_dir '/EIDORS/eidors-v3.12-ng/eidors-v3.12-ng/eidors'];
source_file = [root_dir '/eit/results/EIT_Simulation_Data.mat'];

output_dir = ...
    [root_dir '/eit/results/unknown_size_localization_threshold_audit_v2'];

if exist(output_dir, 'dir')
    error('Audit folder exists. Do not overwrite previous results.');
end

if ~exist(source_file, 'file')
    error('Original EIT simulation file not found.');
end

cd(eidors_dir);
eidors_startup;

S = load(source_file);

imdl = select_imdl(S.fmdl, {'Basic GN dif'});
imdl.hyperparameter.value = 1e-3;
imdl.RtR_prior = @prior_laplace;
imdl.jacobian_bkgnd.value = 1;
imdl.inv_solve.calc_solution_error = 0;

noise_levels = [0, 1, 5, 10];
signal_scale = std(S.vi.meas(:));

if ~isfinite(signal_scale) || signal_scale <= 0
    error('Invalid signal scale.');
end

noise_sds = (noise_levels / 100) * signal_scale;

% Independent random stream from the original anomaly experiment
randn('state', 20261110);

mkdir(output_dir);

% At 0% noise, one deterministic control is sufficient.
% At other levels, use 100 calibration and 100 independent controls.
n_levels = numel(noise_levels);
expected_rows = 1 + 100 * (n_levels - 1);

calibration = zeros(expected_rows, 5);
independent = zeros(expected_rows, 6);

cal_row = 0;
test_row = 0;

for ni = 1:n_levels

    pct = noise_levels(ni);
    sigma = noise_sds(ni);

    n_cal = 1;
    n_test = 1;

    if pct > 0
        n_cal = 100;
        n_test = 100;
    end

    % --------------------------------------------------------
    % Calibration controls
    % --------------------------------------------------------

    peaks = zeros(n_cal, 1);

    for ri = 1:n_cal

        baseline_data = S.vh;
        target_data = S.vh;

        if sigma > 0
            baseline_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));

            target_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));
        end

        rec = inv_solve(imdl, baseline_data, target_data);
        score = max(-rec.elem_data(:), 0);

        if any(~isfinite(score))
            error('Non-finite calibration reconstruction.');
        end

        peaks(ri) = max(score);

        cal_row = cal_row + 1;
        calibration(cal_row,:) = [
            pct, ri, sigma, peaks(ri), NaN
        ];
    end

    % Fixed rule: maximum calibration noise-only peak
    threshold = max(peaks);

    if ~isfinite(threshold)
        error('Invalid calibrated detection threshold.');
    end

    calibration( ...
        cal_row-n_cal+1:cal_row, 5) = threshold;

    % --------------------------------------------------------
    % Independent noise-only controls
    % --------------------------------------------------------

    false_alarms = 0;

    for ri = 1:n_test

        baseline_data = S.vh;
        target_data = S.vh;

        if sigma > 0
            baseline_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));

            target_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));
        end

        rec = inv_solve(imdl, baseline_data, target_data);
        score = max(-rec.elem_data(:), 0);

        if any(~isfinite(score))
            error('Non-finite independent control reconstruction.');
        end

        peak = max(score);
        false_alarm = peak > threshold + 1e-12;

        false_alarms = false_alarms + double(false_alarm);

        test_row = test_row + 1;

        independent(test_row,:) = [
            pct, ri, sigma, peak, threshold, double(false_alarm)
        ];
    end

    fprintf(['Noise %d%%: calibration=%d, independent=%d, ' ...
        'false alarms=%d\n'], ...
        pct, n_cal, n_test, false_alarms);
end

% ------------------------------------------------------------
% Save expanded calibration results
% ------------------------------------------------------------

fid = fopen([output_dir '/Expanded_Null_Calibration.csv'], 'w');

if fid < 0
    error('Cannot write calibration results.');
end

fprintf(fid, ...
    'Noise_Percent,Control_ID,Noise_SD,Peak_Score,Detection_Threshold\n');

for k = 1:cal_row
    fprintf(fid, '%g,%d,%.12g,%.12g,%.12g\n', calibration(k,:));
end

fclose(fid);

% ------------------------------------------------------------
% Save independent false-alarm results
% ------------------------------------------------------------

fid = fopen([output_dir '/Expanded_Independent_Null_Controls.csv'], 'w');

if fid < 0
    error('Cannot write independent control results.');
end

fprintf(fid, ...
    'Noise_Percent,Control_ID,Noise_SD,Peak_Score,Detection_Threshold,False_Alarm\n');

for k = 1:test_row
    fprintf(fid, '%g,%d,%.12g,%.12g,%.12g,%d\n', independent(k,:));
end

fclose(fid);

% ------------------------------------------------------------
% Save protocol metadata
% ------------------------------------------------------------

fid = fopen([output_dir '/Threshold_Audit_Metadata.txt'], 'w');

fprintf(fid, 'NEUROSENTINEL EXPANDED EIT NULL THRESHOLD AUDIT\n');
fprintf(fid, '==============================================\n\n');
fprintf(fid, 'Synthetic EIDORS simulation only.\n');
fprintf(fid, 'EIDORS version: 3.12\n');
fprintf(fid, 'Mesh elements: %d\n', size(S.fmdl.elems,1));
fprintf(fid, 'Measurement count: %d\n', numel(S.vh.meas));
fprintf(fid, 'Noise scale: percentage of SD of saved anomaly measurements.\n');
fprintf(fid, 'Noise applied independently to baseline and target measurements.\n');
fprintf(fid, 'Random seed: 20261110\n');
fprintf(fid, 'Calibration controls: 100 per nonzero noise level.\n');
fprintf(fid, 'Independent controls: 100 per nonzero noise level.\n');
fprintf(fid, 'Threshold: maximum calibration noise-only peak at each level.\n');
fprintf(fid, 'Detection: independent peak must exceed calibrated threshold.\n');
fprintf(fid, 'This audit does not use anomaly masks or anomaly labels.\n');
fprintf(fid, 'It does not establish clinical performance.\n');

fclose(fid);

fprintf('\nEXPANDED NULL THRESHOLD AUDIT COMPLETED\n');
fprintf('Calibration records: %d\n', cal_row);
fprintf('Independent controls: %d\n', test_row);
fprintf('Output: %s\n', output_dir);
