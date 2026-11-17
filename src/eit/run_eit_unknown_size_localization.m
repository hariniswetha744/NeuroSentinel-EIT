
% NeuroSentinel: Unknown-size EIT localization experiment
% Separate reproducible study. Does not modify original validation outputs.
% Synthetic EIT only. No clinical interpretation.

clear;
clc;
format long g;

root_dir = 'C:/NeuroSentinel';
eidors_dir = [root_dir '/EIDORS/eidors-v3.12-ng/eidors-v3.12-ng/eidors'];
source_file = [root_dir '/eit/results/EIT_Simulation_Data.mat'];
output_dir = [root_dir '/eit/results/unknown_size_localization'];

if exist(output_dir, 'dir')
    error('Output folder already exists. Preserve it; do not overwrite.');
end

if ~exist(source_file, 'file')
    error('Original EIT simulation file not found.');
end

cd(eidors_dir);
eidors_startup;

S = load(source_file);

if ~isfield(S, 'fmdl') || ~isfield(S, 'img_h') || ...
        ~isfield(S, 'vh') || ~isfield(S, 'vi')
    error('Required EIT simulation fields are missing.');
end

E = S.fmdl.elems;
N = S.fmdl.nodes;
n_elements = rows(E);

if n_elements ~= 576 || rows(N) ~= 313
    error('Unexpected source mesh dimensions.');
end

if numel(S.vh.meas) ~= 208 || numel(S.vi.meas) ~= 208
    error('Unexpected measurement count.');
end

% Triangle element centroids and areas
element_centroids = ...
    (N(E(:,1),:) + N(E(:,2),:) + N(E(:,3),:)) / 3;

x1 = N(E(:,1),1);
y1 = N(E(:,1),2);
x2 = N(E(:,2),1);
y2 = N(E(:,2),2);
x3 = N(E(:,3),1);
y3 = N(E(:,3),2);

element_area = 0.5 * abs( ...
    (x2-x1).*(y3-y1) - (x3-x1).*(y2-y1));

if any(element_area <= 0) || ...
        any(~isfinite(element_area))
    error('Invalid mesh element areas.');
end

% Five positions and two nominal radii
centers = [
     0.00,  0.00;
    -0.35,  0.00;
     0.35,  0.00;
     0.00,  0.35;
     0.00, -0.35
];

radii = [0.15, 0.22];
noise_levels = [0, 1, 5, 10];

% Noise expressed as a percentage of saved anomaly measurement SD
signal_scale = std(S.vi.meas(:));

if signal_scale <= 0 || ~isfinite(signal_scale)
    error('Invalid measurement noise reference scale.');
end

noise_sds = (noise_levels(:) / 100) * signal_scale;

% Fixed random seed
randn('state', 20261109);

% Explicit, documented inverse reconstruction configuration
imdl = select_imdl(S.fmdl, {'Basic GN dif'});
imdl.hyperparameter.value = 1e-3;
imdl.RtR_prior = @prior_laplace;
imdl.jacobian_bkgnd.value = 1;
imdl.inv_solve.calc_solution_error = 0;

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Output files
calibration_file = [output_dir '/Noise_Threshold_Calibration.csv'];
control_file = [output_dir '/No_Anomaly_Control_Results.csv'];
result_file = [output_dir '/EIT_Unknown_Size_Localization_Results.csv'];
summary_file = [output_dir '/EIT_Unknown_Size_Localization_Summary.csv'];
element_file = [output_dir '/Element_Level_Detection_Masks.csv'];
metadata_file = [output_dir '/Experiment_Metadata.txt'];

% ------------------------------------------------------------
% 1. CALIBRATE DETECTION GATE USING NO-ANOMALY CONTROLS
% ------------------------------------------------------------

calibration_peaks = zeros(numel(noise_levels), 10);
calibration_counts = zeros(numel(noise_levels), 1);
null_thresholds = zeros(numel(noise_levels), 1);

for ni = 1:numel(noise_levels)

    pct = noise_levels(ni);
    sigma = noise_sds(ni);

    n_cal = 1;
    if pct > 0
        n_cal = 10;
    end

    for ri = 1:n_cal

        baseline_data = S.vh;
        target_data = S.vh;

        if pct > 0
            baseline_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));

            target_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));
        end

        rec = inv_solve(imdl, baseline_data, target_data);
        score = max(-rec.elem_data(:), 0);

        calibration_peaks(ni, ri) = max(score);
    end

    calibration_counts(ni) = n_cal;
    null_thresholds(ni) = max( ...
        calibration_peaks(ni, 1:n_cal));
end

fid = fopen(calibration_file, 'w');
if fid < 0
    error('Cannot create calibration file.');
end

fprintf(fid, 'Noise_Percent,Noise_SD,Calibration_Controls,Peak_Detection_Threshold\n');

for ni = 1:numel(noise_levels)
    fprintf(fid, '%d,%.12g,%d,%.12g\n', ...
        noise_levels(ni), noise_sds(ni), ...
        calibration_counts(ni), null_thresholds(ni));
end
fclose(fid);

% ------------------------------------------------------------
% 2. INDEPENDENT NO-ANOMALY FALSE-ALARM CONTROLS
% ------------------------------------------------------------

control_rows = zeros(31, 7);
control_row = 0;

for ni = 1:numel(noise_levels)

    pct = noise_levels(ni);
    sigma = noise_sds(ni);

    n_control = 1;
    if pct > 0
        n_control = 10;
    end

    for ri = 1:n_control

        baseline_data = S.vh;
        target_data = S.vh;

        if pct > 0
            baseline_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));

            target_data.meas = S.vh.meas(:) + ...
                sigma * randn(size(S.vh.meas(:)));
        end

        rec = inv_solve(imdl, baseline_data, target_data);
        score = max(-rec.elem_data(:), 0);
        peak = max(score);

        false_alarm = peak > null_thresholds(ni) + 1e-12;

        predicted = false(n_elements, 1);
        if false_alarm
            predicted = score >= 0.5 * peak;
        end

        control_row = control_row + 1;
        control_rows(control_row,:) = [
            pct, ri, sigma, peak, null_thresholds(ni), ...
            double(false_alarm), nnz(predicted)
        ];
    end
end

fid = fopen(control_file, 'w');
if fid < 0
    error('Cannot create control results file.');
end

fprintf(fid, ['Noise_Percent,Repeat_ID,Noise_SD,Peak_Score,' ...
    'Detection_Threshold,False_Alarm,False_Alarm_Elements\n']);

for k = 1:control_row
    fprintf(fid, '%g,%d,%.12g,%.12g,%.12g,%d,%d\n', ...
        control_rows(k,:));
end
fclose(fid);

% ------------------------------------------------------------
% 3. MULTI-POSITION, MULTI-SIZE ANOMALY EXPERIMENT
% ------------------------------------------------------------

expected_rows = rows(centers) * numel(radii) * 31;
results = zeros(expected_rows, 20);
result_row = 0;
geometry_id = 0;

fid_elements = fopen(element_file, 'w');
if fid_elements < 0
    error('Cannot create element-level output file.');
end

fprintf(fid_elements, ['Scenario_ID,Element_ID,Centroid_X,Centroid_Y,' ...
    'Element_Area,Truth_Mask,Predicted_Mask,Anomaly_Score,' ...
    'Noise_Percent,Repeat_ID\n']);

for pi = 1:rows(centers)

    center = centers(pi,:);

    for rj = 1:numel(radii)

        radius = radii(rj);
        geometry_id = geometry_id + 1;

        % Truth is created for evaluation only.
        % It is never provided to the inverse solver or detector.
        truth = sum((element_centroids-center).^2, 2) <= radius^2;
        n_true = nnz(truth);

        if n_true < 3
            error('Anomaly mask contains too few elements.');
        end

        true_centroid = sum( ...
            bsxfun(@times, element_centroids(truth,:), ...
            element_area(truth)), 1) / sum(element_area(truth));

        % Forward simulation using original saved model
        anomaly_img = S.img_h;
        anomaly_img.elem_data = ones(n_elements, 1);
        anomaly_img.elem_data(truth) = 0.5;

        clean_target = fwd_solve(anomaly_img);

        if numel(clean_target.meas) ~= 208 || ...
                any(~isfinite(clean_target.meas(:)))
            error('Invalid simulated anomaly measurements.');
        end

        for ni = 1:numel(noise_levels)

            pct = noise_levels(ni);
            sigma = noise_sds(ni);

            n_repeats = 1;
            if pct > 0
                n_repeats = 10;
            end

            for rep = 1:n_repeats

                baseline_data = S.vh;
                target_data = clean_target;

                if pct > 0
                    baseline_data.meas = S.vh.meas(:) + ...
                        sigma * randn(size(S.vh.meas(:)));

                    target_data.meas = clean_target.meas(:) + ...
                        sigma * randn(size(clean_target.meas(:)));
                end

                % Reconstruction does not receive truth or true size
                rec = inv_solve(imdl, baseline_data, target_data);

                reconstruction = rec.elem_data(:);

                if numel(reconstruction) ~= n_elements || ...
                        any(~isfinite(reconstruction))
                    error('Invalid reconstructed element values.');
                end

                % Negative change represents the simulated conductivity reduction
                score = max(-reconstruction, 0);
                peak = max(score);

                % Absolute detection gate calibrated on separate null controls
                detected = peak > null_thresholds(ni) + 1e-12;

                % Half-maximum region segmentation, independent of true size
                predicted = false(n_elements, 1);

                if detected
                    predicted = score >= 0.5 * peak;
                end

                n_pred = nnz(predicted);
                overlap = nnz(predicted & truth);
                union_elements = nnz(predicted | truth);

                dice_elements = 2 * overlap / (n_pred + n_true);
                iou_elements = overlap / union_elements;

                intersection_area = sum(element_area(predicted & truth));
                predicted_area = sum(element_area(predicted));
                truth_area = sum(element_area(truth));
                union_area = sum(element_area(predicted | truth));

                dice_area = 2 * intersection_area / ...
                    (predicted_area + truth_area);

                iou_area = intersection_area / union_area;

                false_positive = nnz(predicted & ~truth);
                false_negative = nnz(truth & ~predicted);

                centroid_error = NaN;

                if n_pred > 0
                    predicted_centroid = sum( ...
                        bsxfun(@times, element_centroids(predicted,:), ...
                        element_area(predicted)), 1) / predicted_area;

                    centroid_error = norm( ...
                        predicted_centroid - true_centroid);
                end

                result_row = result_row + 1;

                results(result_row,:) = [
                    geometry_id, pi, center(1), center(2), radius, ...
                    n_true, pct, rep, sigma, peak, ...
                    null_thresholds(ni), double(detected), n_pred, ...
                    dice_elements, iou_elements, dice_area, iou_area, ...
                    centroid_error, false_positive, false_negative
                ];

                scenario_id = sprintf( ...
                    'P%02d_R%02d_N%02d_T%02d', pi, rj, ni, rep);

                for ei = 1:n_elements
                    fprintf(fid_elements, ...
                        '%s,%d,%.8f,%.8f,%.12g,%d,%d,%.12g,%d,%d\n', ...
                        scenario_id, ei, ...
                        element_centroids(ei,1), element_centroids(ei,2), ...
                        element_area(ei), truth(ei), predicted(ei), ...
                        score(ei), pct, rep);
                end
            end
        end
    end
end

fclose(fid_elements);

if result_row ~= expected_rows
    error('Unexpected number of anomaly trials.');
end

% ------------------------------------------------------------
% 4. WRITE TRIAL RESULTS
% ------------------------------------------------------------

fid = fopen(result_file, 'w');
if fid < 0
    error('Cannot create localization results file.');
end

fprintf(fid, ['Geometry_ID,Position_ID,Center_X,Center_Y,Nominal_Radius,' ...
    'True_Elements,Noise_Percent,Repeat_ID,Noise_SD,Peak_Score,' ...
    'Null_Threshold,Detected,Predicted_Elements,Dice_Elements,' ...
    'IoU_Elements,Dice_Area,IoU_Area,Centroid_Error_Mesh_Units,' ...
    'False_Positive_Elements,False_Negative_Elements\n']);

for k = 1:result_row
    fprintf(fid, ...
        '%d,%d,%.6f,%.6f,%.4f,%d,%g,%d,%.12g,%.12g,%.12g,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d\n', ...
        results(k,:));
end
fclose(fid);

% ------------------------------------------------------------
% 5. AGGREGATE BY NOISE LEVEL
% ------------------------------------------------------------

fid = fopen(summary_file, 'w');
if fid < 0
    error('Cannot create summary file.');
end

fprintf(fid, ['Noise_Percent,Trials,Detections,Detection_Rate_Percent,' ...
    'Mean_Dice_Area,Mean_IoU_Area,Mean_Centroid_Error_Detected,' ...
    'Mean_False_Positive_Elements,Mean_False_Negative_Elements\n']);

for ni = 1:numel(noise_levels)

    pct = noise_levels(ni);
    idx = find(results(1:result_row,7) == pct);
    block = results(idx,:);

    detections = sum(block(:,12));
    detection_rate = 100 * detections / rows(block);

    detected_dice = mean(block(:,16));
    detected_iou = mean(block(:,17));

    ce = block(block(:,12) == 1, 18);
    ce = ce(isfinite(ce));

    mean_centroid = NaN;
    if ~isempty(ce)
        mean_centroid = mean(ce);
    end

    mean_fp = mean(block(:,19));
    mean_fn = mean(block(:,20));

    fprintf(fid, '%g,%d,%d,%.4f,%.6f,%.6f,%.6f,%.4f,%.4f\n', ...
        pct, rows(block), detections, detection_rate, ...
        detected_dice, detected_iou, mean_centroid, mean_fp, mean_fn);
end

fclose(fid);

% ------------------------------------------------------------
% 6. DOCUMENT THE FIXED PROTOCOL
% ------------------------------------------------------------

fid = fopen(metadata_file, 'w');
if fid < 0
    error('Cannot create metadata file.');
end

fprintf(fid, 'NEUROSENTINEL UNKNOWN-SIZE EIT LOCALIZATION STUDY\n');
fprintf(fid, '================================================\n\n');
fprintf(fid, 'Data type: Synthetic EIDORS simulation only\n');
fprintf(fid, 'Source file: %s\n', source_file);
fprintf(fid, 'EIDORS version: 3.12\n');
fprintf(fid, 'GNU Octave version: 11.3.0\n');
fprintf(fid, 'Mesh nodes: %d\n', rows(N));
fprintf(fid, 'Mesh elements: %d\n', n_elements);
fprintf(fid, 'Electrodes: %d\n', numel(S.fmdl.electrode));
fprintf(fid, 'Measurements: %d\n', numel(S.vh.meas));
fprintf(fid, 'Geometry positions: %d\n', rows(centers));
fprintf(fid, 'Nominal radii: 0.15 and 0.22 mesh units\n');
fprintf(fid, 'Conductivity: background 1.0, anomaly 0.5\n');
fprintf(fid, 'Noise levels: 0, 1, 5, 10 percent\n');
fprintf(fid, 'Nonzero noise repeats per geometry: 10\n');
fprintf(fid, 'Noise model: independent additive Gaussian noise on baseline and target measurements\n');
fprintf(fid, 'Noise SD: percentage of saved anomaly measurement standard deviation\n');
fprintf(fid, 'Random seed: 20261109\n\n');

fprintf(fid, 'INVERSE MODEL\n');
fprintf(fid, 'Solver: inv_solve_diff_GN_one_step\n');
fprintf(fid, 'Prior: prior_laplace\n');
fprintf(fid, 'Hyperparameter: 0.001\n');
fprintf(fid, 'Reconstruction type: difference\n\n');

fprintf(fid, 'DETECTION RULE\n');
fprintf(fid, '1. Estimate a peak-score detection gate from separate no-anomaly calibration controls.\n');
fprintf(fid, '2. Gate equals the maximum null peak score at each noise level.\n');
fprintf(fid, '3. A detection requires peak score above the calibrated gate.\n');
fprintf(fid, '4. Detected region uses 50 percent of that reconstruction peak.\n');
fprintf(fid, '5. The detector never receives the true mask, radius, or true element count.\n');
fprintf(fid, '6. True masks are used only after detection for evaluation.\n\n');

fprintf(fid, 'CONTROL DESIGN\n');
fprintf(fid, 'Calibration: one clean control at 0 percent and ten controls at each nonzero noise level.\n');
fprintf(fid, 'Held-out false-alarm controls: one clean control at 0 percent and ten at each nonzero noise level.\n');
fprintf(fid, 'Anomaly trials: one clean trial and ten repeated trials at each nonzero noise level for every geometry.\n\n');

fprintf(fid, 'METRICS\n');
fprintf(fid, 'Element-count Dice and IoU; area-weighted Dice and IoU;\n');
fprintf(fid, 'area-weighted centroid error in mesh units; false positives; false negatives;\n');
fprintf(fid, 'detection rate and held-out no-anomaly false-alarm rate.\n\n');

fprintf(fid, 'LIMITATIONS\n');
fprintf(fid, 'This is a numerical simulation study, not clinical EIT.\n');
fprintf(fid, 'The mesh coordinates are normalized model units, not centimetres.\n');
fprintf(fid, 'The null calibration sample is limited and results are exploratory.\n');
fprintf(fid, 'A fixed half-maximum segmentation rule may fail for diffuse or multiple regions.\n');
fprintf(fid, 'No patient data or tumour/recurrence labels are used.\n');

fclose(fid);

% ------------------------------------------------------------
% 7. FINAL EXECUTION SUMMARY
% ------------------------------------------------------------

fprintf('\nUNKNOWN-SIZE EIT LOCALIZATION STUDY COMPLETED\n');
fprintf('Geometry cases: %d\n', geometry_id);
fprintf('Anomaly trials: %d\n', result_row);
fprintf('Total detections: %d\n', sum(results(1:result_row,12)));
fprintf('Held-out control false alarms: %d/%d\n', ...
    sum(control_rows(1:control_row,6)), control_row);
fprintf('Output directory: %s\n', output_dir);
fprintf('Results: %s\n', result_file);
fprintf('Summary: %s\n', summary_file);
fprintf('Element masks: %s\n', element_file);
fprintf('Metadata: %s\n', metadata_file);
