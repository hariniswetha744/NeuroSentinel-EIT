
% NeuroSentinel - Reproducible synthetic EIT scenarios
% Uses EIDORS forward simulations.
% Synthetic research data only. Not clinical or recurrence data.

clear;
clc;

root_dir = 'C:/NeuroSentinel';
eidors_dir = [root_dir '/EIDORS/eidors-v3.12-ng/eidors-v3.12-ng/eidors'];
source_file = [root_dir '/eit/results/EIT_Simulation_Data.mat'];
output_dir = [root_dir '/eit/synthetic_experiment/scenarios'];

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

cd(eidors_dir);
eidors_startup;

S = load(source_file);

% Check source model dimensions
assert(size(S.fmdl.elems,1) == 576);
assert(numel(S.img_h.elem_data) == 576);
assert(numel(S.img_i.elem_data) == 576);
assert(numel(S.vh.meas) == 208);

% Reproducible random noise
randn('seed', 20261108);

% Severity levels interpolate between saved baseline and anomaly
severity_levels = [0, 0.25, 0.50, 0.75, 1.00];
noise_levels = [0, 1, 5, 10];

measurement_file = [output_dir '/EIT_Synthetic_Scenario_Measurements.csv'];
summary_file = [output_dir '/EIT_Synthetic_Scenario_Summary.csv'];

fm = fopen(measurement_file, 'w');
fs = fopen(summary_file, 'w');

assert(fm >= 0 && fs >= 0);

fprintf(fm, ['Scenario_ID,Severity,Noise_Percent,Repeat_ID,' ...
    'Measurement_ID,Voltage,Noise_SD,Label\n']);

fprintf(fs, ['Scenario_ID,Severity,Noise_Percent,Repeat_ID,' ...
    'Label,Measurement_Count,Noise_SD\n']);

scenario_count = 0;

for si = 1:numel(severity_levels)

    severity = severity_levels(si);

    % Construct conductivity image by interpolation
    img = S.img_h;
    img.fwd_model = S.fmdl;
    img.elem_data = S.img_h.elem_data(:) + severity .* ...
        (S.img_i.elem_data(:) - S.img_h.elem_data(:));

    % Generate physically modelled clean measurements
    clean_result = fwd_solve(img);
    clean_voltage = clean_result.meas(:);

    assert(numel(clean_voltage) == 208);
    assert(all(isfinite(clean_voltage)));

    signal_scale = std(clean_voltage);

    for ni = 1:numel(noise_levels)

        noise_pct = noise_levels(ni);

        if noise_pct == 0
            repeats = 1;
        else
            repeats = 10;
        end

        noise_sd = (noise_pct / 100) * signal_scale;

        for ri = 1:repeats

            scenario_count = scenario_count + 1;
            scenario_id = sprintf('EIT_S%03d', scenario_count);

            noisy_voltage = clean_voltage + ...
                noise_sd .* randn(size(clean_voltage));

            label = double(severity > 0);

            fprintf(fs, '%s,%.2f,%d,%d,%d,%d,%.12g\n', ...
                scenario_id, severity, noise_pct, ri, ...
                label, numel(noisy_voltage), noise_sd);

            for mi = 1:numel(noisy_voltage)
                fprintf(fm, '%s,%.2f,%d,%d,%d,%.12g,%.12g,%d\n', ...
                    scenario_id, severity, noise_pct, ri, ...
                    mi, noisy_voltage(mi), noise_sd, label);
            end
        end
    end
end

fclose(fm);
fclose(fs);

fid = fopen([output_dir '/README.txt'], 'w');
fprintf(fid, 'NeuroSentinel synthetic EIT scenario dataset\n');
fprintf(fid, 'Forward model: saved EIDORS 576-element mesh\n');
fprintf(fid, 'Severity levels: 0, 0.25, 0.50, 0.75, 1.00\n');
fprintf(fid, 'Noise levels: 0, 1, 5, 10 percent of signal SD\n');
fprintf(fid, 'Random seed: 20261108\n');
fprintf(fid, 'Scenario count: %d\n', scenario_count);
fprintf(fid, 'Repeated noise trials are not independent subjects.\n');
fprintf(fid, 'One underlying anomaly pattern is used.\n');
fprintf(fid, 'Synthetic research data only; no clinical labels.\n');
fclose(fid);

fprintf('Scenario generation completed: %d scenarios\n', scenario_count);
fprintf('Output: %s\n', output_dir);
