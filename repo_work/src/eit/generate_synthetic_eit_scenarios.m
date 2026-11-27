% NeuroSentinel synthetic EIT reference export
% Octave-compatible. Does not modify original simulation files.

clear;
clc;

root_dir = 'C:/NeuroSentinel';
source_file = [root_dir '/eit/results/EIT_Simulation_Data.mat'];
output_dir = [root_dir '/eit/synthetic_experiment'];

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

S = load(source_file);

assert(size(S.fmdl.elems,1) == 576);
assert(numel(S.vh.meas) == 208);
assert(numel(S.vi.meas) == 208);

baseline = S.vh.meas(:);
anomaly = S.vi.meas(:);

fid = fopen([output_dir '/EIT_Reference_Measurements.csv'], 'w');
fprintf(fid, 'Measurement_ID,Baseline,Anomaly,Difference\n');

for k = 1:numel(baseline)
    fprintf(fid, '%d,%.12f,%.12f,%.12f\n', ...
        k, baseline(k), anomaly(k), anomaly(k)-baseline(k));
end
fclose(fid);

fid = fopen([output_dir '/EIT_Reference_Elements.csv'], 'w');
fprintf(fid, 'Element_ID,Baseline,Anomaly,Reconstruction,Region\n');

for k = 1:numel(S.img_h.elem_data)
    fprintf(fid, '%d,%.12f,%.12f,%.12f,%d\n', ...
        k, S.img_h.elem_data(k), S.img_i.elem_data(k), ...
        S.img_rec.elem_data(k), S.region(k));
end
fclose(fid);

fid = fopen([output_dir '/README.txt'], 'w');
fprintf(fid, 'NeuroSentinel EIT reference export\n');
fprintf(fid, 'Source: existing EIT_Simulation_Data.mat\n');
fprintf(fid, 'Mesh nodes: %d\n', size(S.fmdl.nodes,1));
fprintf(fid, 'Mesh elements: %d\n', size(S.fmdl.elems,1));
fprintf(fid, 'Electrodes: %d\n', numel(S.fmdl.electrode));
fprintf(fid, 'Measurements: %d\n', numel(baseline));
fprintf(fid, 'Anomaly elements: %d\n', sum(S.region(:)));
fprintf(fid, 'Reference exports only; not independent scenarios.\n');
fprintf(fid, 'No patient data or recurrence labels.\n');
fclose(fid);

disp('Reference export completed.');