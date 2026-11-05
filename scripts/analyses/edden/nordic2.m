% NIFTI_NORDIC2 (submodules/NORDIC_Raw) with its own defaults on the four EDDEN 0.9 mm AP runs,
% magnitude and phase in, magnitude and phase out. The random number generator is reset before
% each run, because NIFTI_NORDIC2 draws unseeded random numbers.
%
% Out: data/analyses/edden/nordic2/run-0N_nordic2_{magn,phase}.nii.gz. Finished runs are
% skipped.
%
%   DATA_PATH=... matlab -batch "run('scripts/analyses/edden/nordic2.m')"

repo = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
data = getenv('DATA_PATH');
src = fullfile(data, 'source/openneuro/ds004666/sub-01/ses-0p9mm/dwi');
out = fullfile(data, 'analyses/edden/nordic2');
addpath(fullfile(repo, 'submodules/NORDIC_Raw'));
if ~exist(out, 'dir'); mkdir(out); end
parpool('Threads', 4);

for run = {'01', '02', '03', '04'}
    name = ['run-' run{1} '_nordic2_'];
    if exist(fullfile(out, [name 'phase.nii.gz']), 'file'); continue; end
    stem = fullfile(src, ['sub-01_ses-0p9mm_dir-AP_run-' run{1}]);
    ARG = struct('DIROUT', [out '/'], 'write_gzipped_niftis', 1, 'make_complex_nii', 1);
    rng('default');
    NIFTI_NORDIC2([stem '_part-mag_dwi.nii.gz'], [stem '_part-phase_dwi.nii.gz'], name, ARG);
end
