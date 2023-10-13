% Reverse-correlation CF analysis
% data
clear
clc 
close all
% datadir='/Volumes/Seagate Expansion Drive/7Tsuperficie/SinFiltro/';
% datadir='/Volumes/Untitled 1/7Tsuperficie/SinFiltro/';
datadir='/Volumes/Untitled 1/7Tsuperficie/Procesados-ETI/para_antonieta/';
templateDir = '/Users/pedro/work/maastrichtCRF/code'; 
% data
subdirs = spm_select('List',datadir,'dir','^sub.');
ncond=5; 
nsub = size(subdirs,1);
Srf.Meshes= [templateDir,filesep 'bi_fs_LR.mat'] ;
for isub=2:4%nsub
    disp(isub);
    hf = pwd;
    output_dir= [datadir,deblank(subdirs(isub,:)),filesep];
    cd(output_dir);
    
%     load([datadir,subdirs(isub,:),filesep,'RESultra.mat'], 'res');
%     load([datadir,deblank(subdirs(isub,:)),filesep,'RESultraSinFNoQuitarConf.mat'], 'res');

    %% CF reverse correlation analysis
    Model.SeedRoi = '/Users/pedro/work/maastrichtCRF/code/V1new'; % ROI label for seed region - POINT AT BILATERAL V1 LABEL IN ATLAS MAP
    Model.Template = [templateDir,filesep 'bi_new.mat']; % Atlas retinotopic map of seed region - POINT AT TEMPLATE MAP FILE IN FSAVERGE
    Model.Global_Signal_Correction = false; % Don't correct by global mean signal - AS YOU ALREADY DID PCA ON THE DATA?
    Model.Save_Rmaps = true;
    %% Analyse CF for all data
   CF_filename = 'SrfMinimalV1cleanWhole.mat';
    Model.Name = 'Srf','_SrfMinimalV1cleanWholeFinal_';
    samsrf_revcor_cf(Model, CF_filename, '/Users/pedro/work/maastrichtCRF/code/facehouse'); % POINT AT OCCIPITAL ROI IN TEMPLATE FOLDER

cd(hf); % Return home
end