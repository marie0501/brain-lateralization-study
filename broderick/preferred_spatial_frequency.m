function psf_index = preferred_spatial_frequency(directory)
    psf_index = cell(1, numel(dir(directory)) - 2);  % Preallocate cell array

    files = dir(fullfile(directory, '*.npy'));

    for file_idx = 1:numel(files)
        file = files(file_idx).name;
        beta = load(fullfile(directory, file));
        index_sub = nan(4, size(beta, 2));

        disp(file);

        for freq = 1:4
            for voxel = 1:size(beta, 2)
                temp = beta((freq - 1) * 10 + 1 : freq * 10, voxel);
                [~, index] = max(temp);
                index_sub(freq, voxel) = beta(index, voxel);
            end
        end

        psf_index{file_idx} = index_sub;
    end
end

function main()
    psf = preferred_spatial_frequency('F:\ds003812-download\derivatives\processed\betas');
    ecc = load_all_prf_data('F:\ds003812-download\derivatives\prf_solutions\all', 'eccen');
    roi = load_all_prf_data('F:\ds003812-download\derivatives\prf_solutions\all', 'benson14_varea');
    
    disp(psf{1}); % Assuming psf is a cell array in MATLAB
    
    psf_result = [];
    ecc_result = [];
    roi_result = [];
    side_result = [];
    subj_result = [];
    freq_result = [];

    for sub = 1:length(psf)
        psf_l = psf{sub}(:, 1:length(ecc{sub}{1}));
        psf_r = psf{sub}(:, length(ecc{sub}{1})+1:end);

        for freq = 1:4
            % Left
            psf_result = [psf_result; psf_l(freq, :)'];
            roi_result = [roi_result; roi{sub}{1}];
            ecc_result = [ecc_result; ecc{sub}{1}];
            freq_result = [freq_result; ones(size(ecc{sub}{1}))*freq];
            side_result = [side_result; zeros(size(ecc{sub}{1}))];
            subj_result = [subj_result; ones(size(ecc{sub}{1}))*sub];

            % Right
            psf_result = [psf_result; psf_r(freq, :)'];
            roi_result = [roi_result; roi{sub}{2}];
            ecc_result = [ecc_result; ecc{sub}{2}];
            freq_result = [freq_result; ones(size(ecc{sub}{2}))*freq];
            side_result = [side_result; ones(size(ecc{sub}{2}))];
            subj_result = [subj_result; ones(size(ecc{sub}{2}))*sub];
        end
    end

    data = table(psf_result, roi_result, ecc_result, freq_result, side_result, subj_result);
    data = rmmissing(data);
    writetable(data, 'all.csv');
end

function df = analysis()
    df = main();
    
    for roi = 1:12
        df_roi = df((df.roi == roi) & (df.freq == 4), :);
        modelo_mixto = fitlme(df_roi, 'psf ~ side * ecc + (1|subj)');
        disp(roi);
        disp(modelo_mixto);
    end
end
