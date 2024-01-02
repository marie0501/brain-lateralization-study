clear
cols_tuning = [1,3,5,13,18];
tab=[];

for roi=1:12
    disp(roi)
    tuning=dir(fullfile(['F:\ds003812-download\results\tuning\tuning_voxels\',num2str(roi),filesep], '*.csv'));
    df = dir(fullfile(['F:\ds003812-download\results\df\',num2str(roi),filesep], '*.csv'));
    left_tuning =tuning(contains({tuning.name}, 'left'));
    left_df = df(contains({tuning.name}, 'left'));

    for isub=1: size(left_tuning,1)
       
        % tuning table
        xtab = readtable(deblank([left_tuning(isub,:).folder,filesep,left_tuning(isub,:).name]));
        [vx,idxvx]=sort(xtab.voxel,'ascend');
        xtab = xtab(idxvx,:);
        xtab=xtab(ismember(xtab.frequency_type,'local_sf_magnitude'),:);
        xtab=xtab(:,cols_tuning);
        xtab = addvars(xtab,zeros(size(xtab,1),1),'NewVariableNames','Side');
        xtab = addvars(xtab,isub*ones(size(xtab,1),1),'NewVariableNames','Subj');

        % df table
        ytab= readtable(deblank([left_df(isub,:).folder,filesep,left_df(isub,:).name]));
        ytab=ytab(:,[1,2,7,13]);
        ytab(max(ytab.voxel)+2:end,:)=[];
        xtab = join(xtab, ytab, 'Keys', 'voxel');
        tab=[tab;xtab];
        %tab=[tab;xtab];
    end
    clear xtab

    right_tuning=tuning(contains({tuning.name}, 'right'));
    right_df = df(contains({tuning.name}, 'right'));

    for isub=1: size(right_tuning,1)        
        xtab = readtable(deblank([right_tuning(isub,:).folder,filesep,right_tuning(isub,:).name]));
        xtab=xtab(ismember(xtab.frequency_type,'local_sf_magnitude'),:);
        xtab=xtab(:,cols_tuning);
        xtab = addvars(xtab,ones(size(xtab,1),1),'NewVariableNames','Side');
        xtab = addvars(xtab,isub*ones(size(xtab,1),1),'NewVariableNames','Subj');

        ytab= readtable(deblank([left_df(isub,:).folder,filesep,left_df(isub,:).name]));
        ytab=ytab(:,[1,2,7,13]);
        ytab(max(ytab.voxel)+2:end,:)=[];
        xtab = join(xtab, ytab, 'Keys', 'voxel');
        tab=[tab;xtab];
    end
clear xtab
tab(contains(tab.inf_warning,'True'),:)=[];
tab(tab.preferred_period>29,:)=[];
tab(tab.eccen>10,:)=[];
writetable(tab,[num2str(roi),'_all_tuning_voxels.csv'])
% save F:\ds003812-download\results\tuning\tuning_allV3preferedfrequenciesTab tab
end
%% 
clc
mdl0=fitlme(tab,'preferred_period~eccen*Side+(1|Subj)');
%mdl=fitlme(tab,'preferred_period~eccen*Side+(eccen*Side|Subj)')
% compare(mdl0,mdl)
%%
roi = 12;
tab = readtable(['C:\Users\Marie\Desktop\test\',num2str(roi),'_all_tuning_vbins_grouped.csv']);
bins=[0,1,2,3,4,5,6,7,8,9,10];
bincenter=(bins(1:end-1) + bins(2:end))/2;
pf=zeros(max(tab.Subj),numel(bincenter));
for isub=1:max(tab.Subj)
    tab1=tab(tab.Subj==isub & tab.Side==0,:);
    idx = discretize(tab1.eccen,bins);
    bad=isnan(idx);
    pff=tab1.preferred_period;
    idx(bad)=[];
    pff(bad)=[];
    pf(isub,:)=accumarray(idx,pff,[numel(bincenter),1],@median)';
end
pfR=zeros(max(tab.Subj),numel(bincenter));
for isub=1:max(tab.Subj)
    tab1=tab(tab.Subj==isub & tab.Side==1,:);
    idx = discretize(tab1.eccen,bins);
    bad=isnan(idx);
    pff=tab1.preferred_period;
    idx(bad)=[];
    pff(bad)=[];
    pfR(isub,:)=accumarray(idx,pff,[numel(bincenter),1],@median)';
end
figure; errorbar(mean(pf),std(pf)/sqrt(12),'LineWidth',2, 'Color', [0.6274509803921569, 0.3215686274509804, 0.17647058823529413]) 
hold on
errorbar(mean(pfR),std(pfR)/sqrt(12),'LineWidth',2, 'Color',[0.9568627450980393, 0.6431372549019608, 0.3764705882352941])
legend({'hemisferio izquierdo','hemisferio derecho'})
%grid on
%%  make binned table

prefered_freq=[]; 
eccBin=[];
side=[];
subj=[];
for isub=1: 12
     prefered_freq=[prefered_freq; pf(isub,:)'];
     side=[side;repmat(0,numel(bincenter),1)];
     subj=[subj;repmat(isub,numel(bincenter),1)];
     eccBin=[eccBin;bincenter'];
     prefered_freq=[prefered_freq; pfR(isub,:)'];
     side=[side;repmat(1,numel(bincenter),1)];
     eccBin=[eccBin;bincenter'];
     subj=[subj;repmat(isub,numel(bincenter),1)];
end
othertab=table(prefered_freq,eccBin,side, subj,'VariableNames',{'PF','EccBin','Side','Subj'});
writetable(othertab,'binnedPF.csv')
othertab.EccBin=categorical(othertab.EccBin);

mdl=fitlme(othertab,'PF~1+Side/EccBin')

for i=1: numel(unique(othertab.EccBin))
     x=othertab.PF(othertab.EccBin==bincenter(i) & othertab.Side==0);
     y=othertab.PF(othertab.EccBin==bincenter(i) & othertab.Side==1);
     pp(i)=signrank(x-y);
end

for i=1: numel(unique(othertab.EccBin))
     x=othertab.PF(othertab.EccBin==bincenter(i) & othertab.Side==0);
     y=othertab.PF(othertab.EccBin==bincenter(i) & othertab.Side==1);
     [~,pp(i)]=ttest(x-y);
end
