cleanup
maxroi=12;
tdir='/Volumes/Elements/results/tuning/tuning_voxels/';
dfdir='/Volumes/Elements/results/df/';
resdir='/Users/pedro/work/frecuencyBroderick/';

side={'left','right'};
for iroi=4%1:maxroi
    tab=[];
    for iside=1: 2
        dirt=[tdir,num2str(iroi),filesep];
        dirdf=[dfdir,num2str(iroi),filesep];
        t=spm_select('FPList',dirt,['^tuning_voxels_df_',num2str(iroi),'.*.',side{iside},'*.csv$']);
        df=spm_select('FPList',dirdf,['^df_',num2str(iroi),'.*.',side{iside},'*.csv$']);
        for isub=1: size(t,1)
            fprintf('%d \t %s \t %d \n', iroi, side{iside}, isub)
            % read data
            xtab = readtable(deblank(t(isub,:)));
            xtab=xtab(ismember(xtab.frequency_type,'local_sf_magnitude'),:);
            xtab=xtab(:,[1,2,3,5,13,14,18]);
            [vx,idxvx]=sort(xtab.voxel,'ascend');
            xtab.preferred_period(contains(xtab.inf_warning,'True'))=NaN;
            xtab.tuning_curve_bandwidth(contains(xtab.inf_warning,'True'))=NaN;
            PF=accumarray(xtab.voxel+1, xtab.preferred_period,'',@nanmedian);
            BW=accumarray(xtab.voxel+1, xtab.tuning_curve_bandwidth,'',@median);
            ytab= readtable(deblank(df(isub,:)));
            ytab=ytab(:,[2,6,7,13]);
            ytab(max(ytab.voxel)+2:end,:)=[];
            ytab=addvars(ytab,PF,BW,'NewVariableNames',{'PF','BW'});
            ytab = addvars(ytab,(iside-1)*ones(size(ytab,1),1),'NewVariableNames','Side');
            ytab = addvars(ytab,isub*ones(size(ytab,1),1),'NewVariableNames','Subj');
            tab=[tab;ytab];
            clear ytab xtab
        end % for isub
    end % for iside

    % cleaning
%     tab(tab.PF>5,:)=[];
%     tab(tab.PF<0.5,:)=[];
    tab(tab.eccen>8,:)=[];
%     tab(tab.tuning_curve_bandwidth>23,:)=[];
   
    % save
    writetable(tab,[resdir,num2str(iroi),'.csv'])
    save([resdir,num2str(iroi),'.mat'], 'tab');
end % for iroi
%%

clc
mdl0=fitlme(tab,'preferred_period~eccen*Side+(1|Subj)')
mdl=fitlme(tab,'preferred_period~eccen*Side+(eccen*Side|Subj)')
compare(mdl0,mdl)
%%
bins=[1,3,5,7,9,11];
bincenter=(bins(1:end-1) + bins(2:end))/2;
pf=zeros(max(tab.Subj),numel(bincenter));
for isub=1:max(tab.Subj)
    tab1=tab(tab.Subj==isub & tab.Side==0,:);
    idx = discretize(tab1.eccen,bins);
    bad=isnan(idx);
    pff=tab1.PF;
    idx(bad)=[];
    pff(bad)=[];
    pf(isub,:)=accumarray(idx,pff,[size(bincenter,2),size(bincenter,1)],@median)';
end
pfR=zeros(max(tab.Subj),numel(bincenter));
for isub=1:max(tab.Subj)
    tab1=tab(tab.Subj==isub & tab.Side==1,:);
    idx = discretize(tab1.eccen,bins);
    bad=isnan(idx);
    pff=tab1.PF;
    idx(bad)=[];
    pff(bad)=[];
     pfR(isub,:)=accumarray(idx,pff,[size(bincenter,2),size(bincenter,1)],@median)';
end
figure; errorbar(mean(pf),std(pf)/sqrt(12),'LineWidth',2)
hold on
errorbar(mean(pfR),std(pfR)/sqrt(12),'LineWidth',2)
legend({'Left','Right'})
title('Preferred period')

pbw=zeros(max(tab.Subj),numel(bincenter));
for isub=1:max(tab.Subj)
    tab1=tab(tab.Subj==isub & tab.Side==0,:);
    idx = discretize(tab1.eccen,bins);
    bad=isnan(idx);
    pw=tab1.BW;
    pw(bad)=[];
    pw(bad)=[];
    pbw(isub,:)=accumarray(idx,pw,[size(bincenter,2),size(bincenter,1)],@median)';
end
pbwR=zeros(max(tab.Subj),numel(bincenter));
for isub=1:max(tab.Subj)
    tab1=tab(tab.Subj==isub & tab.Side==1,:);
    idx = discretize(tab1.eccen,bins);
    bad=isnan(idx);
    pw=tab1.BW;
    idx(bad)=[];
    pw(bad)=[];
    pbwR(isub,:)=accumarray(idx,pw,[size(bincenter,2),size(bincenter,1)],@median)';
end
figure; errorbar(mean(pbw),std(pbw)/sqrt(12),'LineWidth',2)
hold on
errorbar(mean(pbwR),std(pbwR)/sqrt(12),'LineWidth',2)
legend({'Left','Right'})
title('Bandwidth')
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
%% 


glme = fitglme(tab,'preferred_period ~ 1+ eccen*Side + (1|Subj)', ...
    'Distribution','Gamma','Link','log','FitMethod','Laplace', ...
    'DummyVarCoding','effects');          
%% 
load 
figure;
tiledlayout(1,2)
nexttile
densityScatterChart(tab.eccen(tab.Side==1),log(tab.preferred_period(tab.Side==1)))
xlim([1,3])
ylim([-1.5, 0.5])
title('left')

nexttile
densityScatterChart(tab.eccen(tab.Side==0),log(tab.preferred_period(tab.Side==0)))
xlim([1,3])
ylim([-1.5, 0.5])
title('right')

 
figure;
tiledlayout(1,2)
nexttile
densityScatterChart(tab.eccen(tab.Side==1),(tab.preferred_period(tab.Side==1)))

title('left')

nexttile
densityScatterChart(tab.eccen(tab.Side==0),(tab.preferred_period(tab.Side==0)))

title('right')

