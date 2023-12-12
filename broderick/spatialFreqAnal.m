% extract betas values an store in lighter files
names={'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj001.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj006.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj007.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj045.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj046.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj062.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj064.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj081.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj095.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj114.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj121.mat'
    'F:\\ds003812-download\\derivatives\\processed\\sub-wlsubj115.mat'
 };
nsub=numel(names);
for isub=1:nsub
    disp(isub)
    load(names{isub});
    betas=squeeze(results.modelmd{2});
    clear results
     [pth,nam,et,n]=spm_fileparts(names{isub});
    save([ 'F:\\ds003812-download\\derivatives\\processed\\betas',nam(5:13),'.mat'],'betas');
end
%% Summaraize SF beta files
clear
names={'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj001.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj006.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj007.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj045.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj046.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj062.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj064.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj081.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj095.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj114.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj121.mat'
    'F:\\ds003812-download\\derivatives\\processed\\betaswlsubj115.mat'};
nsub=numel(names);
for isub=1:nsub
    disp(isub)
    load(names{isub});
    temp=zeros(4,size(betas,1));
    for ifreq=1:4
       [temp(ifreq,:),~]=max(betas(:,1+(ifreq-1)*10:(ifreq-1)*10+10),[],2);
    end
    pSF{isub}=temp;
    
end
save('F:\\ds003812-download\\derivatives\\processed\\allpSF.mat',"pSF")
%% %  extract retinopoty data
dirmaps='/Volumes/Elements/ds003812-download/derivatives/prf_solutions/';
names={'sub-wlsubj001', 'sub-wlsubj045','sub-wlsubj064',	'sub-wlsubj114',...
'sub-wlsubj006','sub-wlsubj046',	'sub-wlsubj081','sub-wlsubj115',...
'sub-wlsubj007','sub-wlsubj062','sub-wlsubj095','sub-wlsubj121'};
nsub=numel(names);
for isub=1:nsub
    disp(isub)
%     subj=[dirmaps,names{isub},'/bayesian_posterior/'];
%     a=gifti([subj,'lh.inferred_varea.func.gii']);
%     b=gifti([subj,'rh.inferred_varea.func.gii']);
    subj=[dirmaps,names{isub},'/atlas/'];
    a=gifti([subj,'lh.benson14_varea.func.gii']);
    b=gifti([subj,'rh.benson14_varea.func.gii']);

    subjmap{isub,1}=a.cdata;
    subjmap{isub,2}=b.cdata;
    subj=[dirmaps,names{isub},'/data/'];
    a=gifti([subj,'lh.full-eccen.func.gii']);
    b=gifti([subj,'rh.full-eccen.func.gii']);
    subjecc{isub,1}=a.cdata;
    subjecc{isub,2}=b.cdata;
end

save('F:\\ds003812-download\\derivatives\\processed\\subjmap2.mat',"subjmap","subjecc")

%% 
clear
load('F:\\ds003812-download\\derivatives\\processed\\allpSF.mat')
load('F:\\ds003812-download\\derivatives\\processed\\subjmap2.mat')
ecc=[];
psf =[];
roi=[];
side=[];
subj=[];
freq=[];

for isub=1:12
     disp(isub)
     idx1=numel(subjmap{isub,1});
     idx2=numel(subjmap{isub,2});
    for ifreq=1:4

        psfL=pSF{isub}(ifreq,1:idx1);
        psfR=pSF{isub}(ifreq,idx1+1:end);

        % Left
        psf = [psf; psfL'];
        roi = [roi; subjmap{isub,1}];
        ecc = [ecc; subjecc{isub,1}];
        freq = [freq; ones(idx1,1)*ifreq];
        side = [side; zeros(idx1,1)];
        subj = [subj; ones(idx1,1)*isub];

        % Right
        psf = [psf; psfR'];
        roi = [roi; subjmap{isub,2}];
        ecc = [ecc; subjecc{isub,2}];
        freq = [freq; ones(idx2,1)*ifreq];
        side = [side; ones(idx2,1)];
        subj = [subj; ones(idx2,1)*isub];

        % idxgood=find(subjmap{isub,1});
        % roi=[roi;subjmap{isub,1}(idxgood)];
        % ecc=[ecc;subjecc{isub,1}(idxgood)];
        % psf=[psf;psfL(idxgood)'];
        % if ifreq==1 | ifreq==2
        % psfcorr=[psfcorr;  wa(psfL(idxgood))'./subjecc{isub,1}(idxgood)];
        %     ww=sqrt(wa.^2+wr.^2);
        %     psfcorr=[psfcorr;  ww(psfL(idxgood))'./subjecc{isub,1}(idxgood)];            
        % end
        % side=[side; zeros(numel(idxgood),1)];
        % subj=[subj; isub*ones(numel(idxgood),1)];
        % stype=[stype; ifreq*ones(numel(idxgood),1)];
        % 
        % idxgood2=find(subjmap{isub,2});
        % roi=[roi;subjmap{isub,2}(idxgood2)];
        % ecc=[ecc;subjecc{isub,2}(idxgood2)];
        % psf=[psf;psfR(idxgood2)'];
        % psfcorr=[psfcorr; wa(psfR(idxgood2))'./subjecc{isub,2}(idxgood2)];
        % side=[side; ones(numel(idxgood2),1)];
        % subj=[subj; isub*ones(numel(idxgood2),1)];
        % stype=[stype; ifreq*ones(numel(idxgood2),1)];
    end
end
tab=table(double(ecc),roi,psf,side,subj,freq,'VariableNames',{'ecc','roi','psf','side','subj','freq'});
save('F:\\ds003812-download\\derivatives\\processed\\tab2.mat','tab');
%% 
clear
tab = readtable("C:\\Users\\Marie\\Documents\\thesis\\broderick\\table1.csv");
rois ={'V1','V2','V3','hV4','VO1','VO2','V3a','V3b','LO1','LO2','TO1','TO2'};
bad=isnan(tab.psf) | isnan(tab.ecc) | isnan(tab.side) | isinf(tab.psf) | isinf(tab.ecc) | isinf(tab.side);
tab(bad,:)=[];
clc


for iroi=1:12
    tab1=tab(tab.roi==iroi,:);
     temp= fitlme(tab1,'psf~ side*ecc + (1 | subject)');
     mdlanova{iroi}=anova(temp);
     mdlfix{iroi}=fixedEffects(temp);
     display(rois{iroi})
     display(mdlanova{iroi})
    
end
