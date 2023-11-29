% extract betas values an store in lighter files
names={'/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj001.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj006.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj007.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj045.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj046.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj062.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj064.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj081.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj095.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj114.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj121.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/sub-wlsubj115.mat'
 };
nsub=numel(names);
for isub=1:nsub
    disp(isub)
    load(names{isub});
    betas=squeeze(results.modelmd{2});
    clear results
     [pth,nam,et,n]=spm_fileparts(names{isub});
    save([ '/Volumes/Elements/ds003812-download/derivatives/processed/betas',nam(5:13),'.mat'],'betas');
end
%% Summaraize SF beta files
cleanup
names={'/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj001.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj006.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj007.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj045.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj046.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj062.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj064.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj081.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj095.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj114.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj121.mat'
    '/Volumes/Elements/ds003812-download/derivatives/processed/betaswlsubj115.mat'};
nsub=numel(names);
for isub=1:nsub
    disp(isub)
    load(names{isub});
    temp=zeros(4,size(betas,1));
    for ifreq=1:4
       [~,temp(ifreq,:)]=max(betas(:,1+(ifreq-1)*10:(ifreq-1)*10+10),[],2);
    end
    pSF{isub}=temp;
    clear betas
end
save('/Volumes/Elements/ds003812-download/derivatives/processed/allpSF.mat',"pSF")
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

save('/Volumes/Elements/ds003812-download/derivatives/processed/subjmap2.mat',"subjmap","subjecc")

%% 
cleanup
wa=[6, 8, 11, 16, 23, 32, 45, 64, 91, 128];
wr=[4, 6, 8, 11, 16, 23, 32, 45, 64, 91];
load('/Volumes/Elements/ds003812-download/derivatives/processed/allpSF.mat')
load('/Volumes/Elements/ds003812-download/derivatives/processed/subjmap2.mat')
ecc=[];
psf =[];
roi=[];
side=[];
subj=[];
psfcorr=[];
stype=[];
for isub=1:12
    for ifreq=1:4
        disp(isub)
        idx1=1:numel(subjmap{isub,1});
        idx2=1:numel(subjmap{isub,2});
        psfL=pSF{isub}(ifreq,idx1);
        psfR=pSF{isub}(ifreq,idx2);

        idxgood=find(subjmap{isub,1});
        roi=[roi;subjmap{isub,1}(idxgood)];
        ecc=[ecc;subjecc{isub,1}(idxgood)];
        psf=[psf;psfL(idxgood)'];
        if ifreq==1 | ifreq==2
        psfcorr=[psfcorr;  wa(psfL(idxgood))'./subjecc{isub,1}(idxgood)];
            ww=sqrt(wa.^2+wr.^2);
            psfcorr=[psfcorr;  ww(psfL(idxgood))'./subjecc{isub,1}(idxgood)];            
        end
        side=[side; zeros(numel(idxgood),1)];
        subj=[subj; isub*ones(numel(idxgood),1)];
        stype=[stype; ifreq*ones(numel(idxgood),1)];

        idxgood2=find(subjmap{isub,2});
        roi=[roi;subjmap{isub,2}(idxgood2)];
        ecc=[ecc;subjecc{isub,2}(idxgood2)];
        psf=[psf;psfR(idxgood2)'];
        psfcorr=[psfcorr; wa(psfR(idxgood2))'./subjecc{isub,2}(idxgood2)];
        side=[side; ones(numel(idxgood2),1)];
        subj=[subj; isub*ones(numel(idxgood2),1)];
        stype=[stype; ifreq*ones(numel(idxgood2),1)];
    end
end
tab=table(double(ecc),roi,psf,side,subj,double(psfcorr),stype,'VariableNames',{'ecc','roi','psf','side','subj','psfcorr','stype'});
save('/Volumes/Elements/ds003812-download/derivatives/processed/tab2.mat','tab');
%% 
cleanup
load('/Volumes/Elements/ds003812-download/derivatives/processed/tab.mat','tab');

bad=isnan(tab.psfcorr) | isnan(tab.ecc) | isnan(tab.side) | isinf(tab.psfcorr) | isinf(tab.ecc) | isinf(tab.side);
tab(bad,:)=[];
clc
for iroi=5%:12
    tab1=tab(tab.roi==iroi,:);
     temp= fitlme(tab1,'psfcorr~ side*ecc + (side*ecc | subj)');
     mdlanova{iroi}=anova(temp)
     mdlfix{iroi}=fixedEffects(temp);
end