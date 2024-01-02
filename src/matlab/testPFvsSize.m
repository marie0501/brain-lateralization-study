dfr=df(:,[2,6,7,13]);
dfr=dfr(1:1782,:);
tr=t(ismember(t.frequency_type,'local_sf_magnitude'),:);
trr=accumarray(tr.voxel+1, tr.preferred_period,'',@median);
dfr=addvars(dfr,trr,'NewVariableNames','PF');

good=(dfr.eccen>1.2) & (dfr.PF<2) & (dfr.sigma>1.2) & (dfr.sigma<2  );
figure; scatter(dfr.sigma(good),dfr.PF(good),60,'filled')
hold on
b = robustfit(dfr.sigma(good),dfr.PF(good));
x=1.2:0.1:2;
y=b(1)+b(2)*x;
plot(x,y)


figure; scatter(dfr.sigma(good),1/dfr.PF(good),60,'filled')

