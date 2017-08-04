function  [rslt] = cnnshowresult_new(rslt, opts, x, y, datameta, transmeta)
global useFileLog;
if useFileLog
    if isfield(opts,'logfilename')
        fid = fopen(opts.logfilename,'a+');
    else
        fid = fopen('result.txt', 'a+');
    end
else
    fid = 1;
end
fprintf(fid,'<result>\n');
if isfield(rslt,'epoch')
    fprintf(fid,'Epoch:%d\n',rslt.epoch);
end
m = size(y, 2);
if strcmp(rslt.type, 'C')
    cnum = size(y,1);
    [~, rslt.plabel] = max(rslt.pv);
    [~, rslt.tlabel] = max(y);
    rslt.accunum = numel(find(rslt.plabel == rslt.tlabel));
    rslt.accuracy = rslt.accunum / m;
    rslt.tnum = histcounts(rslt.tlabel,1:cnum+1);
    rslt.pnum = histcounts(rslt.plabel,1:cnum+1);
    rslt.hitnum = histcounts(rslt.plabel(rslt.plabel == rslt.tlabel),1:cnum+1);
    rslt.recall = rslt.hitnum ./ (rslt.tnum+eps);
    rslt.precision = rslt.hitnum ./ (rslt.pnum+eps);
    for i = 1 : cnum
        fprintf(fid,'CLASS %d -- recall:%.4f(%d/%d),  precision:%.4f(%d/%d)\n',...
            i,rslt.recall(i),rslt.hitnum(i),rslt.tnum(i),rslt.precision(i),rslt.hitnum(i),rslt.pnum(i));
    end
    fprintf(fid,'accuracy:%.4f(%d/%d)\n',rslt.accuracy,rslt.accunum,m);
    fprintf(fid,'loss:%.4f\n',rslt.loss);
    
elseif strcmp(rslt.type, 'R')
    
    
    channel_dim = size(x, 4);
    rslt.dev = rslt.pv - y;
    
    
    rslt.mcMAEmtx = squeeze(sum(reshape(abs(rslt.dev),[],channel_dim,m)));
    rslt.cMAE = sum(rslt.mcMAEmtx,2)./ sum(sum(reshape(y>=0,[],channel_dim,m)),3)';
    rslt.MAE = sum(rslt.cMAE) / channel_dim;
    
    
    ere=zeros(size(y));
    ere(y>0) = abs(rslt.dev(y>0)./ y(y>0));
    
    rslt.mcMREmtx =  squeeze(sum(reshape(ere,[],channel_dim,m)));
    rslt.cMRE = sum(rslt.mcMREmtx,2)./ sum(sum(reshape(y>0,[],channel_dim,m)),3)';
    rslt.MRE = sum(rslt.cMRE) / channel_dim;
    
    
    
    rslt.ese = rslt.dev.^2;
    rslt.mcMSEmtx = squeeze(sum(reshape(rslt.ese,[],channel_dim,m)));
    rslt.cMSE = sum(rslt.mcMSEmtx,2)./ sum(sum(reshape(y>=0,[],channel_dim,m)),3)';
    rslt.MSE = sum(rslt.cMSE) / channel_dim;
    
    rese=zeros(size(y));
    rese(y>0) = rslt.ese(y>0)./ y(y>0);
    
    rslt.mcRMSEmtx =  squeeze(sum(reshape(rese,[],channel_dim,m)));
    rslt.cRMSE = sum(rslt.mcRMSEmtx,2)./ sum(sum(reshape(y>0,[],channel_dim,m)),3)';
    rslt.RMSE = sum(rslt.cRMSE) / channel_dim;
    if nargin > 4
        [~,rslt.tpv] = transformshapedsample(x,rslt.pv,transmeta,datameta);
        [~,tyv] = transformshapedsample(x,y,transmeta,datameta);
        
        rslt.tdev = rslt.tpv - tyv;
        
        rslt.tese = rslt.tdev.^2;
        
        rslt.tere = zeros(size(tyv));
        rslt.tere(tyv>0) = abs(rslt.tdev(tyv>0) ./ tyv(tyv>0));
        
        rslt.tmcMAEmtx = squeeze(sum(reshape(abs(rslt.tdev),[],channel_dim,m)));
        rslt.tcMAE = sum(rslt.tmcMAEmtx, 2)./sum(sum(reshape(tyv>=0,[],channel_dim,m), 3))';
        rslt.tMAE = sum(rslt.tcMAE) / channel_dim;
        
        tere=zeros(size(tyv));
        tere(tyv>0) = abs(rslt.tdev(tyv>0)./ tyv(tyv>0));
        
        rslt.tmcMREmtx =  squeeze(sum(reshape(tere,[],channel_dim,m)));
        rslt.tcMRE = sum(rslt.tmcMREmtx,2)./ sum(sum(reshape(tyv>0,[],channel_dim,m), 3))';
        rslt.tMRE = sum(rslt.tcMRE) / channel_dim;
        
        
        
        rslt.tmcMSEmtx = squeeze(sum(reshape(rslt.tese,[],channel_dim,m)));
        rslt.tcMSE = sum(rslt.tmcMSEmtx,2)./sum(sum(reshape(tyv>=0,[],channel_dim,m), 3))';
        rslt.tMSE = sum(rslt.tcMSE) / channel_dim;
        
        trese=zeros(size(tyv));
        trese(tyv>0) = rslt.tese(tyv>0)./ tyv(tyv>0);
        
        rslt.tmcRMSEmtx =  squeeze(sum(reshape(trese,[],channel_dim,m)));
        rslt.tcRMSE = sum(rslt.tmcRMSEmtx,2)./ sum(sum(reshape(tyv>0,[],channel_dim,m), 3))';
        rslt.tRMSE = sum(rslt.tcRMSE) / channel_dim;
        
        rslt.tloss = opts.lossfunc(rslt.tpv, tyv);
    end
    
    if nargin <= 4
        fprintf(fid,'<data MAE>\n');
        fprintf(fid,'       Flow:%.4f\n      Speed:%.4f\n  Occupancy:%.4f\n  average MAE:%.4f\n',rslt.cMAE,sum(rslt.cMAE)/channel_dim);
        
        fprintf(fid,'<data MRE>\n');
        fprintf(fid,'       Flow:%.4f\n      Speed:%.4f\n  Occupancy:%.4f\n  average MRE:%.4f\n',rslt.cMRE,sum(rslt.cMRE)/channel_dim);
        
        fprintf(fid,'<data MSE>\n');
        fprintf(fid,'       Flow:%.4f\n      Speed:%.4f\n  Occupancy:%.4f\n  average MSE:%.4f\n',rslt.cMSE,sum(rslt.cMSE)/channel_dim);
        
        fprintf(fid,'<data RMSE>\n');
        fprintf(fid,'       Flow:%.4f\n      Speed:%.4f\n  Occupancy:%.4f\n  average RMSE:%.4f\n',rslt.cRMSE,sum(rslt.cRMSE)/channel_dim);
        
        fprintf(fid,'optm loss:%.4f',rslt.loss);
    else
        fprintf(fid,'<data MAE>\n');
        fprintf(fid,'       Flow normed:%.4f true:%.4f\n      Speed normed:%.4f true:%.4f\n  Occupancy normed:%.4f true:%.4f\n  average MAE  normed:%.4f true:%.4f\n',...
            [rslt.cMAE';rslt.tcMAE'],sum(rslt.cMAE)/channel_dim,sum(rslt.tcMAE)/channel_dim);
        
        fprintf(fid,'<data MRE>\n');
        fprintf(fid,'       Flow normed:%.4f true:%.4f\n      Speed normed:%.4f true:%.4f\n  Occupancy normed:%.4f true:%.4f\n  average MRE  normed:%.4f true:%.4f\n',...
            [rslt.cMRE';rslt.tcMRE'],sum(rslt.cMRE)/channel_dim,sum(rslt.tcMRE)/channel_dim);
        
        fprintf(fid,'<data MSE>\n');
        fprintf(fid,'       Flow normed:%.4f true:%.4f\n      Speed normed:%.4f true:%.4f\n  Occupancy normed:%.4f true:%.4f\n  average MSE  normed:%.4f true:%.4f\n',...
            [rslt.cMSE';rslt.tcMSE'],sum(rslt.cMSE)/channel_dim,sum(rslt.tcMSE)/channel_dim);
        
        fprintf(fid,'<data RMSE>\n');
        fprintf(fid,'       Flow normed:%.4f true:%.4f\n      Speed normed:%.4f true:%.4f\n  Occupancy normed:%.4f true:%.4f\n  average RMSE  normed:%.4f true:%.4f\n',...
            [rslt.cRMSE';rslt.tcRMSE'],sum(rslt.cRMSE)/channel_dim,sum(rslt.tcRMSE)/channel_dim);
        fprintf(fid,'normed optm loss:%.4f   true optm loss:%.4f',rslt.loss,rslt.tloss);
    end
    
    
    
    
    
end
fprintf(fid,'\n----------------------------------------------------------------\n');
if useFileLog
    fclose(fid);
end
end
