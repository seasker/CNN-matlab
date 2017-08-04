function  [rslt] = cnnshowresult(rslt, opts, x, y, datameta, transmeta)
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
m = size(x, 3);
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
    if ~isequal(opts.lossfunc ,@msefunc)
        fprintf('WANRING: Loss evaluation indicator will be mse,this is different from the loss function given\n');
    end
    channels = size(x, 4);
    rslt.dev = rslt.pv - y;
    rslt.ese = 1/2 * rslt.dev.^2;
    rslt.ere = zeros(size(y));
    rslt.ere(y>0) = abs(rslt.dev(y>0)./ y(y>0));
    rslt.mcloss = squeeze(sum(reshape(rslt.ese,[],channels,m)));
    rslt.mcmre = squeeze(mean(reshape(rslt.ere,[],channels,m)));
    rslt.closs = squeeze(mean(rslt.mcloss,2));
    rslt.cmre = squeeze(mean(rslt.mcmre,2));
    if nargin > 4
        [~,rslt.tpv] = datatransform(x,rslt.pv,transmeta,datameta);
        [~,tyv] = datatransform(x,y,transmeta,datameta);
        rslt.tdev = rslt.tpv - tyv;
        rslt.tese = 1/2 * rslt.tdev.^2;
        rslt.tere = zeros(size(tyv));
        rslt.tere(tyv>0) = abs(rslt.tdev(tyv>0)./ tyv(tyv>0));
        rslt.tmcloss = squeeze(sum(reshape(rslt.tese,[],channels,m)));
        rslt.tmcmre = squeeze(mean(reshape(rslt.tere,[],channels,m)));
        rslt.tcloss = squeeze(mean(rslt.tmcloss,2));
        rslt.tcmre = squeeze(mean(rslt.tmcmre,2));
        rslt.tloss = opts.lossfunc(rslt.tpv, tyv);
        rslt.tmre = mrefunc(rslt.tpv, tyv);
    end
    for i = 1 : channels
        fprintf(fid,'CHANNEL %d   loss:%.4f   mre:%.4f  ', i, rslt.closs(i), rslt.cmre(i));
        if nargin > 4
            fprintf(fid,'true loss:%.4f  true mre:%.4f', rslt.tcloss(i), rslt.tcmre(i));
        end
        fprintf(fid,'\n');
    end
    fprintf(fid,'loss:%.4f  mre:%.4f  ',rslt.loss, rslt.mre);
    if nargin > 4
        fprintf(fid,'true loss:%.4f  true mre:%.4f',rslt.tloss,rslt.tmre);
    end
    fprintf(fid,'\n----------------------------------------------------------------\n');
    if useFileLog
        fclose(fid);
    end
end

