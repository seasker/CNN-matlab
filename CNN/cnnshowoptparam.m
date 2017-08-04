function cnnshowoptparam(opts)
global useFileLog;
if useFileLog
    if isfield(opts,'logfilename')
        fid = fopen(opts.logfilename, 'a+');
    else
        fid = fopen('result.txt', 'a+');
    end
else
    fid = 1;
end
fns = fieldnames(opts);
fprintf(fid,'<parameters>\n');
for i = 1 : length(fns)
    fv = getfield(opts,fns{i});
    if isinteger(fv)
        fprintf(fid, '  %s=%d',fns{i}, fv);
    elseif isfloat(fv)
        fprintf(fid,'  %s=%.4f',fns{i}, fv);
    elseif isa(fv,'function_handle')
        fprintf(fid,'  %s=%s',fns{i}, func2str(fv));
    end
end
fprintf(fid,'\n----------------------------------------------------------------------------------------------------------------------------------------------------\n');
if useFileLog
    fclose(fid);
end
end

