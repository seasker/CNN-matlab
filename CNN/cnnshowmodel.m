function cnnshowmodel(net, opts)
global useFileLog;
if useFileLog
    if isfield(opts,'logfilename')
        fid = fopen(opts.flname, 'a+');
    else
        fid = fopen('result.txt', 'a+');
    end
else
    fid = 1;
end
n = numel(net.layers);
if isfield(net,'name')
    fprintf(fid,'Net-Name-%s\n',net.name);
end
if isfield(net,'stage')
    fprintf(fid,'Net-Stage-%d\n',net.stage);
end
fprintf(fid,'<model>\n');
totalparam = 0;
for l = 1 : n  %  layer
    if strcmp(net.layers{l}.type, 'i')
        rf = [1,1];
        fprintf(fid,'Input Layer: channels=%d  mapsize=[%d,%d]  rf=[%d,%d]  ',net.layers{l}.outputchannels,...
            net.layers{l}.mapsize(1), net.layers{l}.mapsize(2), rf(1), rf(2));
        if isfield(net.layers{l},'activefunc')
            fprintf(fid,'activefunc=%s',func2str(net.layers{l}.activefunc));
        end
    elseif strcmp(net.layers{l}.type, 's')
        rf = (rf-1).* net.layers{l}.stride + net.layers{l}.poolsize;
        fprintf(fid,'Pooling Layer: channels=%d  mapsize=[%d,%d]  kernelsize=[%d,%d]  stride=[%d,%d]  rf=[%d,%d]  poolmethod=%s  ',...
            net.layers{l}.outputchannels,net.layers{l}.mapsize(1),net.layers{l}.mapsize(2),net.layers{l}.poolsize(1),net.layers{l}.poolsize(2),...
            net.layers{l}.stride(1),net.layers{l}.stride(2), rf(1), rf(2), net.layers{l}.poolmethod);
        if isfield(net.layers{l},'activefunc')
            fprintf(fid,'activefunc=%s',func2str(net.layers{l}.activefunc));
        end
    elseif strcmp(net.layers{l}.type, 'c')
        numkernelparam = 0;
        numbiasparam = 0;
        rf = (rf -1).* net.layers{l}.stride + net.layers{l}.kernelsize;
        for j = 1 : net.layers{l}.outputchannels
            for i = 1 : net.layers{l-1}.outputchannels
                numkernelparam = numkernelparam + numel(net.layers{l}.k{i}{j});
            end
            numbiasparam = numbiasparam + numel(net.layers{l}.b{j});
        end
        numparam = numkernelparam + numbiasparam;
        totalparam = totalparam + numparam;
        fprintf(fid,'Convolution Layer: channels=%d  mapsize=[%d,%d]  kernelsize=[%d,%d]  stride=[%d,%d]  rf=[%d,%d]  numparam=%d(%d + %d)  ',...
            net.layers{l}.outputchannels,net.layers{l}.mapsize(1),net.layers{l}.mapsize(2),net.layers{l}.kernelsize(1),net.layers{l}.kernelsize(2),...
            net.layers{l}.stride(1),net.layers{l}.stride(2),rf(1),rf(2),numparam,numkernelparam, numbiasparam );
        if isfield(net.layers{l},'activefunc')
            fprintf(fid,'activefunc=%s',func2str(net.layers{l}.activefunc));
        end
    elseif strcmp(net.layers{l}.type, 'bn')
        numgammaparam = 0;
        numbetaparam = 0;
        if strcmp(net.layers{l-1}.type,'fc')
            numgammaparam = numgammaparam + numel(net.layers{l}.gamma);
            numbetaparam = numbetaparam + numel(net.layers{l}.beta);
        else
            for j = 1 : net.layers{l}.outputchannels
                numgammaparam = numgammaparam + numel(net.layers{l}.gamma{j});
                numbetaparam = numbetaparam + numel(net.layers{l}.beta{j});
            end
            numparam = numgammaparam + numbetaparam;
            totalparam = totalparam + numparam;
        end
        
        fprintf(fid,'Batch Normalization Layer: channels=%d  mapsize=[%d,%d]  rf=[%d,%d]  numparam=%d(%d + %d)  ',...
            net.layers{l}.outputchannels, net.layers{l}.mapsize(1), net.layers{l}.mapsize(2), rf(1), rf(2), numparam, numgammaparam, numbetaparam);
        if isfield(net.layers{l},'activefunc')
            fprintf(fid,'activefunc=%s',func2str(net.layers{l}.activefunc));
        end
    elseif strcmp(net.layers{l}.type, 'fc')
        numweightparam = numel(net.layers{l}.w);
        numbiasparam = numel(net.layers{l}.b);
        numparam = numweightparam + numbiasparam;
        totalparam = totalparam + numparam;
        if l == n
            fprintf(fid,'Probability Layer: classnum=%d  mapsize=[%d,%d]  rf=[%d,%d]  numparam=%d(%d + %d)  ',...
                net.layers{l}.outputchannels,net.layers{l}.mapsize(1), net.layers{l}.mapsize(2), rf(1),rf(2),numparam, numweightparam, numbiasparam);
        else
            fprintf(fid,'Full Connection Layer: channels=%d mapsize=[%d,%d], rf=[%d,%d]  numparam=%d(%d + %d)  ',...
                net.layers{l}.outputchannels,net.layers{l}.mapsize(1), net.layers{l}.mapsize(2), rf(1),rf(2),numparam, numweightparam, numbiasparam);
            
        end
        if isfield(net.layers{l},'activefunc')
            fprintf(fid,'activefunc=%s',func2str(net.layers{l}.activefunc));
        end
    end
    fprintf(fid,'\n');
end
fprintf(fid,'totalparam:%d',totalparam);
fprintf(fid,'\n------------------------------------------------------------------------------------------------------------------------------------------\n');
if useFileLog
    fclose(fid);
end
end

