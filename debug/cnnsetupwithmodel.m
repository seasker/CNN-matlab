function [ cnn_dest ] = cnnsetupwithmodel( cnn_dest, cnn_source )
n = numel(cnn_dest.layers);
cnn_dest.regloss = 0;
for l = 1 : n  %  layer
    if strcmp(cnn_dest.layers{l}.type, 'c')
        for j = 1 : cnn_dest.layers{l}.outputchannels  %  output /
            for i = 1 : cnn_dest.layers{l-1}.outputchannels  %  input map
                cnn_dest.layers{l}.k{i}{j} = cnn_source.layers{l}.k{i}{j};
                cnn_dest.layers{l}.ik{i}{j} = cnn_source.layers{l}.ik{i}{j};
                cnn_dest.regloss = cnn_dest.regloss + 0.5 * cnn_dest.layers{l}.decay * sum(cnn_dest.layers{l}.k{i}{j}(:).^2);
            end
            cnn_dest.layers{l}.b{j} = cnn_source.layers{l}.b{j};
            cnn_dest.layers{l}.ib{j}= cnn_source.layers{l}.ib{j};
        end
    elseif strcmp(cnn_dest.layers{l}.type, 'bn')
        if isfield(cnn_dest.layers{l-1}, 'a')
            for j = 1 : cnn_dest.layers{l}.outputchannels
                cnn_dest.layers{l}.gamma{j} = cnn_source.layers{l}.gamma{j};
                cnn_dest.layers{l}.igamma{j} = cnn_source.layers{l}.igamma{j};
                cnn_dest.layers{l}.beta{j} = cnn_source.layers{l}.beta{j};
                cnn_dest.layers{l}.ibeta{j} = cnn_source.layers{l}.ibeta{j};
                cnn_dest.regloss = cnn_dest.regloss + 0.5 * cnn_dest.layers{l}.decay * sum(cnn_dest.layers{l}.gamma{j}(:).^2);
                
            end
        else
            cnn_dest.layers{l}.gamma = cnn_source.layers{l}.gamma;
            cnn_dest.layers{l}.igamma = cnn_source.layers{l}.igamma;
            cnn_dest.layers{l}.beta = cnn_source.layers{l}.beta;
            cnn_dest.layers{l}.ibeta = cnn_source.layers{l}.ibeta;
            cnn_dest.regloss = cnn_dest.regloss + 0.5 * cnn_dest.layers{l}.decay * sum(cnn_dest.layers{l}.gamma(:).^2);
        end
    elseif strcmp(cnn_dest.layers{l}.type, 'fc')
        
        cnn_dest.layers{l}.w = cnn_source.layers{l}.w;
        cnn_dest.layers{l}.iw = cnn_source.layers{l}.iw;
        cnn_dest.layers{l}.b = cnn_source.layers{l}.b;
        cnn_dest.layers{l}.ib = cnn_source.layers{l}.ib;
        cnn_dest.regloss = cnn_dest.regloss + 0.5 * cnn_dest.layers{l}.decay * sum(cnn_dest.layers{l}.w(:).^2);
    end
end


end

