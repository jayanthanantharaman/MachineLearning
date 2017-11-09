function [realOutput, layerOutputCells] = FwdPropagate(in, layer, weightCell, biasCell)
    layerCount = size(layer, 2);
    layerOutputCells = cell(1, layerCount);
    out = in;
    for layerIndex = 1:layerCount
        X = out;
        bias = biasCell{layerIndex};
        out = sigmoid(X * weightCell{layerIndex} + bias);
        layerOutputCells{layerIndex} = out;
    end
    realOutput = out;    
end