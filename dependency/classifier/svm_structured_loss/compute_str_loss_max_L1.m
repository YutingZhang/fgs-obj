% ------------------------------------------
% structured loss function for detection
% w             : dim x 1
% bias          : 1 x 1
% xpgt          : dim x # examples (positive example)
% xph (cell)    : dim x # examples (hard negative for positive example)
% stloss (cell) : 1 x # examples (hard negative for positive example)
% xnh (cell)    : dim x # examples (hard negative for negative example)
%

function [loss, grad] = compute_str_loss_max_L1(theta, xgt, xph, stloss, xnh, C1, C2, Cb)

if ~exist('C1', 'var'),
    C1 = 1;
end
if ~exist('C2', 'var'),
    C2 = C1;
end
if ~exist('Cb', 'var'),
    Cb = 1;
end


w = theta(1:end-1);
bias = Cb*theta(end);
wgrad = zeros(size(w));
bgrad = zeros(size(bias));

loss = 0;


npos = length(xph);
nneg = length(xnh);

% # samples for positive and negative bags
posidx = zeros(length(xph), 1);
for i = 1:length(xph),
    posidx(i) = size(xph{i}, 2);
end
negidx = zeros(length(xnh), 1);
for i = 1:length(xnh),
    negidx(i) = size(xnh{i}, 2);
end


% weights
loss = loss + 0.5*sum(w.^2);
wgrad = wgrad + w;

loss = loss + 0.5*sum(bias.^2);
bgrad = bgrad + bias;


% -- feed-forward encoding
% positive, ground-truth
wxgt = bsxfun(@plus, w'*xgt, bias);
% positive bags
xph_mat = cell2mat(xph');
wxph = bsxfun(@plus, w'*xph_mat, bias);
wxph = mat2cell(wxph.', posidx);
% negative bags
xnh_mat = cell2mat(xnh');
wxnh = bsxfun(@plus, w'*xnh_mat, bias);
wxnh = mat2cell(wxnh.', negidx);


% positive examples
ploss = 0;
pwgrad = zeros(size(w));
pbgrad = zeros(size(bias));

for i = 1:npos,
    if ~isempty(wxph{i}),
        margin1 = stloss{i} + wxph{i}' - wxgt(i);
        [margin1, id1] = max(margin1);
    else
        margin1 = 0;
        id1 = [];
    end
    
    margin2 = 1 - wxgt(i);
    margin = max(max(margin1, margin2), 0);
    
    ploss = ploss + margin;
    
    if margin > 0,
        if margin1 > margin2,
            % const1 violate
            pwgrad = pwgrad + bsxfun(@minus, xph{i}(:, id1), xgt(:, i));
        else
            % const2 violate
            pwgrad = pwgrad - xgt(:, i);
            pbgrad = pbgrad - 1;
        end
    end
end

loss = loss + (C1/max(npos, 1))*ploss;
wgrad = wgrad + (C1/max(npos, 1))*pwgrad;
bgrad = bgrad + (C1/max(npos, 1))*pbgrad;


% negative examples
nloss = 0;
nwgrad = zeros(size(w));
nbgrad = zeros(size(bias));

for i = 1:nneg,
    if ~isempty(xnh{i}),
        [margin1, id1] = max(1 + wxnh{i});
        margin = max(margin1, 0);
        
        if margin > 0,
            nloss = nloss + margin;
            nwgrad = nwgrad + xnh{i}(:, id1);
            nbgrad = nbgrad + 1;
        end
    end
end

loss = loss + (C2/max(nneg, 1))*nloss;
wgrad = wgrad + (C2/max(nneg, 1))*nwgrad;
bgrad = bgrad + (C2/max(nneg, 1))*nbgrad;

grad = [wgrad(:) ; Cb*bgrad(:)];

return;
