% ------------------------------------------
% structured loss function for detection
% w             : dim x 1
% bias          : 1 x 1
% xpgt          : dim x # examples (positive example)
% xph (cell)    : dim x # examples (hard negative for positive example)
% stloss (cell) : 1 x # examples (hard negative for positive example)
% xnh (cell)    : dim x # examples (hard negative for negative example)
%

function [loss, grad] = compute_str_loss_sum_L1(theta, xgt, xph, stloss, xnh, C1, C2, Cb)

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

npos_sam = 0;
for i = 1:npos,
    % ground truth
    margin = 1 - wxgt(i);
    margin = max(margin, 0);
    npos_sam = npos_sam + 1;
    
    ploss = ploss + margin;
    
    % const violate
    if margin > 0,
        pwgrad = pwgrad - xgt(:, i);
        pbgrad = pbgrad - 1;
    end
    
    % negative
    if ~isempty(wxph{i}),
        margin = stloss{i} + wxph{i}' - wxgt(i);
        margin = max(margin, 0);
        npos_sam = npos_sam + length(margin);
        
        ploss = ploss + sum(margin);        
        id = margin > 0;
        
        % const violate
        if sum(id) > 0,
            pwgrad = pwgrad + sum(bsxfun(@minus, xph{i}(:, id), xgt(:, i)), 2);
        end
    end
end

loss = loss + (C1/max(npos_sam, 1))*ploss;
wgrad = wgrad + (C1/max(npos_sam, 1))*pwgrad;
bgrad = bgrad + (C1/max(npos_sam, 1))*pbgrad;


% negative examples
nloss = 0;
nwgrad = zeros(size(w));
nbgrad = zeros(size(bias));

nneg_sam = 0;
for i = 1:nneg,
    if ~isempty(xnh{i}),
        margin = 1 + wxnh{i};
        nneg_sam = nneg_sam + length(wxnh{i});
        margin = max(margin, 0);
        
        id = margin > 0;
        
        if sum(id) > 0,
            nloss = nloss + sum(margin);
            nwgrad = nwgrad + sum(xnh{i}(:, id), 2);
            nbgrad = nbgrad + sum(id)*1;
        end
    end
end

loss = loss + (C2/max(nneg_sam, 1))*nloss;
wgrad = wgrad + (C2/max(nneg_sam, 1))*nwgrad;
bgrad = bgrad + (C2/max(nneg_sam, 1))*nbgrad;

grad = [wgrad(:) ; Cb*bgrad(:)];

return;
