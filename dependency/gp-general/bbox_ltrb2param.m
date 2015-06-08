function varargout = bbox_ltrb2param( B, param_type )
% convert BBOX coordinate (y1,x2,y2,x2) to parametrized form
%
% yxhw:
% ( centerY, centerX, H, W )
%
% yxhwl:
% ( centerY, centerX, log(H), log(W) )
%
% yxsal:
% ( centerY, centerX, log(scale), log(aspect) )
%   scale = sqrt( height x width )
%   aspect = height/width
%   normalizedX = centerX/scale, normalizedY = centerY/scale

switch param_type
    case 'yxhw'
        H = B(:,3) - B(:,1);
        W = B(:,4) - B(:,2);
        Y = ( B(:,1) + B(:,3) ) / 2;
        X = ( B(:,2) + B(:,4) ) / 2;
        P = [Y, X, H, W];
    case 'yxhwl'
        M = bbox_ltrb2param( B, 'yxhw' );
        P = bbox_ltrb2param( M, 'yxhw==>yxhwl' );
    case 'yxsal'
        M = bbox_ltrb2param( B, 'yxhw' );
        P = bbox_ltrb2param( M, 'yxhw==>yxsal' );
    case 'yxhw==>yxhwl'
        P = [ B(:,1:2), log( B(:,3:4) ) ];
    case 'yxhw==>yxsal'
        logS = log( B(:,3).*B(:,4) )/2;
        logA = log( B(:,3) ) - log( B(:,4) );
        P = [B(:,1:2),logS,logA];
    otherwise
        error( 'Unknown param_type' );
end

if nargout > 1
    varargout = cell(1,size(P,2));
    for d = 1:size(P,2)
        varargout{d} = P(:,d);
    end
else
    varargout{1} = P;
end

end
