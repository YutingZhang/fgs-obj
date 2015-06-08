function varargout = bbox_param2ltrb( P, param_type )
% convert BBOX coordinate (y1,x2,y2,x2) from parameterized form
% scale = sqrt( height x width )
% aspect = height/width
% refer to bbox_ltrb2param

switch param_type
    case 'yxhw'
        B = [P(:,1)-P(:,3)*0.5,P(:,2)-P(:,4)*0.5,P(:,1)+P(:,3)*0.5,P(:,2)+P(:,4)*0.5];
    case 'yxhwl'
        P(:,3:4) = exp( P(:,3:4) );
        B = bbox_param2ltrb( P, 'yxhw' );
    case 'yxsal'
        H  = exp(P(:,3)+P(:,4)*0.5);
        W  = exp(P(:,3)-P(:,4)*0.5);
        P(:,[3 4]) = [H,W];
        B = bbox_param2ltrb( P, 'yxhw' );
    otherwise
        error( 'Unknown param_type' );
end

if nargout > 1
    varargout = cell(1,size(B,2));
    for d = 1:size(B,2)
        varargout{d} = B(:,d);
    end
else
    varargout{1} = B;
end

end
