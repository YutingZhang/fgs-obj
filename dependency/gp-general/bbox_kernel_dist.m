function [K, E] = bbox_kernel_dist(B1,B2,lambda,param_type)

eb2 = exist('B2','var') && ~isempty(B2);

switch param_type
    case 'yxsal'
    case 'ltrb'
        B1 = bbox_ltrb2param(B1,'yxsal');
        if eb2, B2 = bbox_ltrb2param(B2,'yxsal'); end
    case 'yxhwl'
        B1(:,[3 4]) = [(B1(:,3)+B1(:,4))/2, B1(:,3)-B1(:,4)];
        if eb2, B2(:,[3 4]) = [(B2(:,3)+B2(:,4))/2, B2(:,3)-B2(:,4)]; end
    case 'yxhw'
        B1 = bbox_ltrb2param(B1,'yxhw==>yxsal');
        if eb2, B2 = bbox_ltrb2param(B2,'yxhw==>yxsal'); end
    otherwise
        error( 'Unsupported param_type' );
end

if ~eb2, B2 = B1; end

m = size(B1,1); n = size(B2,1);

E = zeros( m, n, 4 );
for d = 1:4
    E(:,:,d) = bsxfun( @minus, B2(:,d).', B1(:,d) );
end
E = E.*E;

% mS = bsxfun( @plus, B1(:,3), B2(:,3).' );
% E(:,:,[1 2]) = bsxfun( @rdivide, E(:,:,[1 2]), exp(mS) );

% mS = bsxfun( @plus, exp(B1(:,3)*2), exp(B2(:,3)*2).' );
% E(:,:,[1 2]) = bsxfun( @rdivide, E(:,:,[1 2]), mS );

baseS = bsxfun( @min, B1(:,3), B2(:,3).' );
dnS = exp(-baseS*2);
E(:,:,[1 2]) = bsxfun( @times, E(:,:,[1 2]), dnS );

K = reshape( reshape(E,m*n,4) * reshape(lambda,4,1), m, n) ;

end
