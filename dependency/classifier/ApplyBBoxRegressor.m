function new_boxes = ApplyBBoxRegressor( original_boxes, F, regress_model )

T = ApplyRegressor( F, regress_model );  

P = bbox_ltrb2param( original_boxes, 'yxhw' );
Gyx = bsxfun( @plus,  bsxfun( @times, P(:,[3 4]), T(:,[1 2],:) ), P(:,[1 2]) );
Ghw = bsxfun( @times, P(:,[3 4]), exp(T(:,[3 4],:)) );
G = [Gyx,Ghw]; 
G = shiftdim(G,2);

B = bbox_param2ltrb( ...
    reshape( G, size(G,1)*size(G,2), 4 ) , 'yxhw');
B = shiftdim( reshape( B, size(G,1), size(G,2), 4 ), 1 );

new_boxes = B;


