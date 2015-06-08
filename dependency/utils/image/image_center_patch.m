function J = image_center_patch( I, patch_size )

s = size(I);
delta_size = s(1:2) - patch_size;
center_pos = floor(delta_size/2)+1;

J = I( ...
    center_pos(1):center_pos(1)+patch_size(1)-1, ...
    center_pos(2):center_pos(2)+patch_size(2)-1,:);
J = reshape( J, [patch_size, s(3:end)] );

end
