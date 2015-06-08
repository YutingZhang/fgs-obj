function A = vocLoadXML( xml_file_path )

X = VOCreadxml( xml_file_path );
X = X.annotation;

A.im  = X.filename;
A.obj = repmat( struct(), 1, length(X.object) );
[ A.obj.type ] = X.object.name;
for r = 1:length(X.object)
    A.obj(r).x1=str2double(X.object(r).bndbox.xmin)+1; % 0-base to 1-base
    A.obj(r).y1=str2double(X.object(r).bndbox.ymin)+1;
    A.obj(r).x2=str2double(X.object(r).bndbox.xmax)+1;
    A.obj(r).y2=str2double(X.object(r).bndbox.ymax)+1;
    if str2double(X.object(r).difficult);
        A.obj(r).difficulty=inf;
    else
        A.obj(r).difficulty=0;
    end
    A.obj(r).truncation=str2double(X.object(r).truncated);
    % **** need read pose in the furture. {'Left', 'Right', ... } ==> angle
end

end
