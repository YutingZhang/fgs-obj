function show_bboxes( I, boxes, v, box_color, display_type, tag_loc )

if ~exist( 'v', 'var' )
    v = [];
end

if ~exist( 'box_color', 'var' )
    box_color = '';
end

if ~exist( 'display_type', 'var' ) || isempty( display_type )
    display_type = 'rank'; % 'score'
end

if ~exist( 'tag_loc', 'var' ) || isempty( tag_loc )
    tag_loc = 1; % 'score'
end


if isempty(box_color)
    box_color = 'red';
end

boundary_color = '';
if iscell(box_color)
    if length(box_color)>=3
        boundary_color = box_color{3};
    end
    if length(box_color)>=2
        text_color = box_color{2};
    end
    box_color  = box_color{1};
else
    text_color = 'white';
end


if ~isempty(I)
    imshow(I);
end

hold on

if ~isempty(v)
    
    if length(v) ~= size(boxes,1)
        error( 'The length of v (vector) and boxes (1st dim) should be consistent.' );
    end
    
    [~,sortedIdx] = sort(v);
    v = v(sortedIdx);
    boxes = boxes(sortedIdx,:);
    cm = jet(length(v));
end

for i = 1:size(boxes,1)
    y1 = boxes(i,1); x1 = boxes(i,2);
    y2 = boxes(i,3); x2 = boxes(i,4);
    X  = [x1 x2 x2 x1];
    Y  = [y1 y1 y2 y2];
    
%    if isempty(v)
%        patch(X,Y,'white','FaceColor','none','EdgeColor','white','LineWidth',3);
%        patch(X,Y,'red',  'FaceColor','none','EdgeColor','red',  'LineWidth',1.5);
%    else
%        patch(X,Y,'white','FaceColor','none','EdgeColor','white','LineWidth',3);
%        patch(X,Y,'red','FaceAlpha',0.2,'FaceColor',cm(i,:),'EdgeColor','none');
%    end
    % patch(X,Y,'white','FaceColor','none','EdgeColor','white','LineWidth',2.5);
    
    if ~isempty( boundary_color )
        line([x1, x2, x2, x1, x1], [y1, y1, y2, y2 ,y1], [0 0 0 0 0], 'Color', ...
            boundary_color, 'LineWidth', 3 );
    end

    patch(X,Y,box_color,  'FaceColor','none','EdgeColor', box_color,  'LineWidth',1.5);
    
    if ~isempty( v )
        switch display_type
            case 'rank'
                tt = int2str(size(boxes,1)-i+1);
            case 'score'
                tt = sprintf('%.2f',v(i));
            otherwise
                error( 'Unrecognized display type' );
        end
        
        switch tag_loc
            case 1
                text( x1, y1, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            case 2
                text( x2, y1, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            case 3
                text( x2, y2, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            case 4
                text( x1, y2, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            case -1
                text( x1, y1, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            case -2
                text( x2, y1, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            case -3
                text( x2, y2, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            case -4
                text( x1, y2, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', box_color, ...
                    'Margin', 1);
            otherwise
                error( 'Unrecognized tag_loc' );
        end
    end
end

hold off

end
