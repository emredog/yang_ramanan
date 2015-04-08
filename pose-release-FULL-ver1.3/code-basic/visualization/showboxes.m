function showboxes(im, boxes, partcolor, ri, showScores, scores)

numparts = length(partcolor);

if nargin < 4
    ri = ones(numparts); % ED - ones means "show all"
end

if nargin < 5
    isShowScores = false;
else
    isShowScores = showScores;
end

if (nargin < 6) && isShowScores
    error('detectFast:showboxes', 'Not enough argument to display scores.');
end



imagesc(im); axis image; axis off;
if ~isempty(boxes)
    
    box = boxes(:,1:4*numparts);
    xy = reshape(box,size(box,1),4,numparts);
    xy = permute(xy,[1 3 2]);
    x1 = xy(:,:,1);
    y1 = xy(:,:,2);
    x2 = xy(:,:,3);
    y2 = xy(:,:,4);
    for p = 1:size(xy,2)
        if ri(p)
            line([x1(:,p) x1(:,p) x2(:,p) x2(:,p) x1(:,p)]',[y1(:,p) y2(:,p) y2(:,p) y1(:,p) y1(:,p)]',...
                'color',partcolor{p},'linewidth',2);
            
            if isShowScores
                text(x1(1,p), y1(1,p) + 3 , num2str(scores(1,p)), 'color',partcolor{p});
            end
        end
    end
end
drawnow;
