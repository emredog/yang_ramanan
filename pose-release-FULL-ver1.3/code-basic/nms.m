function [top, pick] = nms(boxes, overlap,numpart)
% Non-maximum suppression.
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected detection.

if nargin < 2
  overlap = 0.5;
end
if nargin < 3
  numpart = floor(size(boxes,2)/4);
end

top = [];
if isempty(boxes)
  return;
end

% throw away boxes if the number of candidates are too many
if size(boxes,1) > 1000
  [foo,I] = sort(boxes(:,end),'descend');
  boxes = boxes(I(1:1000),:);
end

% collect bounding boxes and scores  
x1 = zeros(size(boxes,1),numpart);
y1 = zeros(size(boxes,1),numpart);
x2 = zeros(size(boxes,1),numpart);
y2 = zeros(size(boxes,1),numpart);
area = zeros(size(boxes,1),numpart);
for p = 1:numpart
  x1(:,p) = boxes(:,1+(p-1)*4);
  y1(:,p) = boxes(:,2+(p-1)*4);
  x2(:,p) = boxes(:,3+(p-1)*4);
  y2(:,p) = boxes(:,4+(p-1)*4);
  area(:,p) = (x2(:,p)-x1(:,p)+1) .* (y2(:,p)-y1(:,p)+1);
end
% compute the biggest boxes covering detection
%ED - for each detection candidate, define the largest bounding box
minX1 = min(x1,[],2); 
minY1 = min(y1,[],2);
maxX2 = max(x2,[],2);
maxY2 = max(y2,[],2);
rarea = (maxX2-minX1+1) .* (maxY2-minY1+1);
% combine parts and biggest covering boxes
x1 = [x1 minX1];
y1 = [y1 minY1];
x2 = [x2 maxX2];
y2 = [y2 maxY2];
area = [area rarea];

rootScores = boxes(:,end);
[vals1, I] = sort(rootScores);

pick = [];
while ~isempty(I)
  last = length(I);
  i = I(last);
  pick = [pick; i];

  xx1 = bsxfun(@max,x1(i,:), x1(I,:)); %ED - foreach column c, if ( x1(I,c) < x1(i,c) ) then x1(I,c) = x1(i,c) 
  yy1 = bsxfun(@max,y1(i,:), y1(I,:));
  xx2 = bsxfun(@min,x2(i,:), x2(I,:));
  yy2 = bsxfun(@min,y2(i,:), y2(I,:));

  w = xx2-xx1+1; w(w<0) = 0;
  h = yy2-yy1+1; h(h<0) = 0;    
  inter  = w.*h; %ED - calculate all areas

  repmatResult = repmat(area(i,:),size(inter,1),1); %ED a matrix full of "area(i,:)"
  o = inter ./ repmatResult; %ED - elementwise division of actual areas by "area(i,:)"
  o = max(o,[],2);
  I(o > overlap) = [];
end  
top = boxes(pick,:);
