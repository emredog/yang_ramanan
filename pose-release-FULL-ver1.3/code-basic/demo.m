addpath visualization;
if isunix()
    addpath mex_unix;
elseif ispc()
    addpath mex_pc;
end

compile;

% load and display model
load('PARSE_model'); % original model
% load('PARSE_model_modified2'); % Parse_model_modified2 has interval=2
%load('PARSE_Ri-01.mat'); % Parse_model_modified has interval=1
%visualizemodel(model);
%disp('model template visualization');
%disp('press any key to continue');
%pause;
%visualizeskeleton(model);
%disp('model tree visualization');
%disp('press any key to continue');
%pause;

imlist = dir('images/*.jpg');
% partialScores = zeros(100, 26);
for i = 1:length(imlist)
    % load and display image
    im = imread(['images/' imlist(i).name]);    
    disp(imlist(i).name);
    clf; imshow(im); axis image; axis off; drawnow;
    
    [h, w, channels] = size(im);
%     if h ~= 480
        
        % call detect function
        tic;
        [boxes, scores] = detect_fast(im, model, min(model.thresh,-1));
        dettime = toc; % record cpu time
        if isempty(boxes)
            fprintf('No detection after %.3f seconds for %s\n',dettime, imlist(i).name);
            imwrite(im, ['images/result_' imlist(i).name(1:end-4) '.jpg']);
        else
            [boxes, indexOfMax] = nms(boxes, .1); % nonmaximal suppression
            scores = scores(indexOfMax,:);
            % ED - record scores for each part and each image
%             partialScores(i,:) = scores(1,:);
            
            if (isfield(model, 'Ri'))
                 sumOfScores = sum(scores);
                 normalizedScores = scores / sumOfScores;
                 sz = size(scores);
                 if sz(1) > 1
                     Ri = normalizedScores(1,:) > model.Ri;
                 else
                     Ri = normalizedScores > model.Ri;
                 end
                 tabulate(Ri);
            else
                Ri = ones(size(scores));
            end
            colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
            %showboxes(im, boxes(1,:),colorset, Ri, true, scores); % show the best detection
            showboxes(im, boxes(1,:),colorset, Ri, false); % show the best detection
            %showboxes(im, boxes,colorset);  % show all detections
            %fprintf('detection took %.3f seconds for %s\n',dettime, imlist(i).name);
            %disp('press any key to continue');
            saveas(gcf, ['images/result_' imlist(i).name(1:end-4) '.jpg'] ,'jpg');
        end
        
        
        %pause;
%     else % h=480, process it as 4 parts        
%         for x = 0:1
%             for y = 0:1
%                 imCropped = imcrop(im, [x*320 y*240 320 240]);
%                 % call detect function
%                 tic;
%                 boxes = detect_fast(imCropped, model, min(model.thresh,-1));
%                 dettime = toc; % record cpu time
%                 if isempty(boxes)
%                     fprintf('No detection after %.3f seconds\n',dettime);
%                     imwrite(imCropped, ['images/result_' imlist(i).name(1:end-4) '_' num2str(y) '-' num2str(x) '.jpg']);
%                     %disp('press any key to continue');
%                 else
%                     boxes = nms(boxes, .1); % nonmaximal suppression
%                     colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
%                     showboxes(imCropped, boxes(1,:),colorset); % show the best detection
%                     %showboxes(im, boxes,colorset);  % show all detections
%                     fprintf('detection took %.3f seconds for %s\n',dettime, imlist(i).name);
%                     %disp('press any key to continue');
%                     saveas(gcf, ['images/result_' imlist(i).name(1:end-4) '_' num2str(y) '-' num2str(x) '.jpg'] ,'jpg');
%                 end
%                 
%                 %pause;
%             end
%         end
%     end
    
end
% csvwrite('PARSE-Training-Scores-100.csv', partialScores);
disp('done');
