function [boxes, scores] = detect_fast(im, model, thresh)
% boxes = detect(im, model, thresh)
% Detect objects in input using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part

% Compute the feature pyramid and prepare filter
tic;
pyra     = featpyramid(im,model); %ED - Calculates HOG features for different scales
pyratime = toc;
fprintf('Pyramid constructed in %.3f seconds\n',pyratime);
interval = model.interval;
levels   = 1:length(pyra.feat);

% Cache various statistics derived from model
[components,filters,resp] = modelcomponents(model,pyra);
boxes = zeros(10000,length(components{1})*4+2);
scores = zeros(10000, length(components{1})); % ED - preallocation makes it faster (instead of dynamic allocation)
cnt   = 0;

% Iterate over scales and components,
for rlevel = levels,
  for c  = 1:length(model.components),
    parts    = components{c};
    numparts = length(parts);

    %tic;
    % Local scores
    for k = 1:numparts,
      f     = parts(k).filterid;
      level = rlevel-parts(k).scale*interval;
      if isempty(resp{level}),
        resp{level} = fconv(pyra.feat{level},filters,1,length(filters)); %ED - Calculates HOG responses of pyramid against the trained model.
      end
      for fi = 1:length(f)
        parts(k).score(:,:,fi) = resp{level}{f(fi)};
      end
      parts(k).level = level;
    end
    %localTime = toc;
    %fprintf('Local scores for level: %d component %d in %.3f seconds\n',rlevel, c, localTime);
    
    %tic;
    % Walk from leaves to root of tree, passing message to parent
    for k = numparts:-1:2,
      par = parts(k).parent;
      [msg,parts(k).Ix,parts(k).Iy,parts(k).Ik] = passmsg(parts(k),parts(par)); %ED - Calculate deformation score between k and k-1(=par, parent of k) --> msg
                                                                                %ED - k.Ix and k.Iy contains, for each (pi,ti) of k, potential pi of k-1
                                                                                %ED - k.Ik contains, for each (pi,ti) of k, potential ti of k-1
      parts(par).score = parts(par).score + msg; %ED - score of PARENT is updated with msg (deformation score between k and k-1(=par, parent of k))
    end
    %passmsgTime = toc;
    %fprintf('Msg  for level: %d component %d in %.3f seconds\n',rlevel, c, passmsgTime);

    % Add bias to root score
    parts(1).score = parts(1).score + parts(1).b; %ED - Bias score is from model components, ie, a learned value
    [rscore Ik]    = max(parts(1).score,[],3);    %ED - For the root part, select best ti for every pi, based on obtained score.

    % Walk back down tree following pointers
    %thresh = -100.0;
    [Y,X] = find(rscore >= thresh); %ED - finds indices of rscore elements that are >= threshold, let's say there are 250 candidates:
    if length(X) > 1,
      I   = (X-1)*size(rscore,1) + Y;
      [box, scoresforThisLevel] = backtrack(X,Y,Ik(I),parts,pyra); %ED - this function backtrack a single root candidate until it reaches the leaf
      i   = cnt+1:cnt+length(I);
      boxes(i,:) = [box repmat(c,length(I),1) rscore(I)];
      %ED - added "scores"
      scores(i,:) = scoresforThisLevel;
     
      cnt = i(end);
    end
  end
end

boxes = boxes(1:cnt,:);

% Cache various statistics from the model data structure for later use  
function [components,filters,resp] = modelcomponents(model,pyra)
  components = cell(length(model.components),1);
  for c = 1:length(model.components),
    for k = 1:length(model.components{c}),
      p = model.components{c}(k);
      [p.w,p.defI,p.starty,p.startx,p.step,p.level,p.Ix,p.Iy] = deal([]);
      [p.scale,p.level,p.Ix,p.Iy] = deal(0);
      
      % store the scale of each part relative to the component root
      par = p.parent;      
      assert(par < k);
      p.b = [model.bias(p.biasid).w];
      p.b = reshape(p.b,[1 size(p.biasid)]);
      p.biasI = [model.bias(p.biasid).i];
      p.biasI = reshape(p.biasI,size(p.biasid));
      p.sizx  = zeros(length(p.filterid),1);
      p.sizy  = zeros(length(p.filterid),1);
      
      for f = 1:length(p.filterid)
        x = model.filters(p.filterid(f));
        [p.sizy(f) p.sizx(f) foo] = size(x.w);
%         p.filterI(f) = x.i;
      end
      for f = 1:length(p.defid)	  
        x = model.defs(p.defid(f));
        p.w(:,f)  = x.w';
        p.defI(f) = x.i;
        ax  = x.anchor(1);
        ay  = x.anchor(2);    
        ds  = x.anchor(3);
        p.scale = ds + components{c}(par).scale;
        % amount of (virtual) padding to hallucinate
        step     = 2^ds;
        virtpady = (step-1)*pyra.pady;
        virtpadx = (step-1)*pyra.padx;
        % starting points (simulates additional padding at finer scales)
        p.starty(f) = ay-virtpady;
        p.startx(f) = ax-virtpadx;      
        p.step   = step;
      end
      components{c}(k) = p;
    end
  end
  
  resp    = cell(length(pyra.feat),1);
  filters = cell(length(model.filters),1);
  for i = 1:length(filters),
    filters{i} = model.filters(i).w;
  end

% Given a 2D array of filter scores 'child',
% (1) Apply distance transform
% (2) Shift by anchor position of part wrt parent
% (3) Downsample if necessary
function [score,Ix,Iy,Ik] = passmsg(child,parent)
  INF = 1e10;
  K   = length(child.filterid);
  Ny  = size(parent.score,1);
  Nx  = size(parent.score,2);  
  [Ix0,Iy0,score0] = deal(zeros([Ny Nx K]));

  for k = 1:K
    [score0(:,:,k),Ix0(:,:,k),Iy0(:,:,k)] = shiftdt(child.score(:,:,k), child.w(1,k), child.w(2,k), child.w(3,k), child.w(4,k),child.startx(k),child.starty(k),Nx,Ny,child.step);
  end
  %ED - shiftdt makes -some- distance transform to rapidly calculate the
  %score related to distance [dx dx^2 dy dy^2] = ψ(p_child, p_parent)
  
%  csvwrite('inp_score0.csv', score0);
%  csvwrite('inp_Ix0.csv', Ix0);
%  csvwrite('inp_Iy0.csv', Iy0);
%  child_b = child.b;
%  csvwrite('inp_childB.csv', child_b);

  % At each parent location, for each parent mixture 1:L, compute best child mixture 1:K
  L  = length(parent.filterid);
  N  = Nx*Ny;
  i0 = reshape(1:N,Ny,Nx);
  [score,Ix,Iy,Ix,Ik] = deal(zeros(Ny,Nx,L));
  
  for l = 1:L
    b = child.b(1,l,:);
    deletMe = bsxfun(@plus,score0,b);    
    [score(:,:,l),I] = max(deletMe,[],3);    
    i = i0 + N*(I-1);
    Ix(:,:,l)    = Ix0(i);
    Iy(:,:,l)    = Iy0(i);
    Ik(:,:,l)    = I;
  end
  
    
%  fprintf('Rest of passMsg in %.4f seconds\n',time);
  
%  csvwrite('out_score.csv', score);
%  csvwrite('out_Ix.csv', Ix);
%  csvwrite('out_Iy.csv', Iy);
%  csvwrite('out_Ik.csv', Ik);

% Backtrack through DP msgs to collect ptrs to part locations
%ED - DP:dynamic programming
function [box, scores] = backtrack(x,y,mix,parts,pyra)
  numx     = length(x);
  numparts = length(parts);
  
  xptr = zeros(numx,numparts);
  yptr = zeros(numx,numparts);
  mptr = zeros(numx,numparts);
  box  = zeros(numx,4,numparts);
  %ED copy the scores of each part for every candidate:
  scores = zeros(numx,numparts);

  for k = 1:numparts,
    curPart   = parts(k);
    if k == 1, %ED - If current part is root:
      xptr(:,k) = x;
      yptr(:,k) = y;
      mptr(:,k) = mix;
      [h,w,foo] = size(parts(k+1).Ix); %ED - get patch size
      Icur = (mix-1)*h*w + (x-1)*h + y; %ED - fetch indices of best scoring pi for current part
      scores(:,k) = curPart.score(Icur); %ED - fetch best scores for current part
    else %ED - if current part is any other part than root
      % I = sub2ind(size(p.Ix),yptr(:,par),xptr(:,par),mptr(:,par));
      curParent = curPart.parent;
      [h,w,foo] = size(curPart.Ix);
      I   = (mptr(:,curParent)-1)*h*w + (xptr(:,curParent)-1)*h + yptr(:,curParent);      
      xptr(:,k) = curPart.Ix(I);
      yptr(:,k) = curPart.Iy(I);
      mptr(:,k) = curPart.Ik(I);
      Icur = (mptr(:,k)-1)*h*w + (xptr(:,k)-1)*h + yptr(:,k); %ED - fetch indices of best scoring pi for current part
      scores(:,k) = curPart.score(Icur); %ED - fetch best scores for current part
    end
    scale = pyra.scale(curPart.level);
    x1 = (xptr(:,k) - 1 - pyra.padx)*scale+1;
    y1 = (yptr(:,k) - 1 - pyra.pady)*scale+1;
    x2 = x1 + curPart.sizx(mptr(:,k))*scale - 1;
    y2 = y1 + curPart.sizy(mptr(:,k))*scale - 1;
    box(:,:,k) = [x1 y1 x2 y2];
  end
  box = reshape(box,numx,4*numparts);
