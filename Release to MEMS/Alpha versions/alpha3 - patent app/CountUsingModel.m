clear
clc
close all

load net;

% param
cproxpixel = 30; % # of pixels near each centre looked at to count proximal centres
storeImgEachDot = true;
storeInPredictionFolder = true;
browseImgDirectory = true;

% get input_image path and filename
if browseImgDirectory
    [filename, pathname] = uigetfile(...    
    {'*.jpg; *.JPG; *.jpeg; *.JPEG; *.img; *.IMG; *.tif; *.TIF; *.tiff, *.TIFF','Supported Files (*.jpg,*.img,*.tiff,)'; ...
    '*.jpg','jpg Files (*.jpg)';...
    '*.JPG','JPG Files (*.JPG)';...
    '*.jpeg','jpeg Files (*.jpeg)';...
    '*.JPEG','JPEG Files (*.JPEG)';...
    '*.img','img Files (*.img)';...
    '*.IMG','IMG Files (*.IMG)';...
    '*.tif','tif Files (*.tif)';...
    '*.TIF','TIF Files (*.TIF)';...
    '*.tiff','tiff Files (*.tiff)';...
    '*.TIFF','TIFF Files (*.TIFF)'},...    
    'MultiSelect', 'on');
    input_img = strcat(pathname,filename);
else
    input_img = '2020-09-25_mgLAMP with Quasr-BIP12nt_TritonX100_5_gel_.lif_40min_NTC_2_ch00.tif';% Specify the filename here
end

A = imread(input_img);
sens_array = [0.30 0.30 0.25 0.35 0.42 0.30 0.25 0.23 0.23];%universal regional sens
shape_area_threshold_array = [12 8 10 6 5 15 15 15 15];
crop = [0 size(A,2) 0 size(A,1)];
A_doub = A(:,:,2);
A_doub = imadjust(A_doub);

%crop bubbles
A_crop = A_doub;

% %set cropping param and initialize counting matrices
A_sep = zeros(9,4);
A_sep(1,:)= [1,300,1,300];A_sep(2,:)= [1,300,301,600];A_sep(3,:)=[1,300,601,crop(4)-crop(3)];
A_sep(4,:)= [301,600,1,300];A_sep(5,:)=[301,600,301,600];A_sep(6,:)= [301,600,601,crop(4)-crop(3)];
A_sep(7,:)= [601,crop(2)-crop(1),1,300];A_sep(8,:)=[601,crop(2)-crop(1),301,600];
A_sep(9,:)= [601,crop(2)-crop(1),601,crop(4)-crop(3)];
counted = zeros(9,1);
centers_all = [];

for i = 1:9
    %slice image into 9 pieces and make into binary from specificed
    %sensitvity
    binary_sensitivity = sens_array(i);
    shape_area_threshold = shape_area_threshold_array(i);
    A1 = A_crop(A_sep(i,3):A_sep(i,4),A_sep(i,1):A_sep(i,2));
    A2 = imbinarize(A1,'adaptive','Sensitivity',binary_sensitivity);    
    
    %find centres of bright dots
    cc = bwconncomp(A2,26);
    s = regionprops('table',cc,'Area','Centroid');
    idx = find([s.Area] > shape_area_threshold); 
    BW2 = ismember(labelmatrix(cc), idx);  
    ss  = regionprops('table',BW2,'Centroid');
    centers = ss.Centroid;
    
    %count bright dots and visualize them
    radii = 3*ones(size(centers,1),1);
    fn = 'collage'+string(i)'+'.jpg';
    counted(i,1) = size(radii, 1);
    centers_all = [centers_all;(centers + [A_sep(i,1)-1,A_sep(i,3)-1])];
end


% extend orignal picture by 10 pixels to avoid NaN in next step
expand_each_side_by = cproxpixel;
A_doub_expanded = 256*ones(size(A_doub,1)+expand_each_side_by*2, ...
    size(A_doub,2)+expand_each_side_by*2);
A_doub_expanded(expand_each_side_by+1:expand_each_side_by+size(A_doub,1), ...
    expand_each_side_by+1:expand_each_side_by+size(A_doub,2)) = A_doub;
    
centers_all_rounded = round(centers_all);
centers_all_rounded_expaneded = centers_all_rounded+cproxpixel;

% save cropped images centred around centers_all_rounded
if storeImgEachDot
    if storeInPredictionFolder
        fullDirectory = [pwd '/Prediction'];
        if isfolder(fullDirectory)
            rmdir(fullDirectory,'s');
        end
    else
        fullDirectory = [pwd '/' input_img '_imgEachDot'];
    end
    mkdir(fullDirectory);
    for i = 1:size(centers_all_rounded_expaneded,1)
        imgEachDot = imcrop(A_doub_expanded, ...
            [centers_all_rounded_expaneded(i,1)-cproxpixel ...
            centers_all_rounded_expaneded(i,2)-cproxpixel ...
            cproxpixel*2 cproxpixel*2]);
        thisFileName = sprintf('Dot #%5d.png', i);
        fullFileName = fullfile(fullDirectory, thisFileName);
        imwrite(uint8(imgEachDot), fullFileName);
    end
end


% show original tif with counted dots overlay
radii_all = 3*ones(size(centers_all,1),1);

% Using model 'net' to predict # of dots in each imgEachDot
predictionDatasetPath = [pwd '/Prediction'];

imdsPred = imageDatastore(predictionDatasetPath, ...
    'IncludeSubfolders',true);

Pred = classify(net,imdsPred);
doublePred = double(Pred) - 1; % adjust for automatic increment upon using double()

% Label counted signals on original pic
centers_all_afterNN = bsxfun(@times,centers_all,doublePred);
centers_all_afterNN_nonzero = centers_all_afterNN(any(centers_all_afterNN,2),:);

figure(10)
imshow(A_crop)
radii_all = 3*ones(size(centers_all_afterNN_nonzero,1),1);
viscircles(centers_all_afterNN_nonzero,radii_all);
numSignals = size(centers_all_afterNN_nonzero,1)