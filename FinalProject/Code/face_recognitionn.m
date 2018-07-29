R=load_database();
IM=imread('pics\facePics\temp\s1\2d.png');
IM=rgb2gray(IM);
IM=reshape(IM,10304,1);

randimage=round(400*rand(1,1));            % for testing random images
r=R(:,randimage);                          % test image
v=R(:,[1:randimage-1 randimage+1:end]);           % all of our training images in v
N=20;                               % Unique for each image

%removing mean
newMat=uint8(ones(1,size(v,2))); 
meanImage=uint8(mean(v,2));                 % m is the mean of all images.
meanRemoved=v-uint8(single(meanImage)*single(newMat));   % vzm is v with the mean removed. 
%eigenface
L=single(meanRemoved)'*single(meanRemoved);
[eigenFace,D]=eig(L);
eigenFace=single(meanRemoved)*eigenFace;
eigenFace=eigenFace(:,end:-1:end-(N-1));                         % Pick the eignevectors corresponding to the 10 largest eigenvalues. 
%signatures
ourSignatures=zeros(size(v,2),N);
for i=1:size(v,2)
    ourSignatures(i,:)=single(meanRemoved(:,i))'*eigenFace;  % Each row in cv is the signature for one image.
end

%recognition
subplot(121); 
imshow(reshape(IM,112,92));
                  %112,92
subplot(122);
p=IM-meanImage;                              % Subtract the mean
s=single(p)'*eigenFace;
ourMin=[];
for i=1:size(v,2)
    ourMin=[ourMin,norm(ourSignatures(i,:)-s,2)];  % norm of signatures - (eigenvectors*s)
    if(rem(i,20)==0),imshow(reshape(v(:,i),112,92)),end;
    drawnow;
end
[a,i]=min(ourMin);
"sumImage%f.pgm",min(ourMin)
ourmin=min(ourMin);
fprintf("%f \n",min(ourMin));
subplot(122);
imshow(reshape(v(:,i),112,92));
if (ourmin < 3000000)
priorityEigen=10;
end
if ((ourmin>3000001)&&(ourmin < 5000000))
priorityEigen=8;
end
if ((ourmin>5000001)&&(ourmin < 7000000))
priorityEigen=6;
end
if ((ourmin>7000001)&&(ourmin < 9000000))
priorityEigen=4;
end
if (ourmin > 9000000)
priorityEigen=2;
end

%   csce 473 spring 2017
%   semester project

% Specify the folder where the files live.



baseImagePath = 'pics\facePics\orl_faces\';
dirList  = dir(baseImagePath);
dirList  = dirList([dirList.isdir]);  % Folders only

groupMatrix = { 's1'; 
                's2';
                's3';
                's4';
                's5';
                's6'};
 
    %store training set data
    trainingResult = []; 

    %store current testing set data
    testingResult = [];
for iDir = 3 : 8

    aDir = fullfile(baseImagePath, dirList(iDir).name);
    fprintf('Processing training set: %s\n', aDir);
         
    %knn classifier setup
    
        
    
    sumImage = readTrainingSet( aDir );
    %figure;
    %imshow(sumImage);
    
    
    OutImageDir = 'pics\facePics\Output';
    newPath ='pics\facePics\temp'
    baseOutputName = sprintf('sumImage%d.pgm', iDir - 2); 
    fullFileName = fullfile(OutImageDir, baseOutputName); 
    imwrite(sumImage, fullFileName);
    
    %baseline offset
    %for jDir = 3 : 8
        
        bDir = fullfile(baseImagePath, dirList(iDir).name);
        fprintf('Processing testing set: %s\n', bDir);
        trainingResult = [trainingResult; calcGei(sumImage, bDir, '7.pgm')];
    
    %end
    
    
    for jDir = 3 : 8
        
        bDir = fullfile(baseImagePath, dirList(jDir).name);
        fprintf('Processing testing set: %s\n', bDir);
        
       % if (jDir == 3)  % subject 1                 %swap comments for files names to test normal faces
       %     testFileName = '1.pgm';
       % elseif(jDir == 5)  %subject 3
       %     testFileName = '3.pgm';
       % elseif(jDir == 8)  %subject 6
       %     testFileName = '6.pgm';
      %  else
       %     testFileName = '8.pgm';
       % end
        
        testFileName = '1.pgm';           %swap here for
        %obstructed faces
        
        
        testingResult = [testingResult; calcGei(sumImage, bDir, testFileName)];     %for normal face testing
       howClose = abs((testingResult(1,2)-100)+200);
            if (howClose < 5)
            prioritySum=9;
            elseif ((howClose>14)&& (howClose < 25))
            prioritySum=7;
            elseif ((howClose>26)&&(howClose < 35))
            prioritySum=5;
            elseif ((howClose>36)&&(howClose < 45))
            prioritySum=3;
            else
            prioritySum=1;
            end
   end
    
   
    
end




if(prioritySum<priorityEigen)
    imshow(reshape(v(:,i),112,92));
    %fprintf('used eigenface');
else
     disp('testing sets');
    disp(iDir - 2);
    disp('training matrix');
    disp(trainingResult);
    disp('sample');
    disp(testingResult);
    class = knnclassify(testingResult, trainingResult, groupMatrix, 2, 'cityblock');
    disp('result');
    disp(class);
   % fprintf('used mean finder');
end


function [ sumImageBlur ] = readTrainingSet( myFolder )
    filePattern = fullfile(myFolder, '*.pgm'); % Change to whatever pattern you need.
    theFiles = dir(filePattern);          
    for k = 1 : length(theFiles) - 1           %skip image 10 since that is the test image
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(myFolder, baseFileName);
        fprintf(1, 'Now reading %s\n', fullFileName);
        
        % Now do whatever you want with this file name,
        % such as reading it in as an image array with imread()

        currentImage = imread(fullFileName);  %load next image
        %imshow(currentImage);
        %figure;
        %noise removal
        currentImageBW = medfilt2(currentImage, [7 7]);

        %cobvert to bw
        
        currentImageBW = im2bw(currentImageBW, .6);
        %currentImageBW = imbinarize(currentImageBW,'adaptive','ForegroundPolarity','dark','Sensitivity',.5);
        %imshow(currentImageBW);
        %imshowpair(currentImage,currentImageBW,'montage');
        %sum pixels of all images into a single image
        if(k == 1)
            sumImage = currentImageBW;
        else
            sumImage = sumImage + currentImageBW;
        end

    end
    %figure;
    %average pixels
    sumImage = sumImage / length (theFiles);
    figure;
    %imshow(sumImage);
    figure;
    %apply gaussian blur 
    sumImageBlur = imgaussfilt(sumImage, .5);
    figure;
    %imshow(sumImageBlur);

    %imshowpair(sumImage,sumImageBlur,'montage');

%     
end

function [ result ] = calcGei( trainingGeiImage, testImagePath, fileName )
    %read test image for current folder and process
    baseFileName = fileName;
    fullFileName = fullfile(testImagePath, baseFileName);
    testImage = imread(fullFileName);
    %imshow(testImage);
    testImageBW = medfilt2(testImage, [7 7]);

    %cobvert to bw
    testImageBW = double(imbinarize(testImageBW, 'adaptive','ForegroundPolarity','dark','Sensitivity',.5));
    %imshow(testImageBW);
    testImageBlur = imgaussfilt(testImageBW, .5);
    %imshow(testImageBlur);
    %figure;
    %imshowpair(testImageBW,testImageBlur,'montage');

    %get percetage difference between training set and test image
    imDiff = trainingGeiImage - testImageBlur;
    imSum = trainingGeiImage + testImageBlur;  
    percentDiff = abs(200 * mean(imDiff(:)) / mean(imSum(:)));
    disp(percentDiff);

    K = imabsdiff(trainingGeiImage, testImageBlur);
    %figure;
    %imshow(K);

    %get direct difference
    mult = immultiply(trainingGeiImage, testImageBlur);
    directDiff = 100 * abs(mean(imDiff(:))) / sqrt (mean (mult(:)));
    %disp(directDiff);
    result = [percentDiff directDiff];
    %result = [percentDiff];
    
end
