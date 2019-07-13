%%
cnnInputSize = 100;
slidingWindowStride = cnnInputSize/2;
dataPath = 'D:\various project\root_analysis\rootImage';
otsuPath = 'D:\various project\root_analysis\rootImage\Otsu\';
maskPath = strcat(dataPath,'\mask\');       % 图像库路径
imgPath = strcat(dataPath,'\root\');       % 图像库路径
imgDir  = dir([imgPath '*.jpg']); % 遍历所有文件
maskDir  = dir([maskPath '*.jpg']); % 遍历所有文件
%%
for j = 1:length(imgDir)          % 遍历结构体就可以一一处理图片了
    rootImg = imread([imgPath imgDir(j).name]); %读取每张图片
    [height,width,channel] = size(rootImg);    
    root_medium = medfilt2(rgb2gray(rootImg),[4,4]);  
    thresh = graythresh(root_medium);     %自动确定二值化阈值
    root_otsu = im2bw(root_medium,thresh);
    
    imwrite(root_otsu,[otsuPath,strcat(imgDir(j).name(1:end-4), '.png')]);
end
%%
metrics = [0 0 0];
for j = 1:length(maskDir)          % 遍历结构体就可以一一处理图片了
    rootImg = imread([imgPath maskDir(j).name]); %读取每张图片
    rootMask = imread([maskPath maskDir(j).name]); %读取每张图片
    [height,width,channel] = size(rootImg);
    %归一  二值化
    rootMask=double(rgb2gray(rootMask))./255;   
    rootMask = im2bw(rootMask,0.95);       %对图像二值化
    
    root_medium = medfilt2(rgb2gray(rootImg),[4,4]);  
    thresh = graythresh(root_medium);     %自动确定二值化阈值
    root_otsu = im2bw(root_medium,thresh);
    result = resultEvaluate(root_otsu, rootMask);
    metrics = metrics + result;
    
    imwrite(root_otsu,[otsuPath,strcat(maskDir(j).name(1:end-4), '.png')]);
    imwrite(rootMask,[maskPath,strcat(maskDir(j).name(1:end-4), '.png')]);
end
metric = metrics/length(maskDir) 
%%
%计算pixel acc, dsc, iou
function metrics = resultEvaluate(Img, Mask)    
    [m, n] = size(Img);
    pixel = Img==Mask;
    intersection = Img&Mask;
    union = Img|Mask;
    pixel_acc = sum(pixel(:))/(m * n);
    DSC = 2*sum(intersection(:))/(sum(intersection(:))+sum(union(:)));
    IoU = sum(intersection(:))/sum(union(:));
    metrics = [pixel_acc DSC IoU];
end

