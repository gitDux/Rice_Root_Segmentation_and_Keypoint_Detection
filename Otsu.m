%%
cnnInputSize = 100;
slidingWindowStride = cnnInputSize/2;
dataPath = 'D:\various project\root_analysis\rootImage';
otsuPath = 'D:\various project\root_analysis\rootImage\Otsu\';
maskPath = strcat(dataPath,'\mask\');       % ͼ���·��
imgPath = strcat(dataPath,'\root\');       % ͼ���·��
imgDir  = dir([imgPath '*.jpg']); % ���������ļ�
maskDir  = dir([maskPath '*.jpg']); % ���������ļ�
%%
for j = 1:length(imgDir)          % �����ṹ��Ϳ���һһ����ͼƬ��
    rootImg = imread([imgPath imgDir(j).name]); %��ȡÿ��ͼƬ
    [height,width,channel] = size(rootImg);    
    root_medium = medfilt2(rgb2gray(rootImg),[4,4]);  
    thresh = graythresh(root_medium);     %�Զ�ȷ����ֵ����ֵ
    root_otsu = im2bw(root_medium,thresh);
    
    imwrite(root_otsu,[otsuPath,strcat(imgDir(j).name(1:end-4), '.png')]);
end
%%
metrics = [0 0 0];
for j = 1:length(maskDir)          % �����ṹ��Ϳ���һһ����ͼƬ��
    rootImg = imread([imgPath maskDir(j).name]); %��ȡÿ��ͼƬ
    rootMask = imread([maskPath maskDir(j).name]); %��ȡÿ��ͼƬ
    [height,width,channel] = size(rootImg);
    %��һ  ��ֵ��
    rootMask=double(rgb2gray(rootMask))./255;   
    rootMask = im2bw(rootMask,0.95);       %��ͼ���ֵ��
    
    root_medium = medfilt2(rgb2gray(rootImg),[4,4]);  
    thresh = graythresh(root_medium);     %�Զ�ȷ����ֵ����ֵ
    root_otsu = im2bw(root_medium,thresh);
    result = resultEvaluate(root_otsu, rootMask);
    metrics = metrics + result;
    
    imwrite(root_otsu,[otsuPath,strcat(maskDir(j).name(1:end-4), '.png')]);
    imwrite(rootMask,[maskPath,strcat(maskDir(j).name(1:end-4), '.png')]);
end
metric = metrics/length(maskDir) 
%%
%����pixel acc, dsc, iou
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

