%Create the train image, include patch images, mask and heatmap%
%%
cnnInputSize = 100;
slidingWindowStride = cnnInputSize/2;
dataPath = 'D:\various project\root_analysis\rootImage';
savePath = 'D:\various project\root_analysis\rootImage\train';
maskPath = strcat(dataPath,'\mask\');       % ͼ���·��
imgPath = strcat(dataPath,'\root\');       % ͼ���·��
heatMapPath = strcat(dataPath,'\keypoint\heatmap\');       % ͼ���·��
saveMaskPath = strcat(savePath,'\mask\');       
saveImgPath = strcat(savePath,'\image\');
saveHeatMapPath = strcat(savePath,'\heatMap\');
maskDir  = dir([maskPath '*.jpg']); % ���������ļ�
%%
trainImgId = 1;
for j = 1:length(maskDir)          % �����ṹ��Ϳ���һһ����ͼƬ��
    rootImg = imread([imgPath maskDir(j).name]); %��ȡÿ��ͼƬ
    rootMask = imread([maskPath maskDir(j).name]); %��ȡÿ��ͼƬ
    heatMap = imread([heatMapPath maskDir(j).name]);
    [height,width,channel] = size(rootImg);
    %��һ  ��ֵ��
    rootMask=double(rgb2gray(rootMask))./255; 
    thresh = graythresh(rootMask);     %�Զ�ȷ����ֵ����ֵ
    rootMask = im2bw(rootMask,0.95);       %��ͼ���ֵ��
    %�ָ����
    left_top_h=1;left_top_w=1;right_bottom_h=1;right_bottom_w=1;
    while(1)
        right_bottom_h = left_top_h + cnnInputSize -1;
        if right_bottom_h > height
            break;
        end
        rowRootSum = 0;
        while(1)
            right_bottom_w = left_top_w + cnnInputSize -1;
            if right_bottom_w>width
                break;
            end
            tempImg = rootImg(left_top_h:right_bottom_h,left_top_w:right_bottom_w,:);
            tempMask = rootMask(left_top_h:right_bottom_h,left_top_w:right_bottom_w);
            tempHeatMap = heatMap(left_top_h:right_bottom_h,left_top_w:right_bottom_w,:);
            imwrite(tempImg,[strcat(saveImgPath, [num2str(trainImgId) '_']),strcat(maskDir(j).name(1:end-4), '.png')]);
            imwrite(tempMask,[strcat(saveMaskPath, [num2str(trainImgId) '_']), strcat(maskDir(j).name(1:end-4), '.png')]);
            imwrite(tempHeatMap,[strcat(saveHeatMapPath, [num2str(trainImgId) '_']), strcat(maskDir(j).name(1:end-4), '.png')]);
            trainImgId = trainImgId + 1;
            %����mask��û��(����)���򻬶�����cnnInputSize
            rowRootSum = rowRootSum + sum(tempMask(:));
            if sum(tempMask(:))<10
                left_top_w = left_top_w + cnnInputSize;
            else
                left_top_w = left_top_w + slidingWindowStride;
            end
        end
        left_top_w = 1;
        if rowRootSum < 600  %һ�е��ܸ�������ʱ
            left_top_h = left_top_h + cnnInputSize;
        else
             left_top_h = left_top_h + slidingWindowStride;
        end
    end
end


