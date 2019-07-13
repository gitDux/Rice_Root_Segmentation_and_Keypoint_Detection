%Create the train image, include patch images, mask and heatmap%
%%
cnnInputSize = 100;
slidingWindowStride = cnnInputSize/2;
dataPath = 'D:\various project\root_analysis\rootImage';
savePath = 'D:\various project\root_analysis\rootImage\train';
maskPath = strcat(dataPath,'\mask\');       % 图像库路径
imgPath = strcat(dataPath,'\root\');       % 图像库路径
heatMapPath = strcat(dataPath,'\keypoint\heatmap\');       % 图像库路径
saveMaskPath = strcat(savePath,'\mask\');       
saveImgPath = strcat(savePath,'\image\');
saveHeatMapPath = strcat(savePath,'\heatMap\');
maskDir  = dir([maskPath '*.jpg']); % 遍历所有文件
%%
trainImgId = 1;
for j = 1:length(maskDir)          % 遍历结构体就可以一一处理图片了
    rootImg = imread([imgPath maskDir(j).name]); %读取每张图片
    rootMask = imread([maskPath maskDir(j).name]); %读取每张图片
    heatMap = imread([heatMapPath maskDir(j).name]);
    [height,width,channel] = size(rootImg);
    %归一  二值化
    rootMask=double(rgb2gray(rootMask))./255; 
    thresh = graythresh(rootMask);     %自动确定二值化阈值
    rootMask = im2bw(rootMask,0.95);       %对图像二值化
    %分割滑动窗
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
            %若是mask中没有(很少)根则滑动整个cnnInputSize
            rowRootSum = rowRootSum + sum(tempMask(:));
            if sum(tempMask(:))<10
                left_top_w = left_top_w + cnnInputSize;
            else
                left_top_w = left_top_w + slidingWindowStride;
            end
        end
        left_top_w = 1;
        if rowRootSum < 600  %一行的总根数很少时
            left_top_h = left_top_h + cnnInputSize;
        else
             left_top_h = left_top_h + slidingWindowStride;
        end
    end
end


