%Create the train image%
%%
clear all;
cnnInputSize = 100;
slidingWindowStride = cnnInputSize;
dataPath = 'D:\various project\root_analysis\rootImage\test';
savePath = 'D:\various project\root_analysis\rootImage\test';
maskPath = strcat(dataPath,'\mask\');       % ͼ���·��
rootPatchPath = strcat(dataPath,'\image\');       % ͼ���·��
heatmapPatchPath = strcat(dataPath,'\heatMap\');
saveRootMaskPath = strcat(savePath,'\RootMask\');             
maskDir  = dir([maskPath '*.png']); % ���������ļ�
h_num = str2num(maskDir(end).name(1:3));
w_num = str2num(maskDir(end).name(5:7));
root_name = maskDir(end).name(9:end);
h_num=22;w_num=8;

%%
 left_top_h=1;left_top_w=1;right_bottom_h=1;right_bottom_w=1;
for hh = 1:h_num          % �����ṹ��Ϳ���һһ����ͼƬ��
    %�ָ����
    right_bottom_h = left_top_h + cnnInputSize -1;
    for ww = 1:w_num
        id = (hh-1)*w_num + ww;
        patchMask = imread([maskPath maskDir(id).name]);%��ӳ�䵽ԭͼҲ����
        patchRoot = imread([rootPatchPath maskDir(id).name]);%��ӳ�䵽ԭͼҲ����
        temp = imread([heatmapPatchPath maskDir(id).name]);
        patchHeatmap = imresize(temp,100/101,'bilinear');%����100����Ϊcnn������
                
        %rootImg = imread([maskPath maskDir(id).name]);
        right_bottom_w = left_top_w + cnnInputSize -1;
        rowRootSum = 0;       
        tempMask(left_top_h:right_bottom_h,left_top_w:right_bottom_w,:) = patchMask;
        tempHeatmap(left_top_h:right_bottom_h,left_top_w:right_bottom_w,:) = patchHeatmap;
        tempRoot(left_top_h:right_bottom_h,left_top_w:right_bottom_w,:) = patchRoot;
        %tempImg(left_top_h:right_bottom_h,left_top_w:right_bottom_w,:) = rootImg;           
        left_top_w = left_top_w + cnnInputSize;
    end
    left_top_w = 1;
    left_top_h = left_top_h + cnnInputSize;
end
[rootHeatmap, maskHeatmap] = mergeHeatmap(tempRoot,tempMask,tempHeatmap,200);
subplot(1,2,1);imshow(rootHeatmap);
subplot(1,2,2);imshow(maskHeatmap);
imwrite(rootHeatmap,strcat(saveRootMaskPath, ['rootHeatmap' root_name]));
imwrite(maskHeatmap,strcat(saveRootMaskPath, ['maskHeatmap' root_name]));
%%
function [rootHeatmap, maskHeatmap]= mergeHeatmap(root, mask, heatmap, threshhold)
    keyPointMask(:,:) = uint8((heatmap(:,:,1) > threshhold) | (heatmap(:,:,2) > threshhold) | (heatmap(:,:,3) > threshhold));
    rootHeatmap(:,:,1) = (1 - keyPointMask(:,:)).*root(:,:,1) + keyPointMask(:,:).*heatmap(:,:,1);
    rootHeatmap(:,:,2) = (1 - keyPointMask(:,:)).*root(:,:,2) + keyPointMask(:,:).*heatmap(:,:,2);
    rootHeatmap(:,:,3) = (1 - keyPointMask(:,:)).*root(:,:,3) + keyPointMask(:,:).*heatmap(:,:,3);

    maskHeatmap(:,:,1) = (1 - keyPointMask(:,:)).*mask(:,:) + keyPointMask(:,:).*heatmap(:,:,1);
    maskHeatmap(:,:,2) = (1 - keyPointMask(:,:)).*mask(:,:) + keyPointMask(:,:).*heatmap(:,:,2);
    maskHeatmap(:,:,3) = (1 - keyPointMask(:,:)).*mask(:,:) + keyPointMask(:,:).*heatmap(:,:,3);
end

