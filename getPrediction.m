%Create the train image%
%%
clear all;
cnnInputSize = 100;
slidingWindowStride = cnnInputSize;
dataPath = 'D:\various project\root_analysis\rootImage\test';
savePath = 'D:\various project\root_analysis\rootImage\test';
maskPath = strcat(dataPath,'\mask\');       % 图像库路径
saveRootMaskPath = strcat(savePath,'\RootMask\');             
maskDir  = dir([maskPath '*.png']); % 遍历所有文件
h_num = str2num(maskDir(end).name(1:3));
w_num = str2num(maskDir(end).name(5:7));
root_name = maskDir(end).name(9:end);
h_num=22;w_num=8;
%%
 left_top_h=1;left_top_w=1;right_bottom_h=1;right_bottom_w=1;
for hh = 1:h_num          % 遍历结构体就可以一一处理图片了
    %分割滑动窗
    right_bottom_h = left_top_h + cnnInputSize -1;
    for ww = 1:w_num
        id = (hh-1)*w_num + ww;
        rootImg = imread([maskPath maskDir(id).name]);
        right_bottom_w = left_top_w + cnnInputSize -1;
        rowRootSum = 0;       
        tempImg(left_top_h:right_bottom_h,left_top_w:right_bottom_w,:) = rootImg;           
        left_top_w = left_top_w + cnnInputSize;
    end
    left_top_w = 1;
    left_top_h = left_top_h + cnnInputSize;
end
imwrite(tempImg,strcat(saveRootMaskPath, root_name));

