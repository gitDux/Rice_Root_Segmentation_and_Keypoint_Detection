%Create the train image%
%%
cnnInputSize = 100;
slidingWindowStride = cnnInputSize;
dataPath = 'D:\various project\root_analysis\rootImage\test';
savePath = 'D:\various project\root_analysis\rootImage\test';
maskPath = strcat(dataPath,'\mask\');       % ͼ���·��
imgPath = strcat(dataPath,'\root\');       % ͼ���·��
saveMaskPath = strcat(savePath,'\mask\');       
saveImgPath = strcat(savePath,'\image\');       
imgDir  = dir([imgPath '*.jpg']); % ���������ļ�
%%
for j = 1:length(imgDir)          % �����ṹ��Ϳ���һһ����ͼƬ��
    trainImg_h_Id = 1;
    rootImg = imread([imgPath imgDir(j).name]); %��ȡÿ��ͼƬ
    [height,width,channel] = size(rootImg);
    %�ָ����
    left_top_h=1;left_top_w=1;right_bottom_h=1;right_bottom_w=1;
    while(1)
        trainImg_w_Id = 1;
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
            imwrite(tempImg,[strcat(saveImgPath, [[num2str_3(trainImg_h_Id) '_'] [num2str_3(trainImg_w_Id) '_']]),strcat(imgDir(j).name(1:end-4), '.png')]);
            trainImg_w_Id = trainImg_w_Id + 1;
            left_top_w = left_top_w + cnnInputSize;
        end
        left_top_w = 1;
        left_top_h = left_top_h + cnnInputSize;
        trainImg_h_Id = trainImg_h_Id + 1;
    end
end

%%
function str=num2str_3(num)% 
            bai = num2str(fix(num/100));
            shi = num2str(fix((num-str2num(bai)*100)/10));
            ge = num2str(fix((num-str2num(bai)*100-str2num(shi)*10)/1));
            str = [bai shi ge];
end
