

% function [] = test(x,img)

clc;

ext_size = 50;
store = 0;      % 1:training 0:Testing(digit recognition)


%clear all;
%-----------backup the variables need to store the patterns in library------- 
ret = exist('backup.mat', 'file');
if ret == 2
    load('backup')
else
    index=1;
    complete = 0;
    store_sample = zeros(ext_size,ext_size,10);
    zero_forg = zeros(2500,2);zero_frg_cnt = zeros(1,10);
    one_forg = zeros(2500,2);one_frg_cnt = zeros(1,10);
    two_forg = zeros(2500,2);two_frg_cnt = zeros(1,10);
    three_forg = zeros(2500,2);three_frg_cnt = zeros(1,10);
    four_forg = zeros(2500,2);four_frg_cnt = zeros(1,10);
    five_forg = zeros(2500,2);five_frg_cnt = zeros(1,10);
    six_forg = zeros(2500,2);six_frg_cnt = zeros(1,10);
    seven_forg = zeros(2500,2);seven_frg_cnt = zeros(1,10);
    eight_forg = zeros(2500,2);eight_frg_cnt = zeros(1,10);
    nine_forg = zeros(2500,2);nine_frg_cnt = zeros(1,10);
    d_1= zeros(1,10); d_2= zeros(1,10); d_3= zeros(1,10); d_4= zeros(1,10);
    d_5= zeros(1,10);d_6= zeros(1,10);d_7= zeros(1,10);d_8= zeros(1,10);
    d_9= zeros(1,10);d_0= zeros(1,10);
end
%------------------------------------------------------------------------


%---------training library---------------------
 for img = {'zero.jpg','one.jpg', 'two.jpg', 'three.jpg', 'four.jpg', 'five.jpg', 'six.jpg', 'seven.jpg', 'eight.jpg', 'nine.jpg'}
  image = img{1};
%----------------------------------------------

%  image = 'two.jpg';         %Image to be identify 


%---------------------------Image Capturing------------------------------
x = imread(image);
I = rgb2gray(x);
%figure, imshow(BW);

% newRange = 4;  %// choose the new maximum. (new minimum always at 0.0)
% imgMin = double(min(I(:)));
% imgMax = double(max(I(:)));
% I = (image - imgMin) / (imgMax - imgMin) * newRange;

BW = im2bw(I,0.4);
BW = imresize(BW, [200 200]);
get_size = size(BW);
%figure, imshow(BW);

rows = zeros(get_size(1), 1);
cols = zeros(1, get_size(2));
rows_extract = zeros(get_size(1), 1);
rw_index_ext = zeros(get_size(1), 1);
cols_extract = zeros(1, get_size(2));
col_index_ext = zeros(1, get_size(2));
j=1;
k=1;
for i = 1:get_size(1)
    rows(i) = sum(BW(i,:));  % sum of elements in a array
    if rows(i) ~= get_size(2) % check atleast one '0' 
        rows_extract(j) = rows(i);
        rw_index_ext(k) = i;
        j = j + 1;
        k = k + 1;
    end
end 


if j == 1
    fprintf('no number');
    return;
end

% if j == get_size(1)+1
%     fprintf('Not a number');
%     return;
% end


j=1;
k=1;
for i = 1:get_size(2)
    cols(i) = sum(BW(:,i));
    if cols(i) ~= get_size(1)
        cols_extract(j) = cols(i);
        col_index_ext(k) = i;
        j = j + 1;
        k = k + 1;
    end
end 


rows_extract_temp = rows_extract;
rw_index_ext_temp = rw_index_ext;
cols_extract_temp = cols_extract;
col_index_ext_temp = col_index_ext;

rows_extract = rows_extract(rows_extract_temp.*rw_index_ext_temp ~= 0); %trunctating 0s
rw_index_ext = rw_index_ext(rows_extract_temp.*rw_index_ext_temp ~= 0);
cols_extract = cols_extract(cols_extract_temp.*col_index_ext_temp ~= 0);
col_index_ext = col_index_ext(cols_extract_temp.*col_index_ext_temp ~= 0);

extract_rw_size = size(rows_extract);
extract_cl_size = size(cols_extract);

ext_rw_eff = size(rw_index_ext(1):rw_index_ext(extract_rw_size(1)));
ext_cl_eff =  size(col_index_ext(1):col_index_ext(extract_cl_size(2)));

extract_digit = zeros(ext_rw_eff(2), ext_cl_eff(2));
col_indx_start = col_index_ext(1);
for i = 1:ext_cl_eff(2)
    extract_digit(:,i) = BW(rw_index_ext(1):rw_index_ext(extract_rw_size(1)), ...
                    col_indx_start);
    col_indx_start = col_indx_start + 1;
end

new_bw = imresize(extract_digit, [50 50]);
new_bw = im2bw(new_bw,0.5);
figure, imshow(new_bw);

%------------------------------------------------------------------------------
    
% store_sample = zeros(ext_size,ext_size,10);

%---------------captured training images stored in library--------------

if store == 1
    switch image
        case 'zero.jpg'    
            zero_sample = new_bw;
            store_sample(:,:,1,index) = zero_sample;
        case 'one.jpg'
            one_sample = new_bw;
            store_sample(:,:,2,index) = one_sample;
        case 'two.jpg'
            two_sample = new_bw;
              store_sample(:,:,3,index) = two_sample;
        case 'three.jpg'
            three_sample = new_bw;
              store_sample(:,:,4,index) = three_sample;
        case 'four.jpg'
            four_sample = new_bw;
             store_sample(:,:,5,index) = four_sample;
        case 'five.jpg'       
            five_sample = new_bw;
            store_sample(:,:,6,index) = five_sample;
        case 'six.jpg'    
            six_sample = new_bw;
             store_sample(:,:,7,index) = six_sample;
        case 'seven.jpg'    
            seven_sample = new_bw;
             store_sample(:,:,8,index) = seven_sample;
        case 'eight.jpg'    
            eight_sample = new_bw;
             store_sample(:,:,9,index) = eight_sample;
        case 'nine.jpg'    
            nine_sample = new_bw;
             store_sample(:,:,10,index) = nine_sample;
        otherwise
            dump = new_bw;
    end
end
%-----------------------------------------------------------------------

% function [ cnt, frgnd_mat ] = frgnd_cnt( x )

min1=1;min2=1;min3=1;min4=1;min5=1;
min6=1;min7=1;min8=1;min9=1;min0=1;

%-------store foreground pixels and foreground pixel count in libray------
if store == 1
    switch image
        case 'zero.jpg'
%             [zero_frg_cnt,zero_forg] = frgnd_cnt(1,ext_size,store_sample);
            k = 1;
%             zero_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,1,index) == 0
                        zero_forg(k,1,index) = i;
                        zero_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            zero_frg_cnt(index) = k-1;


        case 'one.jpg'
%             [one_frg_cnt,one_forg] = frgnd_cnt(2,ext_size,store_sample);
            k = 1;
            %one_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,2,index) == 0
                        one_forg(k,1,index) = i;
                        one_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            one_frg_cnt(index) = k-1;
        case 'two.jpg'
%             [two_frg_cnt,two_forg] = frgnd_cnt(3,ext_size,store_sample);
            k = 1;
            %two_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,3,index) == 0
                        two_forg(k,1,index) = i;
                        two_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            two_frg_cnt(index) = k-1;
        case 'three.jpg'
%             [three_frg_cnt,three_forg] = frgnd_cnt(4,ext_size,store_sample);
            k = 1;
            %three_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,4,index) == 0
                        three_forg(k,1,index) = i;
                        three_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            three_frg_cnt(index) = k-1;
        case 'four.jpg'
%             [four_frg_cnt,four_forg] = frgnd_cnt(5,ext_size,store_sample);
            k = 1;
            four_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,5,index) == 0
                        four_forg(k,1,index) = i;
                        four_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            four_frg_cnt(index) = k-1;
        case 'five.jpg'
%             [five_frg_cnt,five_forg] = frgnd_cnt(6,ext_size,store_sample);
            k = 1;
            %five_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,6,index) == 0
                        five_forg(k,1,index) = i;
                        five_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            five_frg_cnt(index) = k-1;
        case 'six.jpg'
%             [six_frg_cnt,six_forg] = frgnd_cnt(7,ext_size,store_sample);
            k = 1;
            %six_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,7,index) == 0
                        six_forg(k,1,index) = i;
                        six_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            six_frg_cnt(index) = k-1;
        case 'seven.jpg'
%             [seven_frg_cnt,seven_forg] = frgnd_cnt(8,ext_size,store_sample);
            k = 1;
            %seven_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,8,index) == 0
                        seven_forg(k,1,index) = i;
                        seven_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            seven_frg_cnt(index) = k-1;
        case 'eight.jpg'
%             [eight_frg_cnt,eight_forg] = frgnd_cnt(9,ext_size,store_sample);
            k = 1;
            eight_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,9,index) == 0
                        eight_forg(k,1,index) = i;
                        eight_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            eight_frg_cnt(index) = k-1;
        case 'nine.jpg'
%             [nine_frg_cnt,nine_forg] = frgnd_cnt(10,ext_size,store_sample);
            k = 1;
            %nine_forg = zeros(2500,2);
            for i = 1:ext_size
                for j = 1:ext_size
                    if store_sample(i,j,10,index) == 0
                        nine_forg(k,1,index) = i;
                        nine_forg(k,2,index) = j;
                        k = k + 1;
                    end
                end
            end
            nine_frg_cnt(index) = k-1;
    end
end
%-------------------------------------------------------------------------



if store == 0
    
if complete == 1
    loop = 5;
else
    loop = index;
end
    
%-----foreground calculation of image to be indentify--------------
in_forg = zeros(2500,2);
k = 1;
    for i = 1:ext_size
        for j = 1:ext_size
            if new_bw(i,j) == 0
                in_forg(k,1) = i;
                in_forg(k,2) = j;
                k = k + 1;
            end
        end
    end
in_frg_cnt = k-1;
%-------------------------------------------------------------------


%-----------Jaccard distance calculations---------------------------------

min1=1;min2=1;min3=1;min4=1;min5=1;
min6=1;min7=1;min8=1;min9=1;min0=1;

temp0 = 0;temp1 = 0;temp2 = 0;temp3 = 0;temp4 = 0;temp5 = 0;
temp6 = 0;temp7 = 0;temp8 = 0;temp9 = 0;
for cnt = 1:(loop-1)
p = 0;
fprintf('in')
for i = 1:in_frg_cnt
    for j = 1:zero_frg_cnt(cnt)
        if in_forg(i,1) == zero_forg(j,1,cnt) && in_forg(i,2) == zero_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = zero_frg_cnt(cnt) -p;
d_0(cnt) = 1 - (p / (p + q + r));

if d_0(cnt) < min0
    min0 = d_0(cnt);
end
temp0 = temp0 + d_0(cnt);

p = 0; 
for i = 1:in_frg_cnt
    for j = 1:one_frg_cnt(cnt)
        if in_forg(i,1) == one_forg(j,1,cnt) && in_forg(i,2) == one_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = one_frg_cnt(cnt) -p;
d_1(cnt) = 1 - (p / (p + q + r));

if d_1(cnt) < min1
    min1 = d_1(cnt);
end

temp1 = temp1 + d_1(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:two_frg_cnt(cnt)
        if in_forg(i,1) == two_forg(j,1,cnt) && in_forg(i,2) == two_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = two_frg_cnt(cnt) -p;
d_2(cnt) = 1 - (p / (p + q + r));

if d_2(cnt) < min2
    min2 = d_2(cnt);
end

temp2 = temp2 + d_2(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:three_frg_cnt(cnt)
        if in_forg(i,1) == three_forg(j,1,cnt) && in_forg(i,2) == three_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = three_frg_cnt(cnt) -p;
d_3(cnt) = 1 - (p / (p + q + r));

if d_3(cnt) < min3
    min3 = d_3(cnt);
end

temp3 = temp3 + d_3(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:four_frg_cnt(cnt)
        if in_forg(i,1) == four_forg(j,1,cnt) && in_forg(i,2) == four_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = four_frg_cnt(cnt) -p;
d_4(cnt) = 1 - (p / (p + q + r));

if d_4(cnt) < min4
    min4 = d_4(cnt);
end

temp4 = temp4 + d_4(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:five_frg_cnt(cnt)
        if in_forg(i,1) == five_forg(j,1,cnt) && in_forg(i,2) == five_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = five_frg_cnt(cnt) -p;
d_5(cnt) = 1 - (p / (p + q + r));

if d_5(cnt) < min5
    min5 = d_5(cnt);
end

temp5 = temp5 + d_5(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:six_frg_cnt(cnt)
        if in_forg(i,1) == six_forg(j,1,cnt) && in_forg(i,2) == six_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = six_frg_cnt(cnt) -p;
d_6(cnt) = 1 - (p / (p + q + r));

if d_6(cnt) < min6
    min6 = d_6(cnt);
end

temp6 = temp6 + d_6(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:seven_frg_cnt(cnt)
        if in_forg(i,1) == seven_forg(j,1,cnt) && in_forg(i,2) == seven_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = seven_frg_cnt(cnt) -p;
d_7(cnt) = 1 - (p / (p + q + r));

if d_7(cnt) < min7
    min7 = d_7(cnt);
end

temp7 = temp7 + d_7(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:eight_frg_cnt(cnt)
        if in_forg(i,1) == eight_forg(j,1,cnt) && in_forg(i,2) == eight_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = eight_frg_cnt(cnt) -p;
d_8(cnt) = 1 - (p / (p + q + r));

if d_8(cnt) < min8
    min8 = d_8(cnt);
end

temp8 = temp8 + d_8(cnt);

p = 0;
for i = 1:in_frg_cnt
    for j = 1:nine_frg_cnt(cnt)
        if in_forg(i,1) == nine_forg(j,1,cnt) && in_forg(i,2) == nine_forg(j,2,cnt)
            p = p + 1;
        end
    end
end
q = in_frg_cnt - p;
r = nine_frg_cnt(cnt) -p;
d_9(cnt) = 1 - (p / (p + q + r));

if d_9(cnt) < min9
    min9 = d_9(cnt);
end

temp9 = temp9 + d_9(cnt);

%---------store all distances in log file for%analalys------------------

f_id = fopen('log.txt','wt');
i=0;
fprintf(f_id,'\n--------------reading 1------------------------\n');
for cnt = 1:(loop-1)
for x = [d_0(cnt),d_1(cnt),d_2(cnt),d_3(cnt),d_4(cnt),d_5(cnt),d_6(cnt),d_7(cnt),d_8(cnt),d_9(cnt)]
fprintf(f_id,'%d = %d\n',i,x);
i = i+1;
end
i=0;
fprintf(f_id,'\n--------------reading %d------------------------\n',cnt);
end
% fclose(f_id);
%--------------------------------------------------------------------

end

fprintf(f_id,'\n--------------average------------------------\n');
temp0 = temp0/(loop-1);temp1 = temp1/(loop-1);temp2 = temp2/(loop-1);
temp3 = temp3/(loop-1);temp4 = temp4/(loop-1);temp5 = temp5/(loop-1);
temp6 = temp6/(loop-1);temp7 = temp7/(loop-1);temp8 = temp8/(loop-1);
temp9 = temp9/(loop-1);
j = 0;
max = 1;
max_id = 0;
for y = [temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9]
    if max > y
        max = y;
        max_id = j;
    end
fprintf(f_id,'%d = %d\n',j,y);
j = j+1;
end
fprintf(f_id,'\n--------"%d" DETECTED and VALUE = %d-------\n',max_id,max);
fprintf(f_id,'\n--------------------------------------------------\n');



fprintf(f_id,'\n--------------minimum method--------------------\n');
j = 0;
max = 1;
max_id = 0;
for y = [min0,min1,min2,min3,min4,min5,min6,min7,min8,min9]
    if max > y
        max = y;
        max_id = j;
    end
fprintf(f_id,'%d = %d\n',j,y);
j = j+1;
end
fprintf(f_id,'\n--------"%d" DETECTED and VALUE = %d-------\n',max_id,max);
fprintf(f_id,'\n--------------------------------------------------\n');

fclose(f_id);
%-------------------------------------------------------------------------

end


% 
 end % store end

if store == 1
if index == 11   %library size
    index = 1;
    complete = 1;
else
index = (index + 1);
end

save('backup','index','store_sample','nine_frg_cnt','nine_forg','eight_frg_cnt','eight_forg',...
     'seven_frg_cnt','seven_forg','six_frg_cnt','six_forg','five_frg_cnt','five_forg',...
     'four_frg_cnt','four_forg','three_frg_cnt','three_forg','two_frg_cnt','two_forg',...
      'one_frg_cnt','one_forg','zero_frg_cnt','zero_forg','complete');


end
  
  
   
