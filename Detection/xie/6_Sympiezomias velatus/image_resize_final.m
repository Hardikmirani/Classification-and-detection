clear all; 
close all; 
clc;
dr=uigetdir();
list=dir([dr,'\*.*']);
for imc=1:(length(list)-2)

%     cd all
img1{imc}=imread(strcat(dr,'\',list(imc+2).name));
img = imresize(img1{imc}, [227 227]);
% img=rgb2gray(img);
% f=[num2str(imc) '.jpg'];

% % I1=imread('cifuna1.jpg');
% % [J, rect] = imcrop(img1{imc});
% [J, rect] = imcrop(img1{imc});
%  I2 = imresize(J,[227 227]); 
% % I2 = imresize(img1{imc},[300 300]); 
% In=rgb2gray(I2); % use if the image containing RGB value 3
% figure;imshow(In);
% imwrite(In,'yi.jpg') ;
% figure;imshow(I2);
% imwrite(I2,'y.jpg') ;
%imwrite(I2,strcat('Img',num2str(imc),'.jpg'));
imwrite(img,strcat('6_Sympiezomias velatus',num2str(imc),'.jpg'));
end