function [ padx ] = padadd( datax, szpad)
%UNTITLED Summary of this function goes here
% Pad with the boundary of the two-dimensional matrix
szpadh = szpad(1);
szpadw = szpad(2);
[h, w, num] = size(datax);
padx = zeros(h+szpadh, w+szpadw, num);
padx(floor(szpadh/2)+1:floor(szpadh/2)+h, floor(szpadw/2)+1:floor(szpadw/2)+w, :) = datax(:,:,:);
% for n = 1:num
%     padx(floor(szpadh/2)+1:floor(szpadh/2)+h, floor(szpadw/2)+1:floor(szpadw/2)+w,n) = datax(:,:,n);
%     %     for i = floor(szpadh/2):-1:1
%     %         padx(i,:,n) = padx(floor(szpadh/2)+1,:,n); 
%     %         padx(end+1-i,:,n) = padx(floor(szpadh/2)+h,:,n);
%     %     end
%     %     padx(end-floor(szpadh/2),:,n) = padx(floor(szpadh/2)+h,:,n);
%     %     for i = floor(szpadw/2):-1:1
%     %         padx(:,i,n) = padx(:,floor(szpadw/2)+1,n);
%     %         padx(:,end+1-i,n) = padx(:,floor(szpadw/2)+w,n);
%     %     end
%     %     padx(:,end-floor(szpadw/2),n) = padx(:,floor(szpadw/2)+w,n);
% end
end

