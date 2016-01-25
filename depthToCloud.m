function [pcloud, distance] = depthToCloud(depth, topleft)
% depthToCloud.m - Convert depth image into 3D point cloud
% Author: Liefeng Bo and Kevin Lai
%
% Input: 
% depth - the depth image
% topleft - the position of the top-left corner of depth in the original depth image. Assumes depth is uncropped if this is not provided
%
% Output:
% pcloud - the point cloud, where each channel is the x, y, and z euclidean coordinates respectively. Missing values are NaN.
% distance - euclidean distance from the sensor to each point
%

if nargin < 2
    topleft = [1 1];
end

depth= double(depth);
depth(depth == 0) = nan;

% RGB-D camera constants
%center = [320 240];

[imh, imw] = size(depth);
center=[floor(imh/2),floor(imw/2)];
constant = 570.3;

% convert depth image to 3d point clouds
pcloud = zeros(imh,imw,3);
xgrid = ones(imh,1)*(1:imw) + (topleft(1)-1) - center(1);
ygrid = (1:imh)'*ones(1,imw) + (topleft(2)-1) - center(2);
pcloud(:,:,1) = xgrid.*depth/constant;
pcloud(:,:,2) = ygrid.*depth/constant;
pcloud(:,:,3) = depth;
distance = sqrt(sum(pcloud.^2,3));
